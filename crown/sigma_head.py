"""v11 sigma head (--sigma-head flux-guidance, default off): a per-band guidance correction for flux, fitted on the
internal holdout and delivered as an exact rank <= 16 LoRA delta on `time_in.out_layer`.

Law (Comfy 694815f, comfy/ldm/flux/model.py forward_orig). The modulation vector is additive,

    vec = time_in(emb(t)) + guidance_in(emb(g)) + vector_in(y),

with emb = timestep_embedding(., 256) (time_factor 1000; the timestep is sigma, ModelSamplingFlux.timestep) and
time_in / guidance_in = MLPEmbedder (in_layer -> SiLU -> out_layer). The evaluator fixes g = 3.5
(image_flow_adapter FAMILIES["flux"].guidance) and scores only the 16 band midpoints sigma_b. Let
h_b = SiLU(time_in.in_layer(emb(sigma_b))) be the input of time_in.out_layer and
Delta_b = guidance_in(emb(g_b)) - guidance_in(emb(3.5)). Any dW with dW h_b = Delta_b for all b makes the model at
g = 3.5 compute, at band b, exactly the vec it computes at g_b. With H = [h_b] (16 columns, rank 16 generically) the
minimum-norm solution dW = [Delta_b] H^+ has rank <= 16: a LoRA delta on time_in.out_layer, rank-concatenated onto the
candidate's own factors there (or a new key when the module is not a LoRA target). Bands that keep 3.5 have
Delta_b = 0 and are constrained to stay exactly as they were.

Fit rule (closed form; the grid is the only constant): per band, the grid value with the lowest mean paired loss
difference to g = 3.5 on the holdout's fit draws is adopted only if that gain exceeds its own paired standard error;
otherwise the band keeps 3.5 (shrinkage to the evaluator's value).

dtype chain replicated (model_base._apply_model, forward_orig): the timestep is fp32 and its embedding is cast to the
model dtype; the guidance cond is cast to the model dtype BEFORE it is embedded (convert_tensor), so the embedding
multiplies 1000 * g in that dtype. The algebra runs in float64 on the candidate-merged fp32 weights.
"""

import math

import numpy as np
import torch

from . import contract as C
from .scoring import ScoreReport

MODE = "flux-guidance"
TARGET = "time_in.out_layer"        # the Linear that carries the correction
EMB_DIM = 256                       # MLPEmbedder(in_dim=256) for time_in and guidance_in
DEFAULT_GRID = "1.5,2,2.5,3"


def timestep_embedding_ref(t, dim, max_period=10000, time_factor=1000.0):
    """comfy.ldm.flux.layers.timestep_embedding at 694815f, expression for expression. Used only where Comfy is not
    importable (CPU mocks); the trainer calls Comfy's own function."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def embedding_fn():
    try:
        from comfy.ldm.flux.layers import timestep_embedding
        return timestep_embedding, "comfy.ldm.flux.layers"
    except Exception:  # noqa: BLE001 - no Comfy (CPU mock)
        return timestep_embedding_ref, "crown.sigma_head reference copy"


def with_guidance(cond, g):
    """node_helpers.conditioning_set_values(cond, {"guidance": g}) - what the evaluator and encode_prompt call."""
    try:
        import node_helpers
        return node_helpers.conditioning_set_values(cond, {"guidance": g})
    except ImportError:
        out = []
        for t in cond:
            n = [t[0], t[1].copy()]
            n[1]["guidance"] = g
            out.append(n)
        return out


def parse_grid(text, g_eval):
    vals = sorted({float(x) for x in str(text or "").split(",") if x.strip()} - {float(g_eval)})
    if not vals or not all(math.isfinite(v) and v > 0 for v in vals):
        raise ValueError(f"--sigma-head-grid {text!r}: need positive guidance values other than {g_eval}")
    return vals


def flux_structure(dm):
    """(ok, reason): the diffusion model has flux's additive time/guidance embedders."""
    params = getattr(dm, "params", None)
    if not getattr(params, "guidance_embed", False):
        return False, "the model has no guidance embedding (guidance_embed is False)"
    for name in ("time_in", "guidance_in"):
        m = getattr(dm, name, None)
        if m is None or not all(hasattr(m, a) for a in ("in_layer", "out_layer")):
            return False, f"{name} is not an MLPEmbedder (in_layer -> SiLU -> out_layer)"
        if getattr(m.in_layer, "weight", None) is None or m.in_layer.weight.shape[1] != EMB_DIM:
            return False, f"{name}.in_layer does not take the {EMB_DIM}-d timestep embedding"
    return True, ""


class HeadState:
    """Per-run state of the sigma head. `active` only for flux with the embedder structure; `extra` holds the weight
    wrappers attached for correction modules that are not LoRA targets (so the hook path can score them)."""

    def __init__(self, mode):
        self.mode = mode
        self.active = False
        self.reason = None
        self.grid = []
        self.noises = 1
        self.g_eval = None
        self.extra = {}

    def score_counts(self, twin):
        """(hook scores, twin scores) the stage may need, with nothing reused: base + grid on the fit draws, the patched
        state on the fit draws (reproduction), patched and unpatched on the fresh draws (hook or twin)."""
        n_hook = len(self.grid) + 2 + (0 if twin else 2)
        return n_hook, (2 if twin else 0)


def prepare(cfg, model, lora, log):
    """Validate the request and set up the hook path; never raises. Inactive (a logged no-op) unless flux has the
    embedder structure."""
    st = HeadState(getattr(cfg, "sigma_head", "off"))
    try:
        if st.mode != MODE:
            st.reason = f"unknown sigma head {st.mode!r}"
        elif cfg.family != "flux" or cfg.family not in C.FAMILY_GUIDANCE:
            st.reason = f"family {cfg.family} is not flux"
        else:
            dm = model.model.diffusion_model
            ok, why = flux_structure(dm)
            if not ok:
                st.reason = why
            else:
                st.g_eval = float(C.FAMILY_GUIDANCE[cfg.family])
                st.grid = parse_grid(getattr(cfg, "sigma_head_grid", DEFAULT_GRID), st.g_eval)
                st.noises = max(1, int(getattr(cfg, "sigma_head_noises", 2)))
                if TARGET not in lora.names:
                    # not a LoRA target: an inert wrapper (up None returns the weight unchanged) attached now, before
                    # Comfy loads the model, so the hook path can later route the correction through it
                    from .lora import LoraWrapper

                    w = LoraWrapper(lora.scale)
                    model.add_weight_wrapper(f"diffusion_model.{TARGET}.weight", w)
                    st.extra[TARGET] = w
                st.active = True
    except Exception as e:  # noqa: BLE001 - a bad request is a no-op, never a failed run
        st.active, st.reason = False, f"setup failed: {type(e).__name__}: {e}"
    log(f"sigma head {st.mode}: " + ("armed (grid " + ",".join(f"{g:g}" for g in st.grid) + f", {st.noises} fit draws"
                                      + (", new key" if st.extra else "") + ")" if st.active else f"no-op ({st.reason})"))
    return st


# --- the algebra ---------------------------------------------------------------------

def sigma_embeddings(emb, sigmas, dtype, device, timestep=None):
    """time_in's input at the band sigmas: timestep_embedding(timestep(sigma) in fp32).to(model dtype)."""
    t = torch.tensor(sigmas, dtype=torch.float32, device=device)
    if timestep is not None:
        t = timestep(t).float()
    return emb(t, EMB_DIM).to(dtype)


def guidance_embeddings(emb, values, dtype, device):
    """guidance_in's input: the guidance cond cast to the model dtype, then embedded (1000 * g in that dtype)."""
    return emb(torch.tensor(values, dtype=dtype, device=device), EMB_DIM).to(dtype)


def linear_f64(module, device):
    """(W, b) of one Linear as the model uses it, merged in fp32 and returned in float64 on the CPU: Comfy's own cast
    (dequantisation, then the weight functions - the LoRA wrappers as currently routed) at dtype fp32, so the LoRA
    merge happens in fp32. Plain torch Linears (CPU mocks) apply their weight functions the same way."""
    with torch.no_grad():
        ops = None
        if hasattr(module, "comfy_cast_weights"):
            try:
                import comfy.ops as ops
            except ImportError:
                ops = None
        if ops is not None:
            w, b, st = ops.cast_bias_weight(module, None, dtype=torch.float32, device=device, bias_dtype=torch.float32, offloadable=True)
            try:
                return w.detach().double().cpu(), (b.detach().double().cpu() if b is not None else None)
            finally:
                ops.uncast_bias_weight(module, w, b, st)
        w = module.weight.detach()
        if hasattr(w, "dequantize"):
            w = w.dequantize()
        w = w.to(device=device, dtype=torch.float32)
        for f in (getattr(module, "weight_function", None) or []):
            w = f(w)
        b = getattr(module, "bias", None)
        return w.detach().double().cpu(), (b.detach().double().cpu() if b is not None else None)


def mlp_f64(mlp, device):
    return linear_f64(mlp.in_layer, device) + linear_f64(mlp.out_layer, device)


def mlp_hidden(parts, e):
    """SiLU(in_layer(e)) in float64: rows are the inputs of out_layer."""
    wi, bi = parts[0], parts[1]
    x = e.detach().double().cpu() @ wi.T
    if bi is not None:
        x = x + bi
    return torch.nn.functional.silu(x)


def mlp_out(parts, e):
    wo, bo = parts[2], parts[3]
    y = mlp_hidden(parts, e) @ wo.T
    return y + bo if bo is not None else y


def closed_form(H, D, scale):
    """H [B, n_in] rows h_b, D [B, n_out] rows Delta_b (float64). Returns (up [n_out, r], down [r, n_in]) in fp32 with
    scale * up @ down @ h_b = Delta_b, and the diagnostics. dW = D^T (H^T)^+ via the thin SVD H^T = U S V^T, then an SVD
    of D^T V S^-1 for the smallest exact rank; down = (factor) U^T has orthonormal-derived rows, so the fp32 factors
    carry the solution to fp32 accuracy. Pass rule (floating-point error analysis, no tuned constant): the relative
    residual of the exported fp32 factors at the B points must be <= B * kappa(H) * eps_fp32 - the backward-error bound
    of a rank-B fp32 product - and that bound must be < 1 (otherwise H is too ill-conditioned to carry the delta)."""
    B = H.shape[0]
    Hc, Dc = H.T.contiguous(), D.T.contiguous()                 # [n_in, B], [n_out, B]
    U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    eps64 = torch.finfo(torch.float64).eps
    s_min_ok = bool(S[-1] > max(Hc.shape) * eps64 * S[0])
    kappa = float(S[0] / S[-1]) if s_min_ok else float("inf")
    diag = {"bands": B, "kappa": kappa, "h_norms": [float(x) for x in H.norm(dim=1)],
            "delta_norm": float(Dc.norm()), "ok": False}
    if not s_min_ok:
        diag["reason"] = "H is rank-deficient: the band inputs of time_in.out_layer are not independent"
        return None, None, diag
    M = (Dc @ Vh.T) / S                                         # D^T V S^-1  [n_out, B]
    P, Sig, Qh = torch.linalg.svd(M, full_matrices=False)
    r = int((Sig > max(M.shape) * eps64 * Sig[0]).sum()) if Sig.numel() and Sig[0] > 0 else 0
    if r == 0:
        diag["reason"] = "nothing to deliver (every Delta_b is zero)"
        return None, None, diag
    up = (P[:, :r] * Sig[:r] / scale).float()
    down = (Qh[:r] @ U.T).float()
    res = scale * up.double() @ (down.double() @ Hc) - Dc
    rel = float(res.norm() / Dc.norm())
    tol = B * kappa * torch.finfo(torch.float32).eps
    diag.update({"rank": r, "residual_rel": rel, "residual_max_abs": float(res.abs().max()), "tol": tol,
                 "up_absmax": float(up.abs().max()), "delta_w_absmax": float((scale * up.double() @ down.double()).abs().max())})
    if not (tol < 1.0 and rel <= tol and math.isfinite(rel)):
        diag["reason"] = f"residual {rel:.3g} vs bound {tol:.3g} (kappa {kappa:.3g})"
        return None, None, diag
    diag["ok"] = True
    return up, down, diag


# --- statistics on ScoreReport.per_case ---------------------------------------------

def paired(a, b, keys):
    """Mean and SE of (a - b) over the given case keys (image, mode, band, noise)."""
    d = np.array([a[k] - b[k] for k in keys], dtype=np.float64)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(d.mean()), (float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")), n


def fit_bands(base, grid, g_eval, strata=C.EVAL_STRATA):
    """Per band: the grid value with the lowest mean paired difference to g_eval, adopted iff its gain exceeds its own
    paired SE; else g_eval. `base` / `grid[g]` are per_case dicts over the same keys."""
    out = []
    for b in range(strata):
        keys = sorted(k for k in base if k[2] == b)
        by_g = {}
        best = None
        for g in sorted(grid):
            m, se, n = paired(grid[g], base, keys)
            by_g[f"{g:g}"] = [m, se]
            if best is None or m < best[1]:
                best = (g, m, se, n)
        g, m, se, n = best
        adopt = n > 1 and math.isfinite(se) and -m > se
        out.append({"band": b, "g": g if adopt else g_eval, "best_grid": g, "gain": m, "se": se, "n": n, "by_g": by_g})
    return out


def subset_report(rep, items, k0, k1, strata=C.EVAL_STRATA):
    """The part of `rep` on noise draws [k0, k1) as a ScoreReport (HoldoutScorer's reduction), or None if `rep` does not
    hold every such case. Same path, state and batches give the same numbers, so a confirm report can be reused."""
    if rep is None or k1 <= k0:
        return None
    per_case, per_image, text, no_text = {}, [], [], []
    for it in items:
        modes = {}
        for mode in ("text", "no_text"):
            vals = []
            for b in range(strata):
                for k in range(k0, k1):
                    key = (it["sha256"], mode, b, k)
                    v = rep.per_case.get(key)
                    if v is None:
                        return None
                    per_case[key] = v
                    vals.append(v)
            modes[mode] = float(np.mean(vals))
        text.append(modes["text"])
        no_text.append(modes["no_text"])
        per_image.append(C.EVAL_TEXT_WEIGHT * modes["text"] + (1.0 - C.EVAL_TEXT_WEIGHT) * modes["no_text"])
    return ScoreReport(float(np.mean(per_image)), per_image, text, no_text, per_case)
