"""Resolve base weights, text encoders and VAE for a family from the validator
cache, in the form the evaluator loads them.

Parity status per family (research/docs/research_evaluator_and_field.md §2):
  krea2, z-image  base = same file as the evaluator (bf16)                    -> full parity
  qwen-image      base = same plain-fp8 file; Comfy manual_cast on both sides  -> parity (merge RTN vs SR)
  flux            base = same root checkpoint                                 -> parity
  ideogram4       cache is per-row fp8, evaluator is per-tensor comfy_quant   -> NOT parity (see L2)
Text encoders come from the cache in every family and differ from the
evaluator's fp8_scaled files; that gap is measured, not assumed (L7).
"""

import glob
import json
import os

import torch
from safetensors.torch import load_file

from . import contract as C


def _note(msg):
    """Asset-source audit line; the never-lose panel counts '[assets] fallback' lines."""
    print(f"[assets] {msg}", flush=True)


# --- file helpers --------------------------------------------------------------

def _shards(dir_path):
    for idx in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        p = os.path.join(dir_path, idx)
        if os.path.exists(p):
            wm = json.load(open(p))["weight_map"]
            return [os.path.join(dir_path, f) for f in sorted(set(wm.values()))]
    return sorted(glob.glob(os.path.join(dir_path, "*.safetensors")))


def load_sharded(files):
    sd = {}
    for f in files:
        sd.update(load_file(f, device="cpu"))
    return sd


def _root_safetensors(model_dir, min_bytes=1 << 30):
    files = [f for f in glob.glob(os.path.join(model_dir, "*.safetensors")) if os.path.getsize(f) > min_bytes]
    return sorted(files, key=os.path.getsize, reverse=True)


def resolve_model_dir(model_id: str) -> str:
    if os.path.isdir(model_id):          # the validator may pass a cached model PATH (docs/miner.md)
        return model_id
    cand = str(C.cached_model_dir(model_id))
    if os.path.isdir(cand):
        return cand
    dirs = [d for d in glob.glob(str(C.CACHE_MODELS / "*")) if os.path.isdir(d) and not os.path.basename(d).startswith(".")]
    if len(dirs) == 1:
        return dirs[0]
    raise FileNotFoundError(f"model dir for {model_id!r} not found under {C.CACHE_MODELS}")


def hf_snapshot_dir(repo: str):
    flat = str(C.CACHE_HF / repo.replace("/", "--"))
    if os.path.isdir(flat) and _shards(flat):
        return flat
    snaps = [s for s in glob.glob(str(C.CACHE_HF / ("models--" + repo.replace("/", "--")) / "snapshots" / "*")) if _shards(s)]
    if snaps:                             # a snapshot dir whose shards never landed is not a snapshot
        return sorted(snaps)[-1]
    return None


def sub_dict(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def to_bf16(sd):
    return {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.bfloat16 else v) for k, v in sd.items()}


def dequantize_scaled_fp8(sd):
    """Dequantise a scaled-fp8 state dict (per-row `weight_scale [rows]` or
    per-tensor `weight_scale []`) to bf16. Scale keys are dropped."""
    out = {}
    for k, v in sd.items():
        if k.endswith("weight_scale") or k.endswith("scale_weight") or k.endswith("comfy_quant"):
            continue
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            s = sd.get(k + "_scale")
            if s is None and k.endswith(".weight"):
                s = sd.get(k[:-len(".weight")] + ".scale_weight")
            w = v.float()
            if s is not None:
                w = w * s.float().reshape(-1, *([1] * (v.dim() - 1)))
            out[k] = w.to(torch.bfloat16)
        else:
            out[k] = v
    return out


def evaluator_grid_ideogram4(raw_sd):
    """The ideogram4 base the evaluator merges into, as bf16 training weights, from the task cache's per-row-scaled
    fp8 transformer: each fp8 Linear that carries a `.weight_scale` is dequantised exactly (fp8 x row scale, fp32),
    clamped to +-448 and rounded ONCE to e4m3 at scale 1 - the rule of crown.twin.rebuild_ideogram4_evaluator_base,
    measured against the evaluator's Comfy-Org file - then widened to bf16 (exact: every e4m3 value is a bf16
    value). Every other tensor (biases, norms, modulation, embedders) is kept as stored, as the evaluator keeps it.
    Scale keys are dropped."""
    out = {}
    for k, v in raw_sd.items():
        if k.endswith(".weight_scale"):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base is not None and v.dtype == torch.float8_e4m3fn and (base + ".weight_scale") in raw_sd:
            s = raw_sd[base + ".weight_scale"]
            w = v.float() * s.float().reshape(-1, *([1] * (v.dim() - 1)))
            out[k] = w.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float().to(torch.bfloat16)
        elif v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            out[k] = v.float().to(torch.bfloat16)      # fp8 without a scale: already on its grid
        else:
            out[k] = v
    return out


_FP8_SCALE_KEYS = (".scale_weight", ".scale_input", ".weight_scale", ".input_scale", ".comfy_quant")


def dequantize_t5_fp8(sd, dtype=torch.float16):
    """A T5 state dict in comfyanonymous' fp8 formats -> the layout of t5xxl_fp16.safetensors: every tensor fp16, no
    scale keys, no `scaled_fp8` marker. Scaled file: weight = fp8 x its per-tensor scale_weight (product in fp32, one
    rounding to fp16); plain fp8 file: an exact upcast. The raw scaled dict must never reach Comfy: engine.load_clip calls
    load_text_encoder_state_dicts, which does not run comfy.utils.convert_old_quants (only comfy.sd.load_clip does), so
    nothing would apply the scales (1.6e-4 .. 8e-3 on the real file)."""
    out = {}
    for k, v in sd.items():
        if k == "scaled_fp8" or k.endswith(_FP8_SCALE_KEYS):
            continue
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            w = v.float()
            s = sd.get(k[:-len("weight")] + "scale_weight") if k.endswith(".weight") else None
            if s is None:
                s = sd.get(k + "_scale")
            if s is not None:     # per-tensor scalar (the T5 file) or per-row vector
                s = s.float()
                w = w * (s.reshape(()) if s.numel() == 1 else s.reshape(-1, *([1] * (v.dim() - 1))))
            out[k] = w.to(dtype)
        elif v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


# --- per-family assets ---------------------------------------------------------

class Assets:
    def __init__(self, family: str, model_dir: str, baked_dir=None):
        if family not in C.FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.model_dir = model_dir
        self.baked_dir = baked_dir or os.environ.get("CROWN_ASSETS", str(C.BAKED_DIR))
        self.train_base = "cache"          # or "evaluator": ideogram4 trains on the evaluator's own base file
        self._ckpt = None

    def _full_checkpoint(self):
        if self._ckpt is None:
            roots = _root_safetensors(self.model_dir)
            if not roots:
                raise FileNotFoundError(f"no root checkpoint under {self.model_dir}")
            # the evaluator's rule (image_artifacts.prepare_base): exactly ONE root .safetensors > 5 GiB,
            # else it raises for every miner; we take the largest and say so when the repo is ambiguous
            big = [r for r in roots if os.path.getsize(r) > 5 * 1024 ** 3]
            if len(big) != 1:
                _note(f"root checkpoints > 5 GiB: {len(big)} (evaluator wants exactly one); using {os.path.basename(roots[0])}")
            self._ckpt = load_sharded(roots[:1])
            try:                                   # header metadata (e.g. _quantization_metadata) goes to Comfy like load_torch_file's
                from safetensors import safe_open

                with safe_open(roots[0], "pt") as f:
                    self._ckpt_meta = f.metadata() or None
            except Exception:  # noqa: BLE001
                self._ckpt_meta = None
        return self._ckpt

    def diffusion_meta(self):
        """Safetensors header metadata of the root checkpoint (None for directory layouts)."""
        if getattr(self, "_ckpt_meta", None) is None and self._ckpt is None:
            try:
                self._full_checkpoint()
            except Exception:  # noqa: BLE001
                return None
        return getattr(self, "_ckpt_meta", None)

    def ideogram4_raw_sd(self):
        """The task cache's ideogram4 transformer exactly as stored (per-row scaled fp8 + weight_scale), for
        rebuilding the evaluator's base in the twin (crown.twin.rebuild_ideogram4_evaluator_base)."""
        tdir = os.path.join(self.model_dir, "transformer")
        shards = _shards(tdir) if os.path.isdir(tdir) else _root_safetensors(self.model_dir)[:1]
        return load_sharded(shards)

    def diffusion_sd(self):
        f = self.family
        if f == "ideogram4":
            if getattr(self, "train_base", "cache") == "evaluator":
                # the per-tensor comfy_quant file the evaluator merges into (baked for the twin)
                path = os.path.join(self.baked_dir, C.BAKED_IDEOGRAM4_BASE)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"--train-base evaluator: {path} not baked")
                return dequantize_scaled_fp8(load_file(path))
            tdir = os.path.join(self.model_dir, "transformer")
            shards = _shards(tdir) if os.path.isdir(tdir) else _root_safetensors(self.model_dir)[:1]
            if getattr(self, "train_base", "cache") == "requant":
                # train on the evaluator's own base: the twin's rule (evaluator_grid_ideogram4, the same rule as
                # crown.twin.rebuild_ideogram4_evaluator_base, measured against the evaluator's file: 98.90 % of its
                # codes, every other tensor bitwise identical). The earlier rule rounded twice (bf16, then e4m3) and
                # also snapped the 247 bf16 tensors the evaluator keeps (biases, modulation, final layer: 2.8 % rms)
                return evaluator_grid_ideogram4(load_sharded(shards))
            return dequantize_scaled_fp8(load_sharded(shards))
        if f == "flux":
            # rayonlabs/FLUX.1-dev: flux1-dev.safetensors is a bare diffusion model (double_blocks.*,
            # single_blocks.*, ...), exactly what the evaluator's UNETLoader reads. A full checkpoint
            # (model.diffusion_model. prefix, e.g. FLUX-MonochromeManga) is stripped. fp8 weights stay
            # fp8: the evaluator keeps them and merges with stochastic rounding (v6.1 F1), so the twin
            # scores such a base, exactly as for qwen-image.
            sd = self._full_checkpoint()
            inner = sub_dict(sd, "model.diffusion_model.")
            sd = inner if inner else sd
            self.base_fp8 = any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values())
            return sd if self.base_fp8 else to_bf16(sd)
        roots = _root_safetensors(self.model_dir)
        if f == "krea2":
            # the evaluator takes the single root file > 5 GB; the repo's is raw.safetensors
            pref = [r for r in roots if "raw" in os.path.basename(r).lower()] or roots
            return to_bf16(load_sharded(pref[:1]))
        sd = self._full_checkpoint() if roots else load_sharded(roots[:1])
        self.base_fp8 = any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values())
        if self.base_fp8:
            return sd  # qwen-image: plain fp8, keep it (Comfy manual_cast == evaluator path)
        return to_bf16(sd)

    def text_encoder_sds(self):
        f = self.family
        if f in ("krea2", "ideogram4"):
            repo = "Qwen/Qwen3-VL-4B-Instruct" if f == "krea2" else "Qwen/Qwen3-VL-8B-Instruct"
            d = hf_snapshot_dir(repo)
            if d is None:
                d = os.path.join(self.model_dir, "text_encoder")
                _note(f"fallback: text encoder {repo} not in hf_cache, using {d}")
            sd = load_sharded(_shards(d))
            if any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values()):
                sd = dequantize_scaled_fp8(sd)
            return [sd]
        if f in ("qwen-image", "z-image"):
            return [load_sharded(_shards(os.path.join(self.model_dir, "text_encoder")))]
        # flux: baked clip_l + t5xxl_fp16 (the evaluator's files); fall back to a full checkpoint's
        # embedded encoders only if someone ships one
        baked = [os.path.join(self.baked_dir, n) for n in C.BAKED_FLUX_TE]
        if all(os.path.exists(b) for b in baked):
            _note("text encoders: baked clip_l + t5xxl")
            return [load_file(b, device="cpu") for b in baked]
        # the task repo's own encoders in diffusers layout (text_encoder = CLIP-L, text_encoder_2 =
        # T5-XXL) carry the same key names as the evaluator's files
        te1, te2 = os.path.join(self.model_dir, "text_encoder"), os.path.join(self.model_dir, "text_encoder_2")
        if os.path.isdir(te1) and os.path.isdir(te2) and _shards(te1) and _shards(te2):
            _note("fallback: text encoders from the task repo's text_encoder/text_encoder_2")
            return [to_bf16(load_sharded(_shards(te1))), to_bf16(load_sharded(_shards(te2)))]
        # the build baked the fp8 T5 when the fp16 one did not fit its clock (fetch_assets.sh). It ranks after the
        # repo's own T5 (rayonlabs / PixelWave snapshots: bf16 values that matched t5xxl_fp16 bit for bit once cast
        # to fp16, on 9 sampled tensors) and before an embedded one (FLUX-MonochromeManga embeds the plain, unscaled
        # fp8 T5; the scaled file keeps the fp16 norms and embeddings). Its Linear weights differ from the
        # evaluator's by ~2.65 % RMS (fp8 rounding, same sample).
        clip_l, t5_fp8 = os.path.join(self.baked_dir, C.BAKED_FLUX_TE[0]), os.path.join(self.baked_dir, C.BAKED_FLUX_T5_FP8)
        if os.path.exists(clip_l) and os.path.exists(t5_fp8):
            _note(f"fallback: fp8 T5 - baked clip_l + {C.BAKED_FLUX_T5_FP8} dequantised to fp16 "
                  f"({C.BAKED_FLUX_TE[1]} is not baked; the evaluator encodes with it)")
            return [load_file(clip_l, device="cpu"), dequantize_t5_fp8(load_file(t5_fp8, device="cpu"))]
        _note("fallback: text encoders embedded in the checkpoint")
        ck = self._full_checkpoint()
        sds = [sub_dict(ck, "text_encoders.clip_l.transformer."), sub_dict(ck, "text_encoders.t5xxl.transformer.")]
        if not all(sds):
            raise FileNotFoundError(f"flux text encoders not baked under {self.baked_dir} and not embedded in the checkpoint")
        return sds

    def vae(self):
        """('comfy', state_dict) or ('diffusers', path) — whichever the evaluator uses."""
        f = self.family
        if f == "z-image":
            return "diffusers", os.path.join(self.model_dir, "vae")
        if f == "flux":
            # the evaluator ALWAYS encodes with rayonlabs/FLUX.1-dev/ae.safetensors (image_flow_adapter.py),
            # whatever the task repo ships: the baked copy of that file comes first, the repo's own
            # ae.safetensors (rayonlabs itself) second; an embedded VAE is a last resort only
            baked = os.path.join(self.baked_dir, C.FLUX_VAE_FILE)
            if os.path.exists(baked):
                _note("vae: baked rayonlabs ae.safetensors (the evaluator's)")
                return "comfy", load_file(baked, device="cpu")
            ae = os.path.join(self.model_dir, C.FLUX_VAE_FILE)
            if os.path.exists(ae):
                _note("vae: repo ae.safetensors")
                return "comfy", load_file(ae, device="cpu")
            embedded = sub_dict(self._full_checkpoint(), "vae.")
            if embedded:
                _note("fallback: vae embedded in the checkpoint (evaluator uses rayonlabs ae)")
                return "comfy", embedded
            dvae = os.path.join(self.model_dir, "vae")               # last resort: the repo's diffusers AutoencoderKL
            if os.path.isdir(dvae) and os.path.exists(os.path.join(dvae, "config.json")):
                _note("fallback: vae from the repo's diffusers vae/ (evaluator parity not guaranteed)")
                return "diffusers", dvae
            raise FileNotFoundError(f"flux VAE {C.FLUX_VAE_FILE} not found under {self.model_dir} or {self.baked_dir}")
        baked = os.path.join(self.baked_dir, C.BAKED_VAE[f])
        if os.path.exists(baked):
            _note("vae: baked")
            return "comfy", load_file(baked, device="cpu")
        # fallback keeps the run alive; parity with the evaluator's VAE is then not guaranteed
        _note("fallback: vae from the repo's diffusers vae/ (evaluator parity not guaranteed)")
        return "diffusers", os.path.join(self.model_dir, "vae")

    def release(self):
        self._ckpt = None
