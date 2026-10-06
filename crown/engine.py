"""Training engine.

The loop runs *inside* Comfy's sampling call, so that `model_wrap(noisy, t,
**extra_args)` is the same callable the evaluator uses to score a candidate.
Loss and holdout share one formulation (scoring.py). Selection is raced and
paired (selector.py). Budget is measured, never assumed (budget.py).

Phase 2: when the holdout plateaus with budget left, a second member is
trained from a fresh seed and the two are averaged (soup). Every member keeps
the same holdout, so the soup is scored, not shipped blind.
"""

import gc
import json
import math
import os
import random
import time

import numpy as np
import torch

from . import contract as C
from . import sigma_head as SH
from .budget import Budget
from .lora import Lora, enable_checkpointing, save_lora, select_targets, soup_state
from .scoring import HoldoutScorer, flow_prediction_mse, make_noisy, sigma_of
from .selector import Candidate, Selector
from .twin import verify_loadable

LOG = print


def logs_enabled():
    return os.environ.get("GOD_TRAIN_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def log(*a):
    if logs_enabled():
        LOG(time.strftime("%H:%M:%S"), *a, flush=True)


# --- model / clip / vae, loaded the way the evaluator loads them ------------------

def load_diffusion_model(sd, family, metadata=None):
    import comfy.model_sampling
    import comfy.sd

    fp8 = any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values())
    opts = {} if fp8 else {"dtype": torch.bfloat16}
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=opts, metadata=metadata, disable_dynamic=True)
    if model is None:
        raise RuntimeError("Comfy could not detect the diffusion model")
    for p in model.model.parameters():
        p.requires_grad_(False)
    shift = C.FAMILY_SHIFT.get(family)
    if shift is not None:  # evaluator: ImageFlowAdapter.configure -> ModelSamplingAuraFlow
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        model = ModelSamplingAuraFlow().patch_aura(model, shift)[0]
    sampling = model.model.model_sampling
    if not isinstance(sampling, comfy.model_sampling.CONST) or getattr(sampling, "noise_scale", 1.0) != 1.0:
        raise ValueError("Model is not unit-noise rectified flow")
    return model


def load_clip(sds, family):
    import comfy.sd

    return comfy.sd.load_text_encoder_state_dicts(
        sds, embedding_directory=None, clip_type=getattr(comfy.sd.CLIPType, C.CLIP_TYPES[family]),
        model_options={}, disable_dynamic=True)


def encode_prompt(clip, text, family):
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))  # == nodes.CLIPTextEncode
    g = C.FAMILY_GUIDANCE.get(family)
    if g is not None:
        import node_helpers

        cond = node_helpers.conditioning_set_values(cond, {"guidance": g})
    return cond


class LatentEncoder:
    """Evaluator VAE encoding: Comfy VAE in fp32 on pixels in [0,1], or the
    diffusers posterior mean on pixels in [-1,1] (z-image)."""

    def __init__(self, kind, payload):
        self.kind = kind
        if kind == "comfy":
            import comfy.sd

            self.vae = comfy.sd.VAE(sd=payload, dtype=torch.float32)
            self.vae.first_stage_model.eval().requires_grad_(False)
            reg = getattr(self.vae.first_stage_model, "regularization", None)
            if reg is not None and getattr(reg, "sample", False):
                raise ValueError("Evaluation VAE must encode posterior means")
        else:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(payload, torch_dtype=torch.float32).eval().requires_grad_(False).to("cuda")

    @torch.inference_mode()
    def encode(self, image):
        if self.kind == "comfy":
            pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
            lat = self.vae.encode(pixels).float().cpu()
        else:
            arr = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 127.5 - 1.0
            if arr.shape[0] % 8 or arr.shape[1] % 8:
                raise ValueError("Image dimensions must be divisible by eight")
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda")
            lat = self.vae.encode(t).latent_dist.mode().float().cpu()
        if not torch.isfinite(lat).all():
            raise ValueError("non-finite latent")
        return lat

    def free(self):
        import comfy.model_management as mm

        self.vae = None
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()


def make_guider(model, cond):
    import comfy.samplers

    g = comfy.samplers.CFGGuider(model)
    g.set_conds(cond, [])
    g.set_cfg(C.EVAL_CFG)
    return g


def evaluator_schedule(strata=C.EVAL_STRATA):
    """The sigma schedule the evaluator passes to guider.sample (descending + 0),
    so `transformer_options["sample_sigmas"]` matches at training time too."""
    return torch.tensor([sigma_of(b, strata) for b in reversed(range(strata))] + [0.0])


class TrainSampler:
    """Comfy 'sampler' whose .sample runs the training loop with the real model_wrap."""

    def __init__(self, trainer):
        self.trainer = trainer

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        self.trainer.run(model_wrap, extra_args)
        return latent_image


# --- the trainer ------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg, model, items, raw_conds, out_dir, budget: Budget, twin=None):
        self.base_score = None  # set by run(); save() reads it on the identity-only path too
        self.cfg, self.model, self.items, self.raw_conds, self.out_dir, self.budget = cfg, model, items, raw_conds, out_dir, budget
        self.twin = twin   # EvaluatorTwin or None: evaluator-exact scoring for the confirm stage
        self.device = model.load_device
        self.train_items = [i for i in items if not i["holdout"]]
        self.holdout_items = [i for i in items if i["holdout"]]
        if not self.train_items or not self.holdout_items:
            raise ValueError("need at least one training and one holdout image")
        # v11 --train-on-holdout (default off): the holdout images are trained on as well and the FINAL state ships.
        # Evidence (09-28 R1 z-image a3ed678c): the one entrant that trained on all given images to the end placed 3rd,
        # -0.27 % against five same-code runs that held images out (t -4.0; research/plus5/r1_natural_experiment.md),
        # and on our own z-image trajectories the last saved state was the best in every run. The screens still run on
        # the (now seen) holdout images: their scores are biased and serve only as failure guards and diagnostics.
        self.train_on_holdout = bool(getattr(cfg, "train_on_holdout", False))
        if self.train_on_holdout:
            self.train_items = list(items)
        self.rng = random.Random(cfg.seed)
        self._processed = {}
        self.summary = {"family": cfg.family, "steps": 0, "evals": [], "confirms": [], "members": [], "n_train": len(self.train_items),
                        "n_holdout": len(self.holdout_items)}
        targets = select_targets(model.model.diffusion_model, cfg.include, cfg.exclude)
        if not targets:
            raise ValueError("LoRA include/exclude selected no Linear layers")
        self.lora = Lora(targets, cfg.rank, cfg.alpha, self.device, cfg.seed)
        self.lora.attach(model)
        if getattr(cfg, "plain_delta", ""):
            # v12 P (default off): full-rank deltas on the non-Linear parameters a LoRA cannot reach, shipped as .diff
            n_plain = self.lora.attach_plain(model.model.diffusion_model, cfg.plain_delta)
            if not n_plain:
                raise ValueError(f"--plain-delta {cfg.plain_delta!r} matched no non-Linear parameter")
            self.summary["plain_delta"] = {"regex": cfg.plain_delta, "tensors": n_plain, "numel": self.lora.plain.numel}
            log(f"plain deltas on {n_plain} tensors ({self.lora.plain.numel / 1e6:.2f}M numbers): {cfg.plain_delta}")
        self._parity_done = False
        if getattr(cfg, "exact_merge", False):
            n = self.lora.enable_exact_merge()
            self.summary["exact_merge"] = {"modules": n, "targets": len(targets)}
            log(f"exact fp8 merge on {n}/{len(targets)} Linears")
        if getattr(cfg, "fast_lora", False):
            n = self.lora.enable_fast_path(lowrank_bwd=bool(getattr(cfg, "fast_lora_lowrank", False)))
            self.summary["fast_lora"] = {"routed": n, "targets": len(targets)}
            log(f"fast LoRA path on {n}/{len(targets)} Linears")
        self._ckpt_on = bool(cfg.ckpt)
        if cfg.ckpt:
            log("gradient checkpointing on", enable_checkpointing(model.model.diffusion_model, stride=cfg.ckpt_stride), "blocks")
        log(f"lora targets={len(targets)} params={sum(p.numel() for p in self.lora.params) / 1e6:.1f}M rank={cfg.rank}")
        # v11 --sigma-head (default off): None keeps every default code path exactly as it was
        self._guidance = None          # hook-path guidance override, set only inside the sigma head's scoring
        self._sh = None
        self._sh_candidate = None
        if (getattr(cfg, "sigma_head", "off") or "off") != "off":
            self._sh = SH.prepare(cfg, model, self.lora, log)

    # --- conditioning, processed exactly as the evaluator's sample() does ---------
    # The evaluator calls guider.sample(zeros, latent, ...) per image, which runs
    # process_conds against that image's (scaled) latent. Cache per (prompt, shape).
    def _cond(self, guider, prompt, item):
        lat = item["scaled"]
        g = self._guidance             # None except inside the v11 sigma head's guidance-grid scoring
        key = (prompt, tuple(lat.shape)) if g is None else (prompt, tuple(lat.shape), "guidance", g)
        c = self._processed.get(key)
        if c is None:
            import comfy.sampler_helpers
            import comfy.samplers

            lat = lat.to(self.device)
            raw = self.raw_conds[prompt] if g is None else SH.with_guidance(self.raw_conds[prompt], g)
            conds = {"positive": comfy.sampler_helpers.convert_cond(raw),
                     "negative": comfy.sampler_helpers.convert_cond([])}
            c = comfy.samplers.process_conds(guider.inner_model, torch.zeros_like(lat), conds, self.device, lat, None, C.EVAL_SAMPLER_SEED)
            self._processed[key] = c
        return c

    def _set_cond(self, guider, prompt, item):
        guider.conds = self._cond(guider, prompt, item)

    def _forward(self, guider, extra, prompt, item, noisy, sigma):
        self._set_cond(guider, prompt, item)
        t = torch.full((noisy.shape[0],), sigma, device=self.device, dtype=torch.float32)
        return guider(noisy, t, **extra)

    # --- artifact ------------------------------------------------------------------
    def save(self, state, scale, score, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = os.path.join(self.out_dir, C.OUTPUT_LORA_NAME + ".tmp")
        save_lora(tmp, state, scale, metadata={"crown": tag})
        os.replace(tmp, os.path.join(self.out_dir, C.OUTPUT_LORA_NAME))
        self.summary["saved"] = {"tag": tag, "score": score, "base": self.base_score,
                                 "rel_pct": (score / self.base_score - 1) * 100 if self.base_score else None}
        self._dump()
        log(f"saved {tag} score={score:.6f}")

    def _dump(self):
        self.summary["holdout"] = [it["name"] for it in self.holdout_items]  # lets tools/parity_check.py rebuild the split
        tmp = os.path.join(self.out_dir, "summary.json.tmp")
        json.dump(self.summary, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(self.out_dir, "summary.json"))

    # --- scoring -------------------------------------------------------------------
    def _screen(self, guider, extra, cand: Candidate, noises=1):
        self.lora.set_tensors(cand.state, cand.scale)
        t0 = time.time()
        try:
            rep = self.scorer.score(guider, extra, lambda p, it: self._set_cond(guider, p, it), noises=noises)
        finally:
            self.lora.restore()      # an OOM or a non-finite case must never leave the wrappers on a candidate (v6.1 M2)
        dt = time.time() - t0
        if noises == 1:
            self.band_loss = self._per_band(rep)   # latest screened candidate's per-band loss (for --band-power)
        if noises == 1:
            self.budget.observe_eval(dt)
            cand.screen = rep
            self.summary["evals"].append({"step": cand.step, "tag": cand.tag, "member": cand.member, "score": rep.score,
                                          "text": float(np.mean(rep.text)), "no_text": float(np.mean(rep.no_text)), "t": time.time()})
        else:
            cand.confirm = rep
            self.summary["confirms"].append({"step": cand.step, "tag": cand.tag, "member": cand.member, "noises": noises, "score": rep.score})
        rel = (rep.score / self.base_score - 1) * 100 if self.base_score else 0.0
        log(f"[{'confirm' if noises > 1 else 'screen'} m{cand.member} step {cand.step}] {cand.tag}: {rep.score:.6f} ({rel:+.2f}% vs base) {dt:.0f}s")
        return rep

    def _reload_training_model(self):
        """After the twin's patched clone was loaded, make sure the training model (with
        its LoRA hooks) is resident on the GPU again — Comfy may have evicted it."""
        import comfy.model_management as mm

        try:
            mm.load_models_gpu([self.model])
        except Exception as e:          # once more after freeing the cache; a second failure is fatal, not silent (v6.1 m7)
            log(f"reload training model: {type(e).__name__}: {e}; retrying after empty_cache")
            torch.cuda.empty_cache()
            mm.load_models_gpu([self.model])

    def _twin_score(self, state, scale, noises, offset=0):
        try:
            dev_state = {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in state.items()}
            if offset:   # v11 sigma head: fresh noise draws [offset, offset + noises)
                return self.twin.score(dev_state, scale, noises=noises, noise_offset=offset)
            return self.twin.score(dev_state, scale, noises=noises)
        finally:
            self._reload_training_model()

    def _drop_twin(self, where, exc):
        """The twin is an instrument, never a reason to fail the run: on any failure it is recorded,
        released and dropped, and the rest of the run scores on the hook path (the training model).
        Unguarded, a failure at the parity meter ended the run before training with only the identity
        on disk, and the entrypoint published it as a success (retrospective 2026-09-27 §6)."""
        log(f"evaluator twin failed during {where} ({type(exc).__name__}: {exc}); continuing without it")
        self.summary.setdefault("twin_dropped", []).append({"where": where, "error": f"{type(exc).__name__}: {exc}"[:300]})
        twin, self.twin = self.twin, None
        try:
            twin.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._reload_training_model()
        except Exception as e:  # noqa: BLE001 - the next scoring call loads the training model again
            log(f"reload after dropping the twin: {type(e).__name__}: {e}")

    def _exact_merge_parity(self, cand):
        """Runtime gate for --exact-merge: for three routed modules, the merged weight our wrapper feeds the
        training forward must equal, element for element, the one the evaluator's own ModelPatcher builds
        (the twin: merge + stochastic re-rounding seeded by the state-dict key). Logged and recorded."""
        routed = [n for n in self.lora.names if self.lora.wrappers[n].fp8_dtype is not None]
        names = [routed[i] for i in sorted({0, len(routed) // 2, len(routed) - 1})]
        keys = {f"diffusion_model.{n}.weight": n for n in names}
        dev_state = {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in cand.state.items()}
        try:
            theirs = self.twin.patched_weights(dev_state, cand.scale, list(keys))
        finally:
            self._reload_training_model()
        if not theirs:
            log("exact-merge parity: the twin could not load this state; skipped")
            return
        self.lora.set_tensors(cand.state, cand.scale)
        try:
            with torch.no_grad():
                ours = {k: self.lora.wrappers[n](self.lora.modules[n].weight.to(device=self.device, dtype=torch.bfloat16)).cpu()
                        for k, n in keys.items()}
        finally:
            self.lora.restore()
        res = {k: [int((ours[k] == theirs[k]).sum()), ours[k].numel()] for k in keys}
        self.summary["exact_merge_parity"] = res
        log("exact-merge parity vs the evaluator's patcher: " + ", ".join(f"{k.split('diffusion_model.')[-1]} {a}/{b}" for k, (a, b) in res.items()))

    def _confirm(self, guider, extra, cand: Candidate):
        """Confirm stage: the evaluator twin when we have one (exact numerics, real
        LoraLoader path), otherwise the hook path at more noise draws."""
        noises = self.cfg.confirm_noises
        if self.twin is None:
            return self._screen(guider, extra, cand, noises=noises)
        if not self._parity_done and (self.summary.get("exact_merge") or {}).get("modules"):
            self._parity_done = True
            try:
                self._exact_merge_parity(cand)
            except Exception as e:  # noqa: BLE001 - an instrument, never a reason to fail the run
                log(f"exact-merge parity check could not run: {type(e).__name__}: {e}")
                self._reload_training_model()
        t0 = time.time()
        rep, missing = self._twin_score(cand.state, cand.scale, noises)
        dt = time.time() - t0
        if rep is None:
            log(f"[twin m{cand.member} step {cand.step}] {cand.tag}: NOT LOADABLE by the evaluator: {missing[:3]}")
            self.summary.setdefault("twin_unloadable", []).append({"tag": cand.tag, "step": cand.step, "missing": missing[:5]})
            cand.unloadable = True       # vetoed: the selector never ranks or ships it (v6.1 M6)
            return None
        cand.confirm = rep
        tb = (self.summary.get("parity") or {}).get("twin_base")   # twin numbers compare against the twin base
        rel = (rep.score / tb - 1) * 100 if tb else ((rep.score / self.base_score - 1) * 100 if self.base_score else 0.0)
        hook = cand.screen.score if cand.screen is not None else None
        self.summary["confirms"].append({"step": cand.step, "tag": cand.tag, "member": cand.member, "noises": noises, "score": rep.score,
                                         "via": "twin", "hook_screen": hook})
        log(f"[twin m{cand.member} step {cand.step}] {cand.tag}: {rep.score:.6f} ({rel:+.2f}% vs twin base; hook screen {hook if hook is None else round(hook, 6)}) {dt:.0f}s")
        return rep

    def _confirm_stage(self, guider, extra, top):
        """Confirm the top candidates within the budget. Twin and hook numbers are not comparable (the
        ideogram4 hook/twin gap is +0.71 %), so when the twin fails midway every confirm it produced is
        discarded and the stage restarts on the hook path; the loadability vetoes it already issued stand.
        A failure on the hook path propagates as before."""
        while True:
            via_twin = self.twin is not None
            try:
                for i, c in enumerate(top):
                    need = self.budget.est_eval(self.cfg.confirm_noises) * (1.5 if self.twin else 1.0)
                    if i >= self.cfg.confirm_top:
                        need += self._sigma_head_reserve()   # v11: confirms past --confirm-top never eat the sigma head's time
                    if not self.budget.fits(need):
                        break
                    self._confirm(guider, extra, c)
                return
            except Exception as e:  # noqa: BLE001
                if not via_twin:
                    raise
                self._drop_twin("confirm", e)
                for c in top:
                    c.confirm = None

    def _screens_per_point(self):
        if self.cfg.ema > 0 and getattr(self.cfg, "screen_ema_only", False):
            return 1                                  # ema only at intermediate points
        return 2 if self.cfg.ema > 0 else 1          # raw (+ ema); avg2 only at member end

    def _final_loadable(self, state, scale):
        """Loadability of a state on the evaluator's path: the twin's base (fp8 families) when there is
        one, else the training model. Any twin failure falls back to the training model (v6.1 M6)."""
        dev_state = {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in state.items()}
        if self.twin is not None:
            try:
                self.twin._load()
                ok, missing = verify_loadable(self.twin.base, dev_state, scale)
                self.twin.release()
                self._reload_training_model()
                return ok, missing
            except Exception as e:  # noqa: BLE001
                log(f"final loadability via twin failed ({type(e).__name__}: {e}); using the training model")
                try:
                    self.twin.release()
                    self._reload_training_model()
                except Exception:  # noqa: BLE001
                    pass
        return verify_loadable(self.model, dev_state, scale)

    def _pass_steps(self):
        reps = self._band_repeats() if hasattr(self, "_band_repeats") else None
        per_img = sum(reps) if reps else C.EVAL_STRATA
        return max(1, len(self.train_items) * per_img)

    def _next_screen_step(self, last_eval):
        """--screen-passes: the next scheduled step after `last_eval` (Law 1: the optimum is a number of
        passes). Spacing never drops below what --screen-max-share allows; past the list, the last spacing."""
        sched = [float(x) for x in str(getattr(self.cfg, "screen_passes", "") or "").split(",") if x.strip()]
        ps = self._pass_steps()
        share = float(getattr(self.cfg, "screen_max_share", 0.30))
        point_cost = self.budget.est_eval() * self._screens_per_point()
        min_gap = max(20, int(math.ceil(point_cost * (1 - share) / (share * max(1e-3, self.budget.est_step())))))
        pts = [int(round(p * ps)) for p in sched]
        for s in pts:
            if s >= last_eval + min_gap:
                return s
        gap = (pts[-1] - pts[-2]) if len(pts) >= 2 else ps
        return last_eval + max(min_gap, gap)

    def _eval_every(self, member=None):
        """Steps between eval points so that screening takes ~cfg.eval_share of wall-clock. With
        --adaptive-cadence the interval widens while the curve is still descending (x1.5, x2, x2.5,
        cap x3 after 1, 2, 3, 4+ consecutive real improvements) and snaps back when it stops."""
        fixed = int(getattr(self.cfg, "eval_every", 0) or 0)
        if fixed > 0:
            base = fixed
        else:
            share = self.cfg.eval_share
            cost = self.budget.est_eval() * self._screens_per_point()
            base = max(50, int(math.ceil(cost * (1 - share) / (share * self.budget.est_step()))))
        if not getattr(self.cfg, "adaptive_cadence", False):
            return base
        r = self.selector.trailing_improvements(member)
        factor = min(3.0, 1.0 + 0.5 * r)
        # never widen past what the horizon can still hold: at least two more screens must fit (v6.1 m9)
        factor = min(factor, max(1.0, self.budget.steps_affordable() / (2.0 * base)))
        if factor != getattr(self, "_cadence_factor", 1.0):
            self._cadence_factor = factor
            log(f"cadence: {r} consecutive improvements -> screens every {int(base * factor)} steps (x{factor:.1f})")
        return int(base * factor)

    # --- v11 sigma head (--sigma-head flux-guidance; default off) --------------------
    def _sigma_head_reserve(self):
        """Seconds the armed sigma head needs after the confirm stage (upper bound: no confirm report reused), so the
        member horizon and the confirm count leave it; exactly 0.0 when the head is off or a no-op."""
        sh = self._sh
        if sh is None or not sh.active:
            return 0.0
        n_hook, n_twin = sh.score_counts(self.twin is not None)
        return self.budget.est_eval(sh.noises) * (n_hook + 1.5 * n_twin) + 30.0

    def _model_dtype(self):
        """The dtype the model computes in (model_base.get_dtype_inference), fp32 for CPU fakes."""
        m = self.model.model
        for name in ("get_dtype_inference", "get_dtype"):
            f = getattr(m, name, None)
            if callable(f):
                try:
                    return f()
                except Exception:  # noqa: BLE001
                    pass
        return torch.float32

    def _sh_route(self, state, scale):
        """Route the hook path through `state`, including correction keys that are not LoRA targets."""
        self.lora.set_tensors({n: v for n, v in state.items() if n in self.lora.wrappers}, scale)
        for n, w in self._sh.extra.items():
            if n in state:
                w.up, w.down, w.scale = state[n][0].to(self.device), state[n][1].to(self.device), scale
            else:
                w.up = w.down = None

    def _sh_unroute(self):
        self.lora.restore()
        for w in self._sh.extra.values():
            w.up = w.down = None

    def _sh_score(self, guider, extra, state, scale, noises, offset=0, guidance=None):
        """Hook-path holdout score of `state` on noise draws [offset, offset + noises), conditioned at `guidance`
        (None = the evaluator's own conditioning, exactly as every screen and confirm)."""
        self._sh_route(state, scale)
        self._guidance = guidance
        try:
            return self.scorer.score(guider, extra, lambda p, it: self._set_cond(guider, p, it), noises=noises, noise_offset=offset)
        finally:
            self._guidance = None
            self._sh_unroute()

    def _sh_fresh(self, guider, extra, cand, patched_state, unpatched_fresh):
        """(patched, unpatched) reports on the fresh draws [k, 2k), on the evaluator's path: the twin when there is one,
        else the hook. `unpatched_fresh` (reused from the pick's confirm) is scored only if missing."""
        k = self._sh.noises
        if self.twin is not None:
            p, missing = self._twin_score(patched_state, cand.scale, k, offset=k)
            if p is None:
                raise RuntimeError(f"the twin cannot load the patched state: {missing[:3]}")
            if unpatched_fresh is None:
                unpatched_fresh, missing = self._twin_score(cand.state, cand.scale, k, offset=k)
                if unpatched_fresh is None:
                    raise RuntimeError(f"the twin cannot load the unpatched state: {missing[:3]}")
            return p, unpatched_fresh
        p = self._sh_score(guider, extra, patched_state, cand.scale, k, offset=k)
        if unpatched_fresh is None:
            unpatched_fresh = self._sh_score(guider, extra, cand.state, cand.scale, k, offset=k)
        return p, unpatched_fresh

    def _sigma_head(self, guider, extra, picked):
        """The patched Candidate to ship, or None (the unpatched pick ships). Never raises: any failure is a logged
        no-op recorded in summary["sigma_head"]."""
        sh = self._sh
        rec = self.summary["sigma_head"] = {"mode": sh.mode, "shipped": False}
        if not sh.active:
            rec["skipped"] = sh.reason
            log(f"sigma head: no-op ({sh.reason})")
            self._dump()
            return None
        t0 = time.time()
        try:
            out = self._sigma_head_run(guider, extra, picked, rec)
        except Exception as e:  # noqa: BLE001 - fail closed to the unpatched pick
            rec["skipped"] = f"error: {type(e).__name__}: {e}"[:300]
            log(f"sigma head failed ({type(e).__name__}: {e}); shipping the unpatched pick")
            out = None
            try:
                self._reload_training_model()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._guidance = None
            self._sh_unroute()
            rec["seconds"] = round(time.time() - t0, 1)
        if out is None and "skipped" in rec:
            log(f"sigma head: not shipped ({rec['skipped']})")
        self._dump()
        return out

    def _sigma_head_run(self, guider, extra, picked, rec):
        sh, k, g0 = self._sh, self._sh.noises, self._sh.g_eval
        items = self.holdout_items
        rec.update({"candidate": f"m{picked.member}:{picked.tag}@{picked.step}", "grid": sh.grid, "g_eval": g0, "noises": k,
                    "fit_draws": [0, k], "verify_draws": [k, 2 * k], "verify_via": "twin" if self.twin is not None else "hook"})
        # a confirm report on the same path, state and batches holds the same numbers: reuse its draws
        conf = picked.confirm
        base_fit = SH.subset_report(conf, items, 0, k) if (conf is not None and self.twin is None) else None
        unp_fresh = SH.subset_report(conf, items, k, 2 * k) if conf is not None else None
        n_hook = len(sh.grid) + 1 + (base_fit is None) + (0 if self.twin is not None else 1 + (unp_fresh is None))
        n_twin = 0 if self.twin is None else 1 + (unp_fresh is None)
        need = self.budget.est_eval(k) * (n_hook + 1.5 * n_twin) + 30.0
        rec["budget"] = {"need_s": round(need, 1), "remaining_s": round(self.budget.remaining(), 1), "hook_scores": n_hook,
                         "twin_scores": n_twin, "reused": {"base_fit": base_fit is not None, "unpatched_fresh": unp_fresh is not None}}
        if not self.budget.fits(need):
            rec["skipped"] = f"budget: needs {need:.0f}s, {self.budget.remaining():.0f}s left"
            return None
        log(f"sigma head: fitting per-band guidance on {len(items)} holdout images, grid {sh.grid} at {k} draws ({need:.0f}s budgeted)")
        # 1. fit on draws [0, k): the candidate conditioned at each grid guidance, paired per band against g_eval
        if base_fit is None:
            base_fit = self._sh_score(guider, extra, picked.state, picked.scale, k)
        grid = {g: self._sh_score(guider, extra, picked.state, picked.scale, k, guidance=g).per_case for g in sh.grid}
        bands = SH.fit_bands(base_fit.per_case, grid, g0)
        chosen = [b["g"] for b in bands]
        rec["bands"] = bands
        rec["g_per_band"] = chosen
        keys = sorted(base_fit.per_case)
        composite = {key: (grid[chosen[key[2]]][key] if chosen[key[2]] != g0 else base_fit.per_case[key]) for key in keys}
        gm, gse, gn = SH.paired(composite, base_fit.per_case, keys)
        rec["fit_gain"] = {"mean": gm, "se": gse, "n": gn, "pct": gm / base_fit.score * 100}
        log(f"sigma head: g per band {[f'{g:g}' for g in chosen]}; in-sample gain {gm / base_fit.score * 100:+.3f}% (SE {gse / base_fit.score * 100:.3f}%)")
        if all(g == g0 for g in chosen):
            rec["skipped"] = "no band's gain exceeds its paired SE"
            return None
        # 2. the exact delivery: dW = [Delta_b] H^+ on time_in.out_layer, rank-concatenated onto the candidate
        state, deliv = self._sh_build(picked, chosen, g0)
        rec["delivery"] = deliv
        if state is None:
            rec["skipped"] = f"delivery: {deliv.get('reason')}"
            return None
        # 3a. reproduction: the patched state at g_eval must reproduce the per-band-g scores on the fit draws. Rule: its
        # mean deviation from them stays within one SE of the gain it delivers (below the resolution of every decision)
        p_fit = self._sh_score(guider, extra, state, picked.scale, k)
        rm, rse, rn = SH.paired(p_fit.per_case, composite, keys)
        rep_ok = math.isfinite(rm) and math.isfinite(gse) and abs(rm) <= gse
        rec["reproduction"] = {"mean": rm, "se": rse, "n": rn, "bound": gse, "ok": rep_ok,
                               "realised_gain_pct": (p_fit.score / base_fit.score - 1) * 100}
        self._sh_candidate = Candidate(picked.tag + "+sh", picked.step, state, picked.scale, screen=p_fit, member=picked.member)
        log(f"sigma head: reproduction {rm:+.2e} vs bound {gse:.2e} -> {'ok' if rep_ok else 'FAILED'}")
        if not rep_ok:
            rec["skipped"] = "reproduction: the patched state does not reproduce the per-band-g scores"
            return None
        # 3b. adoption on fresh draws [k, 2k), never used by the fit: patched vs unpatched, beyond one paired SE
        p_new, u_new = self._sh_fresh(guider, extra, picked, state, unp_fresh)
        fkeys = sorted(k2 for k2 in u_new.per_case if k2 in p_new.per_case)
        fm, fse, fn = SH.paired(p_new.per_case, u_new.per_case, fkeys)
        adopt = fn > 1 and math.isfinite(fse) and fm < -fse
        rec["fresh"] = {"mean": fm, "se": fse, "n": fn, "pct": fm / u_new.score * 100, "patched": p_new.score,
                        "unpatched": u_new.score, "adopt": adopt}
        log(f"sigma head: fresh draws {fm / u_new.score * 100:+.3f}% (SE {fse / u_new.score * 100:.3f}%, n {fn}) -> {'ADOPT' if adopt else 'reject'}")
        if not adopt:
            rec["skipped"] = "fresh draws: the gain is not beyond one paired SE"
            return None
        rec["adopted"] = True
        return self._sh_candidate

    def _sh_build(self, cand, chosen, g0):
        """(patched state, diagnostics) or (None, diagnostics). H and Delta from the candidate-merged fp32 weights, the
        closed-form rank <= 16 factors, the composition with the candidate's own factors, then the realised check in
        the model's dtype through the hook path's merge (the evaluator's numerics)."""
        dm = self.model.model.diffusion_model
        dt = self._model_dtype()
        emb, src = SH.embedding_fn()
        ms = getattr(self.model.model, "model_sampling", None)
        sigmas = [sigma_of(b) for b in range(C.EVAL_STRATA)]
        e_sig = SH.sigma_embeddings(emb, sigmas, dt, self.device, getattr(ms, "timestep", None))
        gvals = sorted(set(chosen) | {g0})
        row = {g: i for i, g in enumerate(gvals)}
        e_g = SH.guidance_embeddings(emb, gvals, dt, self.device)
        self._sh_route(cand.state, cand.scale)
        try:
            ti = SH.mlp_f64(dm.time_in, self.device)
            gi = SH.mlp_f64(dm.guidance_in, self.device)
        finally:
            self._sh_unroute()
        H = SH.mlp_hidden(ti, e_sig)                                  # [16, hidden]: inputs of time_in.out_layer
        G = SH.mlp_out(gi, e_g)                                       # [len(gvals), hidden]: guidance_in outputs
        D = torch.stack([G[row[g]] - G[row[g0]] for g in chosen])     # zero rows where the band keeps g0
        up, down, diag = SH.closed_form(H, D, cand.scale)
        diag.update({"embedding": src, "dtype": str(dt), "target": SH.TARGET})
        log(f"sigma head: delivery rank {diag.get('rank')} kappa(H) {diag['kappa']:.3g} residual {diag.get('residual_rel', float('nan')):.2e}"
            f" (bound {diag.get('tol', float('nan')):.2e})")
        if up is None:
            return None, diag
        state = dict(cand.state)
        if SH.TARGET in state:
            u0, d0 = state[SH.TARGET]
            state[SH.TARGET] = (torch.cat([u0.cpu(), up.to(u0.dtype)], 1), torch.cat([d0.cpu(), down.to(d0.dtype)], 0))
            diag["composed"] = [int(d0.shape[0]), int(down.shape[0])]
        else:
            state[SH.TARGET] = (up, down)
            diag["new_key"] = True
        vec = self._sh_vec_check(dm, cand, state, e_sig, e_g, row, chosen, g0)
        diag["vec"] = vec
        if not vec["ok"]:
            diag["ok"] = False
            diag["reason"] = "realised vec: the patched model is not closer to the per-band-g vec than the unpatched one"
            return None, diag
        return state, diag

    def _sh_vec_check(self, dm, cand, state, e_sig, e_g, row, chosen, g0):
        """The delivered shift in the model's own dtype through the hook path's merge: at every corrected band the
        patched vec at g0 must be closer to the vec at g_b than the unpatched vec at g0 is (else the correction does
        worse than nothing); kept bands report their drift."""
        with torch.no_grad():
            self._sh_route(cand.state, cand.scale)
            try:
                t_un = dm.time_in(e_sig)
                g_all = dm.guidance_in(e_g)
            finally:
                self._sh_unroute()
            self._sh_route(state, cand.scale)
            try:
                t_p = dm.time_in(e_sig)
            finally:
                self._sh_unroute()
        idx = torch.tensor([row[g] for g in chosen], device=g_all.device)
        target = (t_un + g_all[idx]).double()                 # forward_orig: vec = time_in(...) + guidance_in(...)
        patched = (t_p + g_all[row[g0]].unsqueeze(0)).double()
        plain = (t_un + g_all[row[g0]].unsqueeze(0)).double()
        err = (patched - target).norm(dim=1)
        shift = (target - plain).norm(dim=1)
        moved = [b for b, g in enumerate(chosen) if g != g0]
        kept = [b for b, g in enumerate(chosen) if g == g0]
        ratio = [float(err[b] / shift[b]) if shift[b] > 0 else float("inf") for b in moved]
        ok = bool(moved) and all(math.isfinite(r) and r < 1.0 for r in ratio) and bool(torch.isfinite(patched).all())
        mean_shift = float(shift[moved].mean()) if moved else float("nan")
        return {"ok": ok, "err_over_shift": ratio, "max_err_over_shift": max(ratio) if ratio else None,
                "kept_drift_over_mean_shift": [float(err[b] / mean_shift) for b in kept] if moved else []}

    # --- run -----------------------------------------------------------------------
    def run(self, guider, extra):
        cfg = self.cfg
        for it in self.items:
            it["scaled"] = it["scaled"].to(self.device)
        self.scorer = HoldoutScorer(self.holdout_items, None, self.device)
        self.selector = Selector()
        import crown.selector as _sel
        _sel.SEL_METRIC = getattr(cfg, "select_metric", "mean") or "mean"
        _sel.PICK_TIEBREAK = getattr(cfg, "pick_tiebreak", "earliest") or "earliest"

        # base: identity LoRA (up == 0), parked on the CPU like every other candidate state and saved
        # BEFORE the first screen, so an artifact exists whatever the first forward does (v6.1 m1/m2)
        ident = Candidate("identity", 0, {n: (u.detach().cpu(), d.detach().cpu()) for n, (u, d) in self.lora.state().items()}, self.lora.scale)
        self.base_score = None
        self.best = ident
        self.save(ident.state, ident.scale, float("nan"), "identity")
        base = self._screen(guider, extra, ident)
        self.base_score = base.score
        self.summary["base_score"] = self.base_score
        self.summary["base_per_band"] = self._per_band(base)
        self.selector.add(ident)
        self.best = ident
        self.save(ident.state, ident.scale, ident.score, "identity")
        log(f"base holdout {self.base_score:.6f} on {len(self.holdout_items)} imgs; train imgs {len(self.train_items)}; eval {self.budget.est_eval():.0f}s")
        # parity meter: the same identity state scored by the evaluator twin at the same draws.
        # Any gap here is base-model / merge / forward numerics, not the LoRA — recorded, not hidden.
        if self.twin is not None:
            t0 = time.time()
            try:
                trep, missing = self._twin_score(ident.state, ident.scale, 1)
            except Exception as e:  # noqa: BLE001 - before training: must not end the run with only the identity on disk
                self._drop_twin("parity", e)
                trep, missing = None, None
            if trep is not None:
                gap = (trep.score / self.base_score - 1) * 100
                self.summary["parity"] = {"hook_base": self.base_score, "twin_base": trep.score, "gap_pct": gap, "twin_source": self.twin.source, "t": time.time() - t0}
                log(f"parity: hook base {self.base_score:.6f} vs twin base {trep.score:.6f} -> gap {gap:+.3f}% ({time.time() - t0:.0f}s)")
            elif missing is not None:
                self.summary["parity"] = {"error": missing[:5]}
                log(f"parity: identity NOT LOADABLE by evaluator path: {missing[:3]}")

        members = []
        member = 0
        slices = int(getattr(cfg, "member_slices", 0) or 0)
        passes = float(getattr(cfg, "member_passes", 0.0) or 0.0)
        if passes > 0 and not self.train_on_holdout:
            log("--member-passes needs --train-on-holdout (a fixed horizon has no holdout pick); ignored")
            passes = 0.0
        if passes > 0:
            # v12 --member-passes P: K follows from the budget and the horizon law (research/plus5/v12_recipe_design.md
            # C1): each member trains exactly P passes over all images, and the artefact is the soup of the K finals.
            # est_step is measured by the first member's first steps; until then the budget's prior is used, so K is
            # re-planned once after member 0 has run (see _plan_members).
            self._planned = None
            slices = self._plan_members(passes, member=0)
            cfg.max_members = max(cfg.max_members, slices)
            cfg.member_slices = slices
            self.summary["member_passes"] = {"P": passes, "K": slices, "plan": dict(self._planned or {})}
        if slices > 1 and not getattr(cfg, "seed2", False):
            # v10 --member-slices: every one of the K members must get its slice (R1 09-28: the ideogram4 places
            # 1-5 were 2-3 member soups; our plateau-driven phase 2 declared late and never started member 1)
            cfg.max_members = max(cfg.max_members, slices)
            self.summary["member_slices"] = slices
        self._cut_members = set()             # v12 C5: members the deadline truncated below their plan (soup excludes them)
        while True:
            if member > 0:
                # --same-init (v12 S2): every member restarts from member 0's init; the training draws stay per member
                self.lora.reinit(cfg.seed + (0 if getattr(cfg, "same_init", False) else 1000 * member))
            m_best, plateau_exit = self._train_member(guider, extra, member)
            members.append(m_best)
            self.summary["members"].append({"member": member, "best_tag": m_best.tag, "best_step": m_best.step,
                                            "best_member": m_best.member, "score": m_best.score, "plateau_exit": plateau_exit})
            if passes > 0 and member == 0:
                # re-plan K with the measured step time: the members after the first get equal step plans
                k2 = self._plan_members(passes, member=1, done=1)
                if k2 != slices:
                    log(f"member-passes: re-planned K {slices} -> {k2} after member 0 ({self.budget.est_step():.2f} s/step)")
                    slices = k2
                    cfg.max_members = max(cfg.max_members, slices)
                    cfg.member_slices = slices
                self.summary["member_passes"].update({"K": slices, "plan": dict(self._planned or {})})
                if slices > 1:
                    plateau_exit = True       # the planned members follow whatever member 0's exit said
            # phase 2: the member returned early because its holdout plateaued and the
            # remaining budget can plausibly carry a fresh member to a comparable optimum
            if getattr(cfg, "seed2", False):
                if plateau_exit:
                    self._seed2(guider, extra, m_best)
                break
            if not plateau_exit or member + 1 >= cfg.max_members:
                break
            member += 1
            log(f"plateau reached with {self.budget.remaining():.0f}s left; starting member {member}")

        # soup of member bests (exact rank-concat average), scored like any candidate; only when the
        # members' bests are distinct states (v6.1 M3: a member without a best of its own is skipped).
        # Under --train-on-holdout (v12 C2) the soup of bests and the confirm stage are skipped: the holdout is seen,
        # so neither could choose anything; the time goes to training and the finals are souped below.
        if (not self.train_on_holdout and len(members) > 1 and len({id(m.state) for m in members}) == len(members)
                and self.budget.fits(self.budget.est_eval())):
            sc = Candidate("soup", self.summary["steps"], soup_state([m.state for m in members]), self.lora.scale, member=-1)
            self._screen(guider, extra, sc)
            self.selector.add(sc)

        # confirm the top candidates with more noise draws (the evaluator's own draws 0..k-1) —
        # through the evaluator twin where numerics differ — then 1-SE pick
        if self.train_on_holdout:
            top = []
            self.summary["confirms_skipped"] = "train-on-holdout"
        else:
            top = self.selector.top(self._confirm_count())
            self._confirm_stage(guider, extra, top)
        picked =self.selector.pick(use_confirm=any(c.confirm is not None for c in top)) or self.best
        if self.train_on_holdout:
            # the holdout was trained on, so its scores cannot choose a step: ship the latest state (EMA preferred at that
            # step, then the last-two average, then raw); soups and the identity are never "latest"
            pref = {"ema": 2, "avg2": 1, "raw": 0}
            pool = [c for c in self.selector.cands if c.member >= 0 and c.step > 0 and not c.unloadable and c.state is not None
                    and c.member not in self._cut_members]
            if not pool:  # every member was cut short: fall back to whatever exists (never ship nothing)
                pool = [c for c in self.selector.cands if c.member >= 0 and c.step > 0 and not c.unloadable and c.state is not None]
            if pool:
                # each member's own final state (EMA preferred at its last step); with several members (--member-slices)
                # the artefact is the exact rank-concat soup of those finals: the ensemble average of independently
                # seeded runs, no selection anywhere (R1 09-28: the z-image winner shipped 2 separately trained members)
                finals = {}
                for c in pool:
                    cur = finals.get(c.member)
                    if cur is None or (c.step, pref.get(c.tag, -1)) > (cur.step, pref.get(cur.tag, -1)):
                        finals[c.member] = c
                latest = max(finals.values(), key=lambda c: (c.step, c.member))
                shipped = latest
                if len(finals) > 1 and self.budget.fits(self.budget.est_eval()):
                    parts = [finals[m] for m in sorted(finals)]
                    sf = Candidate("soup_final", self.summary["steps"], soup_state([p.state for p in parts]), self.lora.scale, member=-1)
                    self._screen(guider, extra, sf)          # biased (seen images): a failure guard, not a choice
                    self.selector.add(sf)
                    if sf.screen is not None and not sf.unloadable and math.isfinite(sf.screen.score):
                        shipped = sf
                        log("train-on-holdout: soup of the members' final states " +
                            ", ".join(f"m{p.member}:{p.tag}@{p.step}" for p in parts))
                    else:
                        log("train-on-holdout: the final soup did not screen; shipping the latest member state")
                log(f"train-on-holdout: shipping {shipped.tag}@{shipped.step} (member {shipped.member}) "
                    f"instead of the holdout pick {picked.tag}@{picked.step}")
                self.summary["train_on_holdout"] = {"selector_pick": f"m{picked.member}:{picked.tag}@{picked.step}",
                                                   "shipped": f"m{shipped.member}:{shipped.tag}@{shipped.step}",
                                                   "finals": [f"m{finals[m].member}:{finals[m].tag}@{finals[m].step}" for m in sorted(finals)],
                                                   "cut_members": sorted(self._cut_members)}
                picked = shipped
        # v11 --sigma-head: a per-band guidance correction fitted on the holdout and verified on fresh draws; it
        # replaces the pick only if it passes every gate, and the loadability check below then runs on it
        patched = self._sigma_head(guider, extra, picked) if self._sh is not None else None
        unpatched = picked
        if patched is not None:
            picked = patched
        # never ship something the evaluator cannot load: check the final state on the evaluator's own
        # loader path (the twin's base where numerics differ, else the training model)
        ok, missing = self._final_loadable(picked.state, picked.scale)
        if patched is not None:
            self.summary["sigma_head"]["loadable"] = {"ok": ok, "missing": missing[:5]}
            self.summary["sigma_head"]["shipped"] = bool(ok)
            if not ok:
                log(f"sigma head: patched state NOT LOADABLE ({missing[:3]}); shipping the unpatched pick")
                picked = unpatched
                ok, missing = self._final_loadable(picked.state, picked.scale)
        self.summary["final_loadable"] = {"ok": ok, "missing": missing[:5]}
        if not ok:
            log(f"FINAL ARTIFACT NOT LOADABLE ({missing[:3]}); falling back to best loadable candidate")
            for alt in sorted([c for c in self.selector.cands if c is not picked and not c.unloadable], key=lambda c: c.score):
                ok2, _ = self._final_loadable(alt.state, alt.scale)
                if ok2:
                    picked = alt
                    break
        self.save(picked.state, picked.scale, picked.score, f"m{picked.member}:{picked.tag}@{picked.step}")
        rep = picked.confirm or picked.screen
        self.summary["final"] = {"picked": picked.tag, "step": picked.step, "member": picked.member, "score": picked.score,
                                 "base": self.base_score, "rel_pct": (picked.score / self.base_score - 1) * 100,
                                 "per_band": self._per_band(rep), "per_image": rep.per_image, "wall_s": self.budget.elapsed()}
        self._dump()
        log("final:", json.dumps(self.summary["final"]))
        # local analysis only - the variable is unset on the validator, and the directory is not the artifact's:
        # every screened candidate's LoRA, so hidden images can score the alternatives the selector passed over
        cdir = os.environ.get("CROWN_SAVE_CANDIDATES")
        if cdir:
            os.makedirs(cdir, exist_ok=True)
            n = 0
            for c in self.selector.cands:
                if c.state is None or c.unloadable:
                    continue
                sc = c.confirm.score if c.confirm is not None else (c.screen.score if c.screen is not None else float("nan"))
                save_lora(os.path.join(cdir, f"m{c.member}_{c.tag}_{c.step}.safetensors"), c.state, c.scale,
                          metadata={"crown": f"m{c.member}:{c.tag}@{c.step}", "score": f"{sc:.6f}",
                                    "picked": str(c is picked)})
                n += 1
            sc = self._sh_candidate          # v11: the patched pick, shipped or not, for the paired hidden-image readout
            if sc is not None:
                save_lora(os.path.join(cdir, f"m{sc.member}_{sc.tag}_{sc.step}.safetensors"), sc.state, sc.scale,
                          metadata={"crown": f"m{sc.member}:{sc.tag}@{sc.step}", "score": f"{sc.score:.6f}",
                                    "picked": str(sc is picked)})
                n += 1
            log(f"saved {n} candidates to {cdir}")

    def _per_band(self, rep):
        acc = {}
        for (sha, mode, band, k), v in rep.per_case.items():
            acc.setdefault(band, []).append(v)
        return [float(np.mean(acc[b])) for b in sorted(acc)]

    def _member_guard_s(self):
        """v12 C2/C3: the seconds a member's end costs under --train-on-holdout (one EMA guard screen plus the shared
        final-soup screen and loadability check, amortised over the members)."""
        k = max(1, int(getattr(self.cfg, "member_slices", 0) or 1))
        return self.budget.est_eval() * (1.2 + 2.0 / k) + 5.0

    def _plan_members(self, passes, member=0, done=0):
        """v12 C1: K = max(1, round(avail / T_member)) with T_member = P x pass_steps x est_step + the member guard;
        avail is the whole remaining horizon under --train-on-holdout (no confirm stage). Records the plan."""
        budget = self.budget
        avail = budget.remaining() - 30.0 - self._sigma_head_reserve()
        steps_per = int(round(passes * self._pass_steps()))
        t_member = steps_per * budget.est_step() + self._member_guard_s()
        k_more = max(1 if done == 0 else 0, int(round(avail / max(1.0, t_member))))
        k = done + k_more
        self._planned = {"steps_per_member": steps_per, "t_member_s": round(t_member, 1), "avail_s": round(avail, 1),
                         "est_step": round(budget.est_step(), 4), "planned_at_member": member, "K": k}
        log(f"member-passes: P {passes} = {steps_per} steps/member ({t_member:.0f} s incl. guard) into {avail:.0f} s -> K {k}"
            + (f" ({done} done)" if done else ""))
        return max(1, k)

    def _train_member(self, guider, extra, member):
        cfg, lora, budget = self.cfg, self.lora, self.budget
        groups = [{"params": lora.up_params, "lr_mult": cfg.lora_plus_ratio}, {"params": lora.down_params, "lr_mult": 1.0}]
        if lora.plain is not None:   # --plain-delta: the recipe's learning rate, as for every other trained tensor
            groups.append({"params": [lora.plain.param], "lr_mult": 1.0})
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8)
        ema = [p.detach().clone() for p in lora.params] if cfg.ema > 0 else None
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7919 * member)
        t_start = time.time()
        t_member = t_start                    # t_start moves when polishing starts; this one does not
        toh = self.train_on_holdout
        # this member's wall-clock horizon: whatever is left, minus what the confirm stage will need. Under
        # --train-on-holdout (v12 C2) there is no confirm stage, so the whole remainder is training time.
        confirm_s = 0.0 if toh else budget.est_eval(cfg.confirm_noises) * cfg.confirm_top
        horizon_end = time.time() + max(60.0, budget.remaining() - confirm_s
                                        - self._sigma_head_reserve())   # 0.0 unless the v11 sigma head is armed
        if toh:   # v12: the 60 s floor must never put the horizon past the deadline (the end guard would be lost)
            horizon_end = min(horizon_end, time.time() + max(5.0, budget.remaining() - self._sigma_head_reserve()))
        # v10 --member-slices K: this member gets (time left)/(K - member) of the pre-confirm horizon, so all K members
        # finish; the cosine LR below spans the slice. The last member takes the rest.
        slices = int(getattr(cfg, "member_slices", 0) or 0)
        sliced = slices > 1 and member < slices - 1 and not getattr(cfg, "seed2", False)
        if sliced:   # never longer than the full horizon (mock 09-28: a 60 s floor swallowed a 75 s budget)
            horizon_end = min(horizon_end, t_start + max(5.0, (horizon_end - t_start) / (slices - member)))
        # v12 C3/C4: under --train-on-holdout the member's end guard is budgeted inside its own slice, and with
        # --member-passes the member trains exactly its planned steps (cosine over those steps, not the clock)
        plan_steps = 0
        if toh:
            horizon_end -= self._member_guard_s()
            if getattr(self, "_planned", None) and float(getattr(cfg, "member_passes", 0.0) or 0.0) > 0:
                plan_steps = int(self._planned["steps_per_member"])
        step, last_eval, cases, raw_hist = 0, 0, [], []
        m_best = None            # this member's own best (v6.1 M3), separate from the global self.best
        next_screen = None       # --screen-passes schedule (v7)
        bad = 0                  # consecutive non-finite steps (v6.1 M4)
        plateau_exit = False
        peak, warm_base, polishing, tag_prefix, pol_points = cfg.lr, 0, False, "", 0
        while True:
            now = time.time()
            if plan_steps and step >= plan_steps:
                break                         # v12 C4: the planned passes are done; the end guard follows
            if toh:
                due = False                   # v12 C2: no in-member screens on a seen holdout; the end guard only
            elif getattr(cfg, "screen_passes", ""):
                if next_screen is None:
                    next_screen = self._next_screen_step(last_eval)
                due = step >= next_screen
            else:
                due = (step - last_eval) >= self._eval_every(member)
            need = budget.est_step() + (budget.est_eval() * (self._screens_per_point() + 0.2) + 5 if due else 0)
            if now + need > horizon_end or not budget.fits(need):
                break
            # phase-2 exit: this member's screens have plateaued and another member could
            # reach a comparable optimum (time-to-best of this member, plus its evals) in the time left
            # with --member-slices the slices themselves schedule the members: no plateau hand-over inside a slice
            # (mock 09-28: early exits gave slices of 6 / 9 / 50 s and a last member far past its optimum)
            want_phase2 = cfg.phase2 and member + 1 < cfg.max_members and slices <= 1
            want_polish = getattr(cfg, "polish", False) and not polishing
            # v6.2: a curve whose best is its first or second real point has already peaked (flux: optimum
            # at 6 min of 45, plateau declared at 27 min under a 3-point window) -> two points suffice
            win = getattr(cfg, "plateau_window", 3)
            if 0 <= self.selector.best_point_index(member) <= 1:
                win = min(win, 2)
            want_seed2 = getattr(cfg, "seed2", False) and member == 0
            plateau = (want_phase2 or want_polish or polishing or want_seed2) and self.selector.plateaued(window=win, member=member)
            mb = m_best or self.best
            if polishing and pol_points >= 3 and plateau:
                # the anneal has converged too: hand the rest to the confirm stage
                log(f"m{member} polish plateau at step {step} (best {mb.tag}@{mb.step}); {budget.remaining():.0f}s left for confirms")
                break
            if plateau and member == 0 and getattr(cfg, "seed2", False) and mb.step > 0 and mb.member == 0:
                # v7: exit only if a clean second seed of s* steps, its screen, the soup screen and one soup
                # confirm fit; otherwise keep training member 0 (never leave dead time)
                need_s2 = mb.step * budget.est_step() * 1.02 + self._seed2_reserve()
                if budget.fits(need_s2) and self._well_formed(mb):
                    plateau_exit = True
                    log(f"m0 plateau at step {step} (best {mb.tag}@{mb.step}); seed2 for {mb.step} steps fits ({need_s2:.0f}s of {budget.remaining():.0f}s)")
                    break
            if plateau and want_phase2 and not getattr(cfg, "seed2", False):
                need_next = max(1, mb.step) * budget.est_step() + 3 * budget.est_eval() + budget.est_eval(cfg.confirm_noises) * cfg.confirm_top
                if budget.fits(need_next):
                    plateau_exit = True
                    log(f"m{member} plateau at step {step} (best {mb.tag}@{mb.step}); handing {budget.remaining():.0f}s to the next member")
                    break
            if plateau and want_polish:
                if (m_best is not None and m_best.step > 0 and m_best.member == member
                        and horizon_end - now > 3 * budget.est_eval() + 60):
                    # anneal from the best state with a low, decaying LR for the time left, instead of
                    # training on past the optimum (the champion's telemetry shows 50-90% of budget lost there)
                    lora.load_state({n: (u.to(self.device), d.to(self.device)) for n, (u, d) in m_best.state.items()})
                    peak = cfg.lr * cfg.polish_lr_frac
                    opt = torch.optim.AdamW(groups, lr=peak, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8)
                    if ema is not None:
                        ema = [q.detach().clone() for q in lora.params]
                    t_start, warm_base, polishing, tag_prefix = now, step, True, "pol-"
                    raw_hist.clear()
                    self.summary["polish"] = {"from": f"{m_best.tag}@{m_best.step}", "at_step": step, "peak_lr": peak, "seconds": horizon_end - now}
                    log(f"m{member} plateau at step {step}: polishing from {m_best.tag}@{m_best.step} at lr {peak:.1e} for {horizon_end - now:.0f}s")
            if plan_steps:
                p = min(1.0, step / max(1, plan_steps))       # v12 C4: the cosine spans the planned steps
            else:
                p = min(1.0, (now - t_start) / max(1.0, horizon_end - t_start))
            warm = min(1.0, (step - warm_base + 1) / max(1, cfg.warmup_steps))
            lr = peak * warm * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]

            # case cycle: every (image, band) pair once per pass, shuffled; uniform over bands like the score
            cases = self._cases_at(cases, len(self.train_items), p)
            i, band = cases.pop()
            item = self.train_items[i]
            sigma = sigma_of(band)
            noise = torch.randn(item["scaled"].shape, generator=gen, dtype=torch.float32).to(self.device)
            frac = float(getattr(cfg, "empty_prompt_frac", 0.5))
            if frac == 0.5:
                prompt = item["caption"] if (self.summary["steps"] % 2 == 0) else ""   # 50/50, like the score
            else:
                prompt = "" if self.rng.random() < frac else item["caption"]
            ts = time.time()
            try:
                with torch.enable_grad():
                    noisy, target = make_noisy(item["scaled"], noise, sigma)
                    den = self._forward(guider, extra, prompt, item, noisy, sigma)
                    loss = flow_prediction_mse(noisy, den, target, sigma).mean()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(lora.params, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                bad = 0
            except (ValueError, FloatingPointError):
                # one non-finite case (flow_prediction_mse raises) skips the step instead of ending the run (v6.1 M4)
                opt.zero_grad(set_to_none=True)
                noisy = den = loss = None
                bad += 1
                self.summary["nonfinite"] = self.summary.get("nonfinite", 0) + 1
                if bad >= 20:
                    log(f"m{member} {bad} consecutive non-finite losses at step {step}: stopping this member with best-so-far")
                    break
                continue
            except torch.OutOfMemoryError:
                # fail closed: never let one OOM end the run with the identity artifact
                opt.zero_grad(set_to_none=True)
                noisy = den = loss = None
                torch.cuda.empty_cache()
                self.summary["oom"] = self.summary.get("oom", 0) + 1
                if not self._ckpt_on:
                    enable_checkpointing(self.model.model.diffusion_model, stride=1)
                    self._ckpt_on = True
                    cases.append((i, band))
                    log(f"m{member} OOM at step {step}: gradient checkpointing enabled, retrying the step ({self._mem()})")
                    continue
                log(f"m{member} OOM again at step {step} with checkpointing on: stopping this member with best-so-far ({self._mem()})")
                break
            if ema is not None:
                with torch.no_grad():
                    for e, q in zip(ema, lora.params):
                        e.mul_(cfg.ema).add_(q.detach(), alpha=1 - cfg.ema)
            budget.observe_step(time.time() - ts)
            step += 1
            self.summary["steps"] += 1
            if step % 25 == 0:
                log(f"m{member} step {step} loss {loss.item():.4f} band {band} lr {lr:.2e} step {budget.est_step():.2f}s left {horizon_end - time.time():.0f}s {self._mem()}")
            if due:
                last_eval = step
                next_screen = None
                m_best = self._screen_candidates(guider, extra, member, step, ema, raw_hist, m_best, tag_prefix=tag_prefix)
                if polishing:
                    pol_points += 1
        # member end: a final screen (if the last eval point is stale) plus the average of the
        # last two raw states — one extra candidate that costs one screen per member, not per point.
        # Under --train-on-holdout (v12 C2) the end is one EMA guard screen: it fails closed (a non-finite or
        # unloadable final is excluded from the soup) and it is the state the soup takes.
        if toh:
            horizon_end += self._member_guard_s()     # the guard's own reserve is spent now
            guarded = False
            if step > 0 and budget.fits(budget.est_eval() * 1.2):
                n_before = len(self.selector.cands)
                m_best = self._screen_candidates(guider, extra, member, step, ema, raw_hist, m_best, tag_prefix=tag_prefix,
                                                 final=True, ema_only_final=ema is not None)
                guarded = len(self.selector.cands) > n_before
            if step > 0 and not guarded:
                # fail-safe: the guard could not run (no time, or the screen failed) -> the final still enters the pool
                # unscreened rather than the run regressing to an earlier state or the identity
                src = lora.state_from_flat([e.clone() for e in ema]) if ema is not None else lora.state()
                fin = Candidate(tag_prefix + ("ema" if ema is not None else "raw"), step,
                                {n: (u.detach().cpu().clone(), d.detach().cpu().clone()) for n, (u, d) in src.items()}, lora.scale, member=member)
                self.selector.add(fin)
                log(f"m{member} end guard did not run at step {step}: final {fin.tag} kept unscreened")
            if plan_steps and step < (2 * plan_steps) // 3:
                # v12 C5: the rounding rule K = round(avail / T) tolerates at most half a member of slack, so a member
                # cut below 2/3 of its plan (< K/(K+1/2) at K = 2) is not a comparable member
                self._cut_members.add(member)
                log(f"m{member} truncated at {step} of {plan_steps} planned steps: excluded from the final soup")
        else:
            if step > last_eval and not plateau_exit and budget.fits(budget.est_eval() * (2.2 if ema is not None else 1.2)):
                m_best = self._screen_candidates(guider, extra, member, step, ema, raw_hist, m_best, tag_prefix=tag_prefix, final=True)
            if len(raw_hist) >= 2 and budget.fits(budget.est_eval() * 1.2):
                m_best = self._screen_candidates(guider, extra, member, step, None, raw_hist, m_best, avg_only=True, tag_prefix=tag_prefix)
        m_best = m_best or self.best          # a member that never screened has no best of its own
        if sliced and not plateau_exit:
            plateau_exit = True               # the slice is over: hand the rest to the next member
        if toh and plan_steps and member + 1 < int(getattr(cfg, "member_slices", 0) or 0):
            plateau_exit = True               # v12: the planned members always follow
        self.summary.setdefault("member_seconds", []).append(round(time.time() - t_member, 1))
        log(f"member {member} done: steps={step} best={m_best.tag}@{m_best.step} {m_best.score:.6f} plateau_exit={plateau_exit}")
        return m_best, plateau_exit

    @staticmethod
    def _mem():
        try:
            return f"gpu {torch.cuda.memory_allocated() / 2**30:.1f}/{torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
        except Exception:  # noqa: BLE001 - CPU mock
            return "gpu n/a"

    def _confirm_count(self):
        """How many top candidates to confirm: at least --confirm-top, more when time is left over
        (a member that stopped at its plateau leaves it), never more than 8."""
        cfg = self.cfg
        per = max(1.0, self.budget.est_eval(cfg.confirm_noises))
        spare = self.budget.remaining() - 30.0 - self._sigma_head_reserve()   # remaining() already nets the publish reserve (v6.1 m6)
        k = int(spare // per)
        k = min(8, max(cfg.confirm_top, k))
        if k != cfg.confirm_top:
            log(f"confirm stage: {k} candidates fit at {cfg.confirm_noises} noises ({spare:.0f}s spare, {per:.0f}s each)")
        return k

    def _band_stop(self):
        """--band-stop (v12 L1) parsed to 16 per-band fractions, or None when off. 4 values cover the band groups
        0-3 / 4-9 / 10-13 / 14-15."""
        if getattr(self, "_band_stop_cache", "unset") != "unset":
            return self._band_stop_cache
        raw = str(getattr(self.cfg, "band_stop", "") or "").strip()
        stop = None
        if raw:
            vals = [float(v) for v in raw.split(",")]
            n = C.EVAL_STRATA
            if len(vals) == 4 and n == 16:
                vals = [vals[0]] * 4 + [vals[1]] * 6 + [vals[2]] * 4 + [vals[3]] * 2
            if len(vals) != n or not all(0.0 < v <= 1.0 for v in vals):
                raise ValueError(f"--band-stop {raw!r}: need 4 or {n} values in (0, 1]")
            stop = vals
            log(f"band stop (v12 L1): {stop}")
        self._band_stop_cache = stop
        return stop

    def _cases_at(self, cases, n_items, p):
        """The case cycle at progress p: every (image, band) pair once per pass (x the band repeats), shuffled; under
        --band-stop only the bands still open at p (cases of a closed band are dropped, the pass then refills from the
        open bands; at least the band with the latest stop stays open)."""
        stop = self._band_stop()
        if stop is None:
            if not cases:
                reps = self._band_repeats()
                cases = [(i, b) for i in range(n_items) for b in range(C.EVAL_STRATA) for _ in range(reps[b])]
                self.rng.shuffle(cases)
            return cases
        open_b = {b for b in range(C.EVAL_STRATA) if p < stop[b]} or {max(range(C.EVAL_STRATA), key=lambda b: stop[b])}
        cases = [c for c in cases if c[1] in open_b]
        if not cases:
            reps = self._band_repeats()
            cases = [(i, b) for i in range(n_items) for b in sorted(open_b) for _ in range(reps[b])]
            self.rng.shuffle(cases)
        if self.summary.get("band_stop_open") != len(open_b):
            self.summary["band_stop_open"] = len(open_b)
            self.summary.setdefault("band_stop_log", []).append([round(p, 4), sorted(open_b)])
        return cases

    def _band_repeats(self):
        """How many times each sigma band appears per case-cycle pass: 1 (uniform) unless --band-power
        is set, then (loss_b / median loss)^p rounded, from the latest holdout per-band loss."""
        n = C.EVAL_STRATA
        bl = getattr(self, "band_loss", None)
        power = float(getattr(self.cfg, "band_power", 0.0) or 0.0)
        if power <= 0 or not bl or len(bl) != n:
            return [1] * n
        med = float(np.median(bl)) or 1.0
        reps = [max(1, int(round((float(v) / med) ** power))) for v in bl]
        if self.summary.get("band_repeats") != reps:
            self.summary["band_repeats"] = reps
            log(f"band repeats (power {power}): {reps}")
        return reps

    def _seed2_reserve(self):
        """What must remain after the seed2 member: its screen, the soup screen, one confirm, verify + slack."""
        b = self.budget
        return 2 * b.est_eval() + b.est_eval(self.cfg.confirm_noises) * (1.5 if self.twin else 1.0) + 60

    def _seed2(self, guider, extra, m0):
        """v7: a clean second seed for exactly s* steps on the training images (annealed to s*), then the
        soup of it and member 0's best. Both are screened on the untouched holdout and join the selector;
        the confirm stage and the 1-SE rule decide what ships."""
        cfg, budget = self.cfg, self.budget
        self.lora.reinit(cfg.seed + 1000 * 1)
        st = self._train_fixed(guider, extra, 1, m0.step, items=self.train_items, reserve=self._seed2_reserve())
        if st is None or not budget.fits(2 * budget.est_eval() + 30):
            self.summary["seed2"] = {"skipped": "did not finish", "steps": m0.step}
            return
        c1 = Candidate("seed2", m0.step, st, self.lora.scale, member=1)
        self._screen(guider, extra, c1)
        self.selector.add(c1)
        soup = Candidate("soup2", m0.step, soup_state([m0.state, st]), self.lora.scale, member=-1)
        self._screen(guider, extra, soup)
        self.selector.add(soup)
        for c in (c1, soup):
            if c.score < self.best.score:
                self.best = c
                self.save({n: (u.to(self.device), d.to(self.device)) for n, (u, d) in c.state.items()}, c.scale, c.score, f"{c.tag}@{c.step}")
        self.summary["seed2"] = {"s_star": m0.step, "seed2": c1.score, "soup": soup.score, "m0": m0.screen.score if m0.screen else None}
        log(f"seed2: member-0 best {m0.screen.score if m0.screen else float('nan'):.6f}, seed2 {c1.score:.6f}, soup {soup.score:.6f}")

    def _well_formed(self, m0):
        """Member 0's curve supports a blind second seed (--seed2): a confirmed-interior optimum (a later
        screened point exists and is worse than the best by more than the paired SE) at a non-trivial step."""
        if m0.member != 0 or m0.step < 50 or m0.screen is None:
            return False
        later = [c for c in self.selector.cands if c.member == 0 and c.screen is not None and c.step > m0.step]
        if not later:
            return False
        worse = 0
        for c in later:
            mean, se, n = c.screen.paired_diff(m0.screen)
            if n > 1 and mean > se:
                worse += 1
        return worse >= 1

    def _train_fixed(self, guider, extra, member, steps, items=None, reserve=None):
        """Blind member: exactly `steps` steps on `items` (default: the training images, never the
        holdout), cosine annealed over those steps, EMA state returned; no screens. Stops early (returns
        None) only on the budget or a second OOM."""
        cfg, lora, budget = self.cfg, self.lora, self.budget
        groups = [{"params": lora.up_params, "lr_mult": cfg.lora_plus_ratio}, {"params": lora.down_params, "lr_mult": 1.0}]
        if lora.plain is not None:
            groups.append({"params": [lora.plain.param], "lr_mult": 1.0})
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8)
        ema = [p.detach().clone() for p in lora.params] if cfg.ema > 0 else None
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7919 * member)
        items = items if items is not None else self.train_items
        cases = []
        step = 0
        t_start = time.time()
        while step < steps:
            keep = reserve if reserve is not None else budget.est_eval(cfg.confirm_noises) * 2 + 60
            if not budget.fits(budget.est_step() + keep):
                log(f"m{member} (blind) stopped at step {step}/{steps}: budget")
                return None
            p = step / max(1, steps)
            warm = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
            lr = cfg.lr * warm * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            cases = self._cases_at(cases, len(items), p)
            i, band = cases.pop()
            item = items[i]
            sigma = sigma_of(band)
            noise = torch.randn(item["scaled"].shape, generator=gen, dtype=torch.float32).to(self.device)
            prompt = item["caption"] if (self.summary["steps"] % 2 == 0) else ""
            ts = time.time()
            try:
                with torch.enable_grad():
                    noisy, target = make_noisy(item["scaled"], noise, sigma)
                    den = self._forward(guider, extra, prompt, item, noisy, sigma)
                    loss = flow_prediction_mse(noisy, den, target, sigma).mean()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(lora.params, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            except (ValueError, FloatingPointError):
                opt.zero_grad(set_to_none=True)
                noisy = den = loss = None
                self.summary["nonfinite"] = self.summary.get("nonfinite", 0) + 1
                continue
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                noisy = den = loss = None
                torch.cuda.empty_cache()
                self.summary["oom"] = self.summary.get("oom", 0) + 1
                if not self._ckpt_on:
                    enable_checkpointing(self.model.model.diffusion_model, stride=1)
                    self._ckpt_on = True
                    cases.append((i, band))
                    log(f"m{member} (blind) OOM at step {step}: gradient checkpointing enabled, retrying ({self._mem()})")
                    continue
                log(f"m{member} (blind) OOM again at step {step}: member abandoned ({self._mem()})")
                return None
            if ema is not None:
                with torch.no_grad():
                    for e, q in zip(ema, lora.params):
                        e.mul_(cfg.ema).add_(q.detach(), alpha=1 - cfg.ema)
            budget.observe_step(time.time() - ts)
            step += 1
            self.summary["steps"] += 1
            if step % 100 == 0 or step == steps:
                log(f"m{member} (blind, all {len(items)} imgs) step {step}/{steps} loss {loss.item():.4f} lr {lr:.2e} {self._mem()} {time.time() - t_start:.0f}s")
        if ema is not None:
            return {n: (t[0].detach().cpu().clone(), t[1].detach().cpu().clone()) for n, t in lora.state_from_flat([e.clone() for e in ema]).items()}
        return {n: (u.detach().cpu().clone(), d.detach().cpu().clone()) for n, (u, d) in lora.state().items()}

    def _screen_candidates(self, guider, extra, member, step, ema, raw_hist, m_best, avg_only=False, tag_prefix="", final=False,
                           ema_only_final=False):
        lora = self.lora
        cands = []
        if avg_only:
            cands.append(Candidate(tag_prefix + "avg2", step, soup_state(raw_hist[-2:]), lora.scale, member=member))
        else:
            raw_state = {n: (u.detach().cpu().clone(), d.detach().cpu().clone()) for n, (u, d) in lora.state().items()}
            ema_only = (ema is not None and getattr(self.cfg, "screen_ema_only", False) and not final) or ema_only_final
            if not ema_only:
                cands.append(Candidate(tag_prefix + "raw", step, raw_state, lora.scale, member=member))
            if ema is not None:
                cands.append(Candidate(tag_prefix + "ema", step, {n: (t[0].cpu(), t[1].cpu()) for n, t in lora.state_from_flat([e.clone() for e in ema]).items()}, lora.scale, member=member))
            raw_hist.append(raw_state)
            del raw_hist[:-2]                     # only the last two raw states are ever used (v6.1 m5)
        for c in cands:
            c.state = {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in c.state.items()}
            try:
                self._screen(guider, extra, c)
            except (torch.OutOfMemoryError, ValueError, FloatingPointError) as e:
                torch.cuda.empty_cache()
                key = "oom" if isinstance(e, torch.OutOfMemoryError) else "nonfinite"
                self.summary[key] = self.summary.get(key, 0) + 1
                log(f"screen {key} at step {step} ({c.tag}): candidate skipped ({self._mem()})")
                c.state = {n: (u.cpu(), d.cpu()) for n, (u, d) in c.state.items()}
                continue
            c.state = {n: (u.cpu(), d.cpu()) for n, (u, d) in c.state.items()}
            self.selector.add(c)
            if m_best is None or c.score < m_best.score:
                m_best = c
            if c.score < self.best.score:
                self.best = c
                self.save({n: (u.to(self.device), d.to(self.device)) for n, (u, d) in c.state.items()}, c.scale, c.score, f"m{member}:{c.tag}@{step}")
        return m_best
