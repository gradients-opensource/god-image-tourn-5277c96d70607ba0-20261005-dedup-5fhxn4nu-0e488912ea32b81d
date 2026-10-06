"""Evaluator twin: score candidate LoRA states through the evaluator's own code path.

Why. On two families the numbers the evaluator computes differ from what our
training hook computes, no matter how careful the hook is:
  - ideogram4: the evaluator loads a per-tensor `comfy_quant` fp8 file (not the
    per-row copy in the task cache), merges the LoRA with stochastic rounding
    seeded by crc32(key), and runs fp8 activations through `_scaled_mm`;
  - qwen-image: same file as training, but the merge is stochastic rounding to
    plain fp8, not round-to-nearest.
Rather than re-implementing those numerics, the twin *is* the evaluator in
process: the base is loaded the way `image_artifacts.prepare_base` loads it,
each candidate is applied with `comfy.sd.load_lora_for_models` (what
`nodes.LoraLoader` calls), and the forward runs under `CFGGuider.sample`
exactly as `evaluators/diffusion.py` does. The candidate is exported to the same
tensor dict we would upload, so a key the evaluator cannot match is caught here.

Cost: a second model resident (ideogram4 9.3 GB fp8, qwen-image 20 GB fp8) and
a patch/unpatch per scored candidate. Used for the confirm stage only.
"""

import logging
import os

import torch

from . import contract as C
from .lora import export_state_dict
from .scoring import HoldoutScorer


class _MissingKeys(logging.Handler):
    def __init__(self):
        super().__init__()
        self.missing = []

    def emit(self, record):
        m = record.getMessage()
        if m.startswith(("lora key not loaded:", "NOT LOADED ")):
            self.missing.append(m)


def apply_lora_like_evaluator(model, state, scale):
    """Return (patched_clone, missing) — `missing` non-empty or zero patches means the
    evaluator would raise 'Incomplete LoRA application' on this artifact."""
    import comfy.sd

    lora_sd = export_state_dict(state, scale)
    h = _MissingKeys()
    root = logging.getLogger()
    root.addHandler(h)
    try:
        patched, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, 1.0, 0.0)
    finally:
        root.removeHandler(h)
    if patched is None or not patched.patches:
        h.missing.append("zero model patches")
    return patched, h.missing


def verify_loadable(model, state, scale):
    """G0 in process: does this state load with zero unmatched keys? Returns (ok, missing)."""
    patched, missing = apply_lora_like_evaluator(model, state, scale)
    return (not missing), missing


class _ScoreSampler:
    def __init__(self, twin, noises, noise_offset=0):
        self.twin, self.noises, self.noise_offset, self.report = twin, noises, noise_offset, None

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        guider = model_wrap
        self.report = self.twin.scorer.score(guider, extra_args, lambda p, it: self.twin._set_cond(guider, p, it), noises=self.noises,
                                             noise_offset=self.noise_offset)
        return latent_image


E4M3_MARKER = b'{"format": "float8_e4m3fn"}'   # the comfy_quant bytes of every quantised Linear in the evaluator's file


def rebuild_ideogram4_evaluator_base(raw_sd, device):
    """The evaluator's hard-coded ideogram4 base (Comfy-Org ideogram4_fp8_scaled.safetensors: every quantised
    Linear is the raw weight cast to e4m3 with weight_scale 1.0 and a comfy_quant marker; bias and the other
    tensors bf16) rebuilt from the task cache's per-row-scaled fp8 copy of the same weights: dequantise
    (fp8 x row scale, fp32) and cast to e4m3, tensor by tensor on `device`. Measured 2026-09-26: 98.90 % of the
    file's codes, the 206 other tensors bitwise identical, evaluator scores within 0.058 % on four LoRAs in the
    same rank order - so the 8.7 GB file is no longer baked into the image."""
    marker = torch.tensor(list(E4M3_MARKER), dtype=torch.uint8)
    out = {}
    for k, v in raw_sd.items():
        if k.endswith(".weight_scale"):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base is not None and v.dtype == torch.float8_e4m3fn and (base + ".weight_scale") in raw_sd:
            s = raw_sd[base + ".weight_scale"]
            w = v.to(device).float() * s.to(device).float().reshape(-1, *([1] * (v.dim() - 1)))
            out[k] = w.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).cpu()
            out[base + ".weight_scale"] = torch.tensor(1.0, dtype=torch.float32)
            out[base + ".comfy_quant"] = marker.clone()
        else:
            out[k] = v
    return out, {"model_type": "ideogram4_cond"}


class EvaluatorTwin:
    FAMILIES = ("ideogram4", "qwen-image")

    def __init__(self, family, assets, holdout_items, raw_conds, device, baked_dir=None):
        from . import engine  # local import: engine imports twin

        self.family, self.raw_conds, self.device = family, raw_conds, device
        baked_dir = baked_dir or os.environ.get("CROWN_ASSETS", str(C.BAKED_DIR))
        import comfy.utils

        if family == "ideogram4":
            path = os.path.join(baked_dir, C.BAKED_IDEOGRAM4_BASE)
            if os.path.exists(path):
                sd, meta = comfy.utils.load_torch_file(path, safe_load=True, device=torch.device("cpu"), return_metadata=True)
                self.source = path
            else:
                sd, meta = rebuild_ideogram4_evaluator_base(assets.ideogram4_raw_sd(), device)
                self.source = "rebuilt from the task cache (e4m3 cast, scale 1)"
        else:
            # qwen-image, or any family whose base file is fp8 (e.g. an fp8 FLUX checkpoint): the same
            # file the evaluator picks, merged the evaluator's way (v6.1 F1)
            sd, meta = assets.diffusion_sd(), assets.diffusion_meta()
            if not getattr(assets, "base_fp8", False) and family != "qwen-image":
                raise ValueError(f"twin not needed for {family}: bf16 hook path == evaluator numerics")
            self.source = "cache (same file as evaluator)"
        # the base is built on demand (score) and released afterwards: it must not occupy the GPU
        # while the trainer runs (9.3 GB on ideogram4, 20 GB on qwen-image)
        self._sd, self._meta = sd, meta
        self.base = None
        self.items = holdout_items
        self.scorer = HoldoutScorer(holdout_items, None, device)
        self._processed = {}
        self._active = None

    # conditioning processed against the *patched* twin's inner model, per (prompt, shape)
    def _set_cond(self, guider, prompt, item):
        key = (prompt, tuple(item["scaled"].shape))
        c = self._processed.get(key)
        if c is None:
            import comfy.sampler_helpers
            import comfy.samplers

            lat = item["scaled"].to(self.device)
            conds = {"positive": comfy.sampler_helpers.convert_cond(self.raw_conds[prompt]),
                     "negative": comfy.sampler_helpers.convert_cond([])}
            c = comfy.samplers.process_conds(guider.inner_model, torch.zeros_like(lat), conds, self.device, lat, None, C.EVAL_SAMPLER_SEED)
            self._processed[key] = c
        guider.conds = c

    def _load(self):
        from . import engine

        if self.base is None:
            # Comfy's loader pops keys out of the dict it is given: hand it a copy so the base
            # can be rebuilt for every score() call
            self.base = engine.load_diffusion_model(dict(self._sd), self.family, dict(self._meta) if self._meta else self._meta)

    def release(self):
        """Drop the twin's model (and any patched clone) from the GPU; the caller reloads the
        training model afterwards."""
        import gc

        import comfy.model_management as mm

        self.base = None
        self._processed = {}
        gc.collect()
        mm.unload_all_models()
        torch.cuda.empty_cache()

    def patched_weights(self, state, scale, keys):
        """The weights the evaluator's ModelPatcher produces for `keys` under this LoRA state
        (patch_weight_to_device with return_weight: the merge and, for fp8, the stochastic re-rounding),
        upcast to bf16 on the CPU. Empty dict if the state would not load."""
        try:
            self._load()
            patched, missing = apply_lora_like_evaluator(self.base, state, scale)
            if missing:
                return {}
            return {k: patched.patch_weight_to_device(k, device_to=self.device, return_weight=True).to(torch.bfloat16).cpu()
                    for k in keys}
        finally:
            del state
            self.release()

    def score(self, state, scale, noises=1, noise_offset=0):
        """ScoreReport for `state` computed the evaluator's way, or None (+reason) if it
        would not load. The base is loaded for the call and released afterwards; the
        conditioning cache is per patched model, so it is reset each call. `noise_offset`
        (v11 sigma head) scores the evaluator's draws [offset, offset + noises)."""
        from . import engine

        try:
            self._load()
            patched, missing = apply_lora_like_evaluator(self.base, state, scale)
            if missing:
                return None, missing
            self._processed = {}
            guider = engine.make_guider(patched, self.raw_conds[""])
            sampler = _ScoreSampler(self, noises, noise_offset)
            lat0 = self.items[0]["scaled"].to(self.device)
            guider.sample(torch.zeros_like(lat0), lat0, sampler, engine.evaluator_schedule(), disable_pbar=True, seed=C.EVAL_SAMPLER_SEED)
            return sampler.report, []
        finally:
            del state
            self.release()
