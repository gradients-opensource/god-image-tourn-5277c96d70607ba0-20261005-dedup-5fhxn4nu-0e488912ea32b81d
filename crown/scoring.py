"""Evaluator-identical scoring.

This module reproduces validator/evaluation/denoising_mse.py and the scoring
loop of evaluators/diffusion.py *exactly*: same seeds, same sigmas, same loss,
same aggregation. It is the training objective and the selection metric, so
that what we optimise is what we are scored on.

Everything here is deterministic. Two calls with the same LoRA state return the
same numbers; two candidates are compared on the same noise draws (paired).
"""

import hashlib
import math

import numpy as np
import torch

from . import contract as C


# --- seeds and cases (denoising_mse.py) ----------------------------------------

def case_seed(image_hash: str, stratum: int, noise_index: int, master_seed: int = C.EVAL_MASTER_SEED) -> int:
    key = f"{master_seed}:{image_hash}:{stratum}:{noise_index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def sigma_of(band: int, strata: int = C.EVAL_STRATA) -> float:
    # Midpoints exclude the unstable endpoints while covering uniform flow time.
    return (band + 0.5) / strata


def case_noise(shape, image_hash: str, band: int, k: int) -> torch.Tensor:
    """The exact noise tensor the evaluator draws for (image, band, k). CPU, fp32."""
    g = torch.Generator().manual_seed(case_seed(image_hash, band, k))
    return torch.randn(shape, generator=g, dtype=torch.float32)


def flow_prediction_mse(noisy, denoised, target, sigma: float) -> torch.Tensor:
    """Per-case MSE on the velocity target. Comfy returns x0 = x_t - sigma * v."""
    predicted = (noisy.float() - denoised.float()) / sigma
    losses = (predicted - target.float()).square().flatten(1).mean(1)
    if not torch.isfinite(losses).all():
        raise ValueError("Non-finite denoising MSE")
    return losses


def make_noisy(latent: torch.Tensor, noise: torch.Tensor, sigma: float):
    """(noisy, target) for one case. `latent` is the model-scaled latent."""
    return (1.0 - sigma) * latent + sigma * noise, noise - latent


# --- holdout scorer (evaluators/diffusion.py, image_denoising.py) ---------------

class HoldoutScorer:
    """Scores a set of held-out images the way the evaluator scores the test set.

    `items` are dicts with keys: sha256, scaled (model-scaled latent, 1xCxHxW on
    CPU or device), caption.  `conds` maps prompt -> conditioning; "" must be
    present.  `model_wrap` and `extra_args` are what CFGGuider hands its sampler.

    Returns a ScoreReport with everything needed for paired comparison.
    """

    def __init__(self, items, conds, device, strata: int = C.EVAL_STRATA, batch: int = C.EVAL_BATCH):
        self.items = list(items)
        self.conds = conds
        self.device = device
        self.strata = strata
        self.batch = batch
        # noise is fixed per (image, band, k): generate once, keep on CPU
        self._noise_cache = {}

    def _noise(self, item, band, k):
        key = (item["sha256"], band, k)
        n = self._noise_cache.get(key)
        if n is None:
            n = case_noise(item["scaled"].shape, item["sha256"], band, k)
            self._noise_cache[key] = n
        return n

    @torch.no_grad()
    def score(self, model_wrap, extra_args, set_cond, noises: int = 1, noise_offset: int = 0):
        """Score with `noises` draws per stratum starting at index `noise_offset`.

        noises=1, offset=0 is the cheap screen (the evaluator's noise index 0).
        noises=16, offset=0 is the evaluator's full case set.
        `set_cond(prompt, item)` must switch the guider's conditioning before
        forwards; it receives the item because conditioning is processed against
        that image's latent shape, exactly as the evaluator does per image.
        """
        per_case = {}   # (sha, mode, band, k) -> loss
        per_image = []
        text_means, notext_means = [], []
        for item in self.items:
            x = item["scaled"].to(self.device)
            modes = {}
            for mode in ("text", "no_text"):
                prompt = item["caption"] if mode == "text" else ""
                set_cond(prompt, item)
                losses = []
                for band in range(self.strata):
                    s = sigma_of(band, self.strata)
                    ks = list(range(noise_offset, noise_offset + noises))
                    for start in range(0, len(ks), self.batch):
                        chunk = ks[start:start + self.batch]
                        noise = torch.cat([self._noise(item, band, k) for k in chunk]).to(self.device)
                        noisy, target = make_noisy(x.expand(len(chunk), *x.shape[1:]), noise, s)
                        t = torch.full((noisy.shape[0],), s, device=self.device, dtype=torch.float32)
                        den = model_wrap(noisy, t, **extra_args)
                        l = flow_prediction_mse(noisy, den, target, s).cpu().tolist()
                        for k, v in zip(chunk, l):
                            per_case[(item["sha256"], mode, band, k)] = v
                        losses += l
                modes[mode] = float(np.mean(losses))
            text_means.append(modes["text"]); notext_means.append(modes["no_text"])
            per_image.append(C.EVAL_TEXT_WEIGHT * modes["text"] + (1.0 - C.EVAL_TEXT_WEIGHT) * modes["no_text"])
        return ScoreReport(float(np.mean(per_image)), per_image, text_means, notext_means, per_case)


class ScoreReport:
    __slots__ = ("score", "per_image", "text", "no_text", "per_case")

    def __init__(self, score, per_image, text, no_text, per_case):
        self.score, self.per_image, self.text, self.no_text, self.per_case = score, per_image, text, no_text, per_case

    def paired_diff(self, other: "ScoreReport"):
        """Mean and standard error of (self - other) over shared cases.

        Cases are identified by (image, mode, band, k); with deterministic seeds the
        same key is the same noise draw, so this is a paired comparison and its SE
        is the right yardstick for 'is candidate A really better than B'.
        """
        keys = [k for k in self.per_case if k in other.per_case]
        if not keys:
            return float("nan"), float("nan"), 0
        d = np.array([self.per_case[k] - other.per_case[k] for k in keys], dtype=np.float64)
        n = len(d)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), n
