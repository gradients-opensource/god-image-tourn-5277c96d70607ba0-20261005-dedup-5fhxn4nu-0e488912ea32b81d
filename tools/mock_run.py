"""CPU end-to-end exercise of crown.engine with a mock model — no Comfy, no GPU.

Stands in for the Comfy pieces the engine touches (model patcher, guider,
process_conds) with tiny fakes, then runs the real Trainer: identity save,
screens, plateau -> second member -> soup, confirm, 1-SE pick, artifact +
summary.json. Catches control-flow bugs before they cost GPU time.

    python tools/mock_run.py [--seconds 60] [--out /tmp/mockrun]
"""

import argparse
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# --- fake comfy modules the engine imports lazily --------------------------------
comfy = types.ModuleType("comfy")
sampler_helpers = types.ModuleType("comfy.sampler_helpers")
sampler_helpers.convert_cond = lambda c: c
samplers = types.ModuleType("comfy.samplers")
samplers.process_conds = lambda model, noise, conds, device, lat, mask, seed: conds
comfy_sd = types.ModuleType("comfy.sd")
class _FakePatched:
    def __init__(self, base, lora_sd):
        self.base, self.lora_sd = base, lora_sd
        self.patches = {k: 1 for k in lora_sd if k.endswith("lora_up.weight")}
def _fake_load_lora_for_models(model, clip, lora, sm, sc):
    # mimic Comfy: every lora key must map onto a real diffusion_model.<name>.weight
    names = {n for n, m in model.model.diffusion_model.named_modules() if isinstance(m, torch.nn.Linear)}
    import logging
    for k in lora:
        base = k[len("diffusion_model."):].rsplit(".", 2)[0] if k.endswith(("lora_up.weight", "lora_down.weight")) else k[len("diffusion_model."):].rsplit(".", 1)[0]
        if base not in names:
            logging.getLogger().warning("lora key not loaded: %s", k)
    return _FakePatched(model, lora), None
comfy_sd.load_lora_for_models = _fake_load_lora_for_models
comfy_mm = types.ModuleType("comfy.model_management")
comfy_mm.load_models_gpu = lambda models, *a, **k: None
comfy.sampler_helpers, comfy.samplers, comfy.sd, comfy.model_management = sampler_helpers, samplers, comfy_sd, comfy_mm
sys.modules.update({"comfy": comfy, "comfy.sampler_helpers": sampler_helpers, "comfy.samplers": samplers, "comfy.sd": comfy_sd, "comfy.model_management": comfy_mm})

from crown import contract as C  # noqa: E402
from crown import engine  # noqa: E402
from crown.budget import Budget  # noqa: E402

C.EVAL_STRATA = 16


class HookLinear(torch.nn.Linear):
    """nn.Linear that applies Comfy-style weight functions in forward."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight_function = []

    def forward(self, x):
        w = self.weight
        for f in self.weight_function:
            w = f(w)
        return torch.nn.functional.linear(x, w, self.bias)


class TinyFlow(torch.nn.Module):
    """Velocity predictor on flattened latents; base has a fixed error the LoRA can learn away."""

    def __init__(self, dim, cond_dim=8, hidden=64):
        super().__init__()
        self.inp = HookLinear(dim + 1 + cond_dim, hidden)
        self.blocks = torch.nn.ModuleList([HookLinear(hidden, hidden) for _ in range(4)])
        self.out = HookLinear(hidden, dim)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_t, sigma, cond):
        h = torch.cat([x_t.flatten(1), sigma[:, None], cond.expand(x_t.shape[0], -1)], 1)
        h = torch.tanh(self.inp(h))
        for b in self.blocks:
            h = torch.tanh(b(h)) + h
        return self.out(h).reshape(x_t.shape)


class FakeModel:
    def __init__(self, dm):
        self.load_device = torch.device("cpu")
        self.model = types.SimpleNamespace(diffusion_model=dm, process_latent_in=lambda l: l)

    def add_weight_wrapper(self, key, fn):
        name = key[len("diffusion_model."):-len(".weight")]
        self.model.diffusion_model.get_submodule(name).weight_function.append(fn)


class FakeGuider:
    """Callable like CFGGuider: (noisy, t, **extra) -> denoised = x_t - sigma * v_pred."""

    def __init__(self, model, cond_vecs):
        self.model, self.cond_vecs, self.conds, self.inner_model = model, cond_vecs, None, object()

    def __call__(self, noisy, t, **extra):
        cond = self.cond_vecs[self.conds["positive"]]
        v = self.model.model.diffusion_model(noisy, t, cond)
        return noisy - t[:, None, None, None] * v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "crown-mock"))
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--max-members", type=int, default=2)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--plateau-window", type=int, default=3)
    ap.add_argument("--screen-ema-only", action="store_true")
    ap.add_argument("--adaptive-cadence", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    Cc, H, W = 4, 8, 8
    dim = Cc * H * W

    # dataset: latents drawn around a few "concepts"; captions map to cond vectors
    prompts = ["", "a red cube", "a blue sphere", "a green cone"]
    cond_vecs = {p: (torch.zeros(1, 8) if p == "" else torch.randn(1, 8)) for p in prompts}
    items = []
    for i in range(a.n_images):
        cap = prompts[1 + i % 3]
        x = torch.randn(1, Cc, H, W) * 0.5 + 0.3 * cond_vecs[cap].mean()
        items.append({"name": f"{i:03d}.png", "caption": cap, "sha256": f"{i:064x}", "scaled": x, "holdout": i % 4 == 0})
    dm = TinyFlow(dim)
    model = FakeModel(dm)
    guider = FakeGuider(model, cond_vecs)

    cfg = types.SimpleNamespace(
        family="mock", seed=0, include=None, exclude=None, rank=4, alpha=4.0, ckpt=False, ckpt_stride=1,
        lr=3e-3, lr_final_frac=0.1, warmup_steps=5, weight_decay=0.0, grad_clip=1.0, lora_plus_ratio=1.0, ema=0.99,
        eval_share=0.25, confirm_top=3, confirm_noises=4, phase2=True, max_members=a.max_members,
        polish=a.polish, polish_lr_frac=0.3, band_power=0.0, plateau_window=a.plateau_window, screen_ema_only=a.screen_ema_only,
        empty_prompt_frac=0.5, select_metric="mean", eval_every=0, holdout_names="", flip=False, adaptive_cadence=a.adaptive_cadence, screen_passes="", screen_max_share=0.3, seed2=False)
    budget = Budget(time.time() + a.seconds, kill_margin_s=0.0, publish_reserve_s=2.0)
    os.makedirs(a.out, exist_ok=True)
    os.environ["GOD_TRAIN_LOGS"] = "1"

    class FakeTwin:
        """Same numerics as the hook path in the mock; exercises the twin control flow + loader check."""
        source = "mock"
        base = model                      # the evaluator-path final check verifies against the twin's base
        def __init__(self, tr_ref):
            self.tr_ref = tr_ref
        def _load(self):
            pass
        def release(self):
            pass
        def score(self, state, scale, noises=1):
            tr = self.tr_ref[0]
            from crown.twin import apply_lora_like_evaluator
            _, missing = apply_lora_like_evaluator(model, state, scale)
            if missing:
                return None, missing
            tr.lora.set_tensors(state, scale)
            rep = tr.scorer.score(guider, {"model_options": {}, "seed": 42}, lambda p, it: tr._set_cond(guider, p, it), noises=noises)
            tr.lora.restore()
            return rep, []
    ref = []
    tr = engine.Trainer(cfg, model, items, {p: p for p in prompts}, a.out, budget, twin=FakeTwin(ref))
    ref.append(tr)
    tr.run(guider, {"model_options": {}, "seed": 42})

    s = json.load(open(os.path.join(a.out, "summary.json")))
    print("\n=== summary ===")
    print("base", round(s["base_score"], 6), "| final", s["final"]["picked"], "@", s["final"]["step"], "member", s["final"]["member"],
          "score", round(s["final"]["score"], 6), "rel", round(s["final"]["rel_pct"], 2), "%")
    print("steps", s["steps"], "| screens", len(s["evals"]), "| confirms", len(s["confirms"]), "| members", [(m["member"], m["best_tag"], m["best_step"], m["plateau_exit"]) for m in s["members"]])
    print("parity:", s.get("parity"), "| final_loadable:", s.get("final_loadable"), "| confirm via:", {c.get("via", "hook") for c in s["confirms"]})
    art = os.path.join(a.out, C.OUTPUT_LORA_NAME)
    from safetensors.torch import load_file

    sd = load_file(art)
    ups = [k for k in sd if k.endswith("lora_up.weight")]
    print("artifact", os.path.getsize(art), "bytes |", len(ups), "targets | alpha sample", float(sd[ups[0].replace("lora_up.weight", "alpha")]), "| rank", sd[ups[0].replace("lora_up", "lora_down")].shape[0])
    ok = s["final"]["score"] <= s["base_score"] and len(ups) > 0 and len(s["evals"]) >= 3
    print("MOCK RUN", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
