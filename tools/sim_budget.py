"""Budget / time-policy simulator: the REAL crown engine on a virtual clock (no GPU).

Why: three GPU runs were spent learning that a time policy could not fit (flux plateau declared at 27 min,
krea2 exit gate, krea2 arithmetic). Every timing decision the engine makes — screen cadence, plateau
exit, second member, seed2 fit, confirm count, final slack before the deadline — depends only on
measured costs and the shape of the holdout curve. This harness runs the actual engine code with:
  * a virtual clock (crown.engine.time / crown.budget.time patched) advanced by the MEASURED step time
    per optimiser step, the measured screen cost per holdout image per noise, the measured load time;
  * synthetic holdout scores from a family's MEASURED curve expressed in passes over the (image, band)
    cases (1 pass = n_train * 16 steps), with per-case noise so paired SEs are realistic;
  * the tiny mock model of tools/mock_run.py for the actual training arithmetic (cheap, real control flow).
It reports the timeline, what the engine chose, the regret of the shipped pick against the curve's true
minimum, and the slack left before the validator's clock.

    python tools/sim_budget.py --family flux --hours 0.75 --n-train 24 --n-holdout 3 [--policy v7-passes+seed2]
    python tools/sim_budget.py --grid                       # every family x regime x policy -> table
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import types
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
_spec = importlib.util.spec_from_file_location("mock_run", os.path.join(HERE, "mock_run.py"))
mock_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock_run)                     # installs the fake comfy modules

import torch  # noqa: E402

from crown import budget as budget_mod  # noqa: E402
from crown import engine  # noqa: E402
from crown import scoring  # noqa: E402
from crown.selector import Candidate  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# Measured family profiles (docs/research_v7.md, facts from /root/facts/runs.json, 2026-09-25).
# curve: (passes, relative loss % vs base) of the member-0 EMA screens; drift continues linearly after
# the last point with the slope of the last segment. step_s: s per optimiser step; img_s: s per holdout
# image per noise (1-noise screen); load_s: entrypoint start -> first screen; twin: evaluator-twin family.
PROFILES = {
    "flux":      {"step_s": 1.03, "img_s": 9.6, "load_s": 30, "twin": False,
                  "curve": [(0, 0.0), (2.3, -6.85), (4.5, -6.30), (6.8, -5.40), (9.1, -4.05), (11.2, -3.13)],   # PixelWave, 9 train imgs
                  "note": "optimum location below 2.3 passes unobserved (first screen)"},
    "ideogram4": {"step_s": 0.70, "img_s": 9.1, "load_s": 70, "twin": True,
                  "curve": [(0, 0.0), (1.2, -9.74), (2.4, -10.63), (3.6, -10.19), (4.7, -9.69), (5.9, -9.08)]},   # requant, 24 imgs
    "z-image":   {"step_s": 0.66, "img_s": 6.4, "load_s": 25, "twin": False,
                  "curve": [(0, 0.0), (0.8, -27.78), (1.7, -28.31), (2.5, -28.32), (3.4, -28.44), (4.2, -28.57), (5.0, -28.49),
                            (5.9, -28.43), (6.7, -28.27), (7.5, -28.08), (8.4, -28.00), (9.9, -27.95)]},
    "krea2":     {"step_s": 1.24, "img_s": 12.1, "load_s": 23, "twin": False,
                  "curve": [(0, 0.0), (1.2, -1.58), (2.4, -2.37), (3.7, -2.13), (4.9, -1.61)]},
    "qwen-image": {"step_s": 0.64, "img_s": 6.8, "load_s": 34, "twin": True,
                   "curve": [(0, 0.0), (0.8, -4.36), (1.7, -5.38), (2.5, -5.64), (3.4, -5.85), (4.2, -6.01), (5.0, -6.07), (5.9, -6.18), (6.7, -6.20)]},
}
RAW_PENALTY = 0.12     # raw screens sit this many % points above the EMA at the same step (measured median)
SOUP_GAIN = 0.30       # a soup of two distinct members: % points below the better member (flux 2026-09-25: -0.32%)
TWIN_RELOAD_S = 20.0   # twin: base build + release + training-model reload per call
VERIFY_S = 8.0         # final loadability check


def curve_at(profile, passes):
    pts = profile["curve"]
    if passes <= pts[0][0]:
        return pts[0][1]
    for (p0, r0), (p1, r1) in zip(pts, pts[1:]):
        if passes <= p1:
            return r0 + (r1 - r0) * (passes - p0) / (p1 - p0)
    (p0, r0), (p1, r1) = pts[-2], pts[-1]
    return r1 + (r1 - r0) / (p1 - p0) * (passes - p1)


def true_min(profile, max_passes):
    grid = [i * 0.05 for i in range(int(max_passes / 0.05) + 1)]
    return min(curve_at(profile, p) for p in grid)


class VClock:
    def __init__(self, t0=1_000_000.0):
        self.t = t0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def strftime(self, fmt):
        import time as _t
        return _t.strftime(fmt, _t.gmtime(self.t))


class Sim:
    def __init__(self, family, hours, n_train, n_holdout, policy, seed=0, case_noise=1.0, profile_override=None):
        self.p = dict(PROFILES[family], **(profile_override or {}))
        self.family, self.hours, self.n_train, self.n_holdout = family, hours, n_train, n_holdout
        self.policy, self.rng, self.seed = policy, random.Random(seed), seed
        self.case_noise = case_noise
        self.clock = VClock()
        self.current = None                # candidate being scored (set by the wrapped _screen/_confirm)
        self.member_best = {}              # member -> best true r so far (for soups)
        self.events = []
        self.pass_steps = n_train * 16

    # -- synthetic truth ---------------------------------------------------------------------------
    def true_r(self, cand):
        tag, member, step = cand.tag, cand.member, cand.step
        if tag in ("identity", "base") or step == 0 and member >= 0:
            return 0.0
        if member == -1:                                   # phase-2 soup of distinct member bests
            bests = [v for k, v in self.member_best.items() if 0 <= k < 100]
            return (min(bests) if bests else 0.0) - SOUP_GAIN
        passes = step / self.pass_steps
        r = curve_at(self.p, passes)
        base_tag = tag.replace("pol-", "")
        if base_tag == "raw":
            r += RAW_PENALTY
        elif base_tag == "avg2":
            r += RAW_PENALTY / 2
        return r

    def report(self, cand, noises):
        r = self.true_r(cand)
        if cand.member >= 0 and cand.tag not in ("identity", "base"):
            self.member_best[cand.member] = min(self.member_best.get(cand.member, 0.0), r)
        seen = r
        per_case, per_image, text, notext = {}, [], [], []
        # stable seed: Python's str hash is randomised per process (PYTHONHASHSEED), which made every
        # simulator number a single irreproducible draw (found 2026-09-26)
        rng = random.Random(zlib.crc32(f"{self.seed}|{cand.tag}|{cand.member}|{cand.step}|{noises}".encode()))
        for i in range(self.n_holdout):
            img_f = 0.8 + 0.4 * ((i * 7919) % 97) / 97.0     # fixed image difficulty
            vals = {"text": [], "no_text": []}
            for mode in ("text", "no_text"):
                for band in range(16):
                    base = img_f * (0.7 if band == 0 else 0.35 / (1 + band) + 0.05)
                    for k in range(noises):
                        v = base * (1 + (seen + rng.gauss(0, self.case_noise)) / 100.0)
                        per_case[(f"img{i}", mode, band, k)] = v
                        vals[mode].append(v)
            text.append(sum(vals["text"]) / len(vals["text"]))
            notext.append(sum(vals["no_text"]) / len(vals["no_text"]))
            per_image.append(0.5 * text[-1] + 0.5 * notext[-1])
        return scoring.ScoreReport(sum(per_image) / len(per_image), per_image, text, notext, per_case)

    # -- run -----------------------------------------------------------------------------------------
    def run(self):
        sim = self
        orig_time_engine, orig_time_budget = engine.time, budget_mod.time
        orig_score_scoring, orig_score_engine = scoring.HoldoutScorer.score, engine.HoldoutScorer.score
        engine.time = sim.clock
        budget_mod.time = sim.clock
        orig_observe = budget_mod.Budget.observe_step

        def observe_step(bself, dt):
            sim.clock.sleep(sim.p["step_s"])
            orig_observe(bself, sim.p["step_s"])
        budget_mod.Budget.observe_step = observe_step

        def fake_score(sself, model_wrap, extra_args, set_cond, noises=1, noise_offset=0):
            sim.clock.sleep(sim.p["img_s"] * sim.n_holdout * noises)
            return sim.report(sim.current, noises)
        scoring.HoldoutScorer.score = fake_score
        engine.HoldoutScorer.score = fake_score

        orig_screen, orig_confirm = engine.Trainer._screen, engine.Trainer._confirm

        def screen(tself, guider, extra, cand, noises=1):
            sim.current = cand
            rep = orig_screen(tself, guider, extra, cand, noises)
            sim.events.append((sim.minutes(), "screen", cand.member, cand.step, cand.tag, round(sim.true_r(cand), 3)))
            return rep

        def confirm(tself, guider, extra, cand):
            sim.current = cand
            sim.events.append((sim.minutes(), "confirm", cand.member, cand.step, cand.tag, round(sim.true_r(cand), 3)))
            return orig_confirm(tself, guider, extra, cand)
        engine.Trainer._screen, engine.Trainer._confirm = screen, confirm

        class Twin:
            source = "sim"
            base = None

            def _load(self):
                pass

            def release(self):
                pass

            def score(self, state, scale, noises=1):
                sim.clock.sleep(sim.p["img_s"] * sim.n_holdout * noises * 0.5 + TWIN_RELOAD_S)
                return sim.report(sim.current, noises), []

        orig_verify = engine.verify_loadable

        def verify(*a, **k):
            sim.clock.sleep(VERIFY_S)
            return True, []
        engine.verify_loadable = verify

        try:
            torch.manual_seed(0)
            Cc, H, W = 4, 8, 8
            prompts = ["", "cap"]
            cond_vecs = {p_: (torch.zeros(1, 8) if p_ == "" else torch.randn(1, 8)) for p_ in prompts}
            items = []
            for i in range(self.n_train + self.n_holdout):
                items.append({"name": f"{i:03d}.png", "caption": "cap", "sha256": f"{i:064x}",
                              "scaled": torch.randn(1, Cc, H, W), "holdout": i >= self.n_train})
            model = mock_run.FakeModel(mock_run.TinyFlow(Cc * H * W))
            guider = mock_run.FakeGuider(model, cond_vecs)
            pol = self.policy
            cfg = types.SimpleNamespace(
                family="sim", seed=0, include=None, exclude=None, rank=4, alpha=4.0, ckpt=False, ckpt_stride=1,
                lr=1e-3, lr_final_frac=0.1, warmup_steps=20, weight_decay=0.0, grad_clip=1.0, lora_plus_ratio=1.0, ema=0.99,
                eval_share=pol.get("eval_share", 0.15), confirm_top=3, confirm_noises=4, phase2=pol.get("phase2", True),
                max_members=pol.get("max_members", 2), polish=False, polish_lr_frac=0.3, band_power=0.0,
                plateau_window=pol.get("plateau_window", 3), screen_ema_only=pol.get("screen_ema_only", False),
                empty_prompt_frac=0.5, select_metric="mean", eval_every=pol.get("eval_every", 0), holdout_names="",
                flip=False, adaptive_cadence=pol.get("adaptive_cadence", False), twin=self.p["twin"],
                screen_passes=pol.get("screen_passes", ""), screen_max_share=pol.get("screen_max_share", 0.3), seed2=pol.get("seed2", False))
            deadline = self.clock.t + self.hours * 3600.0
            self.t_start = self.clock.t
            budget = budget_mod.Budget(deadline)
            self.clock.sleep(self.p["load_s"])
            out = tempfile.mkdtemp(prefix="sim-")
            os.environ["GOD_TRAIN_LOGS"] = "0"
            tr = engine.Trainer(cfg, model, items, {p_: p_ for p_ in prompts}, out, budget, twin=Twin() if self.p["twin"] else None)
            tr.run(guider, {"model_options": {}, "seed": 42})
            s = json.load(open(os.path.join(out, "summary.json")))
        finally:
            budget_mod.Budget.observe_step = orig_observe
            engine.Trainer._screen, engine.Trainer._confirm = orig_screen, orig_confirm
            engine.verify_loadable = orig_verify
            # the virtual clock and the fake scorer must not outlive the simulation: left in place they poisoned
            # every later check in the same process (mock_checks 2026-09-27: a check after seed2_fits_or_declines
            # read the stopped virtual clock -> a 0 s step and ZeroDivisionError, then a member that never ended)
            engine.time, budget_mod.time = orig_time_engine, orig_time_budget
            scoring.HoldoutScorer.score, engine.HoldoutScorer.score = orig_score_scoring, orig_score_engine
        fin = s.get("final") or {}
        picked = Candidate(fin.get("picked", "identity"), fin.get("step", 0), {}, 1.0, member=fin.get("member", 0))
        r_pick = self.true_r(picked)
        horizon_passes = s["steps"] / self.pass_steps + 1
        rmin = true_min(self.p, max(horizon_passes, 1.0))
        end_min = self.minutes()
        return {
            "family": self.family, "hours": self.hours, "n_train": self.n_train, "policy": self.policy.get("name", "?"),
            "pick": f"{fin.get('picked')}@{fin.get('step')} m{fin.get('member')}", "true_r_pick": round(r_pick, 3), "true_min": round(rmin, 3),
            "regret": round(r_pick - rmin, 3), "members": [(m["member"], m["best_step"], m["plateau_exit"]) for m in s.get("members", [])],
            "seed2": s.get("seed2"), "screens": sum(1 for e in self.events if e[1] == "screen"), "confirms": sum(1 for e in self.events if e[1] == "confirm"),
            "m0_exit_min": next((round(e[0], 1) for e in self.events if e[1] == "screen" and e[2] == 1), None),
            "end_min": round(end_min, 1), "slack_min": round(self.hours * 60 - end_min, 1), "steps": s["steps"],
        }

    def minutes(self):
        return (self.clock.t - self.t_start) / 60.0


SCHED = "0.5,1,1.5,2,2.5,3,3.5,4,5,6,8,10,12,16"
POLICIES = {
    "v6.3":            {"name": "v6.3"},
    "v6.3+adaptive":   {"name": "v6.3+adaptive", "adaptive_cadence": True},
    "v7-passes":       {"name": "v7-passes", "screen_passes": SCHED},
    "v7-passes+seed2": {"name": "v7-passes+seed2", "screen_passes": SCHED, "seed2": True},
}

REGIMES = [   # (family, hours, n_train, n_holdout, label)
    ("flux", 0.75, 24, 3, "R1"), ("ideogram4", 0.75, 24, 3, "R1"), ("z-image", 0.75, 24, 3, "R1"),
    ("krea2", 0.75, 20, 3, "R2"), ("qwen-image", 1.0, 30, 3, "R2"),
    ("krea2", 1.0, 28, 4, "boss"), ("ideogram4", 1.0, 30, 4, "boss"), ("z-image", 1.0, 32, 4, "boss"), ("qwen-image", 1.5, 40, 5, "boss"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="flux")
    ap.add_argument("--hours", type=float, default=0.75)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-holdout", type=int, default=3)
    ap.add_argument("--policy", default="v6.3", choices=sorted(POLICIES))
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--events", action="store_true")
    a = ap.parse_args()
    if not a.grid:
        sim = Sim(a.family, a.hours, a.n_train, a.n_holdout, POLICIES[a.policy])
        r = sim.run()
        if a.events:
            for e in sim.events:
                print(f"  {e[0]:6.1f} min  {e[1]:8} m{e[2]} step {e[3]} {e[4]:12} true {e[5]:+.3f}")
        print(json.dumps(r, indent=1))
        return
    print(f"{'regime':6} {'family':10} {'h':>4} {'ntr':>3} {'policy':14} {'pick':24} {'r_pick':>7} {'r_min':>7} {'regret':>6} {'scr':>3} {'cnf':>3} {'m1@':>5} {'end':>5} {'slack':>5}  members / seed2")
    for fam, h, ntr, nho, label in REGIMES:
        for pname, pol in POLICIES.items():
            r = Sim(fam, h, ntr, nho, pol).run()
            rps = ""
            if pol.get("seed2"):
                s2 = r.get("seed2") or {}
                rps = "seed2 " + (s2.get("skipped") or ("ran" if s2 else "no exit"))
            print(f"{label:6} {fam:10} {h:>4} {ntr:>3} {pname:14} {r['pick'][:24]:24} {r['true_r_pick']:>7} {r['true_min']:>7} {r['regret']:>6} "
                  f"{r['screens']:>3} {r['confirms']:>3} {str(r['m0_exit_min']):>5} {r['end_min']:>5} {r['slack_min']:>5}  {r['members']} {rps}")


if __name__ == "__main__":
    main()
