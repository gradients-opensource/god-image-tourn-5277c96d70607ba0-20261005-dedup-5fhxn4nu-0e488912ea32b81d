"""Defect-specific CPU checks for the v6.1 fixes, on top of tools/mock_run.py's fakes. Each check
builds a fresh Trainer on the tiny mock model and asserts the behaviour the audit found broken:

  identity_first      an artifact exists before the first screen (a failing first forward cannot leave nothing)
  restore_on_error    a scorer exception during a screen leaves the LoRA wrappers on the training params
  per_member_soup     with two members the second member's best is its OWN candidate and the soup exists
  train_on_holdout_final --train-on-holdout trains on every image and ships the latest state (EMA first)
  train_on_holdout_soup  --train-on-holdout with --member-slices 2 ships the rank-concat soup of both members' finals
  member_same_init     --same-init: members after the first restart from member 0's init seed, their finals still differ
  plain_delta_soup     --plain-delta: a krea2-style plain constant trains, soups by the mean, ships as .diff; off by default
  member_passes_plan   --member-passes P + --train-on-holdout: K planned from the budget, equal planned steps, no confirm
                      stage or in-member screens, EMA end guards, the soup of all K finals ships
  member_slices_three --member-slices 3: three fresh-seed members, about equal slices, each with its own best, and the
                      soup is the exact rank-concat of all three
  nonfinite_skip      non-finite losses are skipped and counted, the run completes
  unloadable_veto     a candidate the twin refuses is never picked
  selector_rules      1-SE tiebreak prefers the better of two same-step candidates; top() is one per point
  selector_trajectory 1-SE prefers earlier steps only within the best candidate's member; a soup wins on score
  pick_tiebreak_latest  --pick-tiebreak latest takes the LATEST state of the best's own run within one SE, never a
                      significantly worse one; the default stays "earliest"; an unknown value raises
  cadence_cap         with --adaptive-cadence at least three eval points still fit a short horizon
  trailing_ignores_0  the identity point is not counted as an improvement
  screen_passes_schedule / seed2_fits_or_declines   the screen schedule and the --seed2 margin (simulator)
  twin_failure_continues  a twin exception at the parity meter or mid-confirm drops the twin; the run trains
                      and finishes on the hook path instead of ending with only the identity on disk
  requant_matches_twin  ideogram4 --train-base requant builds the evaluator's base by the twin's measured rule: every
                      scaled fp8 Linear rounded once to e4m3 (bit-identical to rebuild_ideogram4_evaluator_base), every
                      other tensor (biases, norms) kept as stored, scale keys dropped

v11 --sigma-head flux-guidance (crown/sigma_head.py), on a flux-structured fake (TinyFluxVec: vec = time_in(emb(sigma)) +
guidance_in(emb(g)) + vector_in(y), over-guided above g = 1):

  sigma_head_exact        the closed-form delivery on flux-sized MLPEmbedders (256 -> 3072, random weights, a rank-16
                          candidate LoRA merged on all four Linears): after the rank<=16 delta is rank-concatenated,
                          time_in'(emb s_b) + guidance_in(emb 3.5) == time_in(emb s_b) + guidance_in(emb g_b) at every band
                          (kept bands unchanged) to ~1e-5 in fp32; also reports the bf16 realisation
  sigma_head_stage_ships  the stage on a candidate: per-band fit, exact delivery (rank-concat), reproduction, fresh-draw
                          adoption; the patched state loads
  sigma_head_new_key      time_in.out_layer excluded from the LoRA targets: the correction ships as a new key, scored on
                          the hook path through the inert wrapper attached at init (left inert afterwards)
  sigma_head_twin_fresh   with a twin, the adoption test runs on the twin at the fresh draws [k, 2k) and reuses the pick's
                          twin confirm for the unpatched side (one twin call)
  sigma_head_rejects      a patched candidate that does not beat the unpatched one on the fresh draws is not shipped (stage
                          and full run)
  sigma_head_unloadable   a patched state the loader refuses is replaced by the unpatched pick in the final check
  sigma_head_budget_skip  a budget that cannot hold the stage skips it before any scoring
  sigma_head_non_flux     any family but flux: a logged no-op, zero reserve, the pick ships unchanged
  sigma_head_run_flux     a full run on the flux fake with the head armed: the stage runs after the pick and the artifact
                          matches its decision
  sigma_head_default_off  the flag absent or "off": no state, zero reserve, no summary key, no extra wrapper, and the
                          conditioning cache keys are the pre-v11 ones

    python tools/mock_checks.py            (exit 1 on any failure)
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("mock_run", os.path.join(HERE, "mock_run.py"))
mock_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_run)          # installs the fake comfy modules

import torch  # noqa: E402

from crown import engine  # noqa: E402
from crown.budget import Budget  # noqa: E402
from crown.selector import Candidate, Selector  # noqa: E402
from crown.scoring import ScoreReport  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc()          # where it failed, not only what


def build(seconds=40, n_images=12, max_members=1, twin=True, **over):
    torch.manual_seed(0)
    Cc, H, W = 4, 8, 8
    prompts = ["", "a red cube", "a blue sphere", "a green cone"]
    cond_vecs = {p: (torch.zeros(1, 8) if p == "" else torch.randn(1, 8)) for p in prompts}
    items = []
    for i in range(n_images):
        cap = prompts[1 + i % 3]
        x = torch.randn(1, Cc, H, W) * 0.5 + 0.3 * cond_vecs[cap].mean()
        items.append({"name": f"{i:03d}.png", "caption": cap, "sha256": f"{i:064x}", "scaled": x, "holdout": i % 4 == 0})
    dm = mock_run.TinyFlow(Cc * H * W)
    model = mock_run.FakeModel(dm)
    guider = mock_run.FakeGuider(model, cond_vecs)
    cfg = types.SimpleNamespace(
        family="mock", seed=0, include=None, exclude=None, rank=4, alpha=4.0, ckpt=False, ckpt_stride=1,
        lr=3e-3, lr_final_frac=0.1, warmup_steps=5, weight_decay=0.0, grad_clip=1.0, lora_plus_ratio=1.0, ema=0.99,
        eval_share=0.25, confirm_top=3, confirm_noises=4, phase2=True, max_members=max_members,
        polish=False, polish_lr_frac=0.3, band_power=0.0, plateau_window=2, screen_ema_only=False,
        empty_prompt_frac=0.5, select_metric="mean", eval_every=0, holdout_names="", flip=False, adaptive_cadence=False)
    for k, v in over.items():
        setattr(cfg, k, v)
    out = tempfile.mkdtemp(prefix="crown-check-")
    budget = Budget(time.time() + seconds, kill_margin_s=0.0, publish_reserve_s=2.0)
    os.environ["GOD_TRAIN_LOGS"] = "0"
    extra = {"model_options": {}, "seed": 42}

    class FakeTwin:
        source = "mock"
        base = model
        veto_steps = set()

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
            if getattr(self, "veto_current", None) is not None and self.veto_current(state):
                return None, ["vetoed-by-check"]
            tr.lora.set_tensors(state, scale)
            rep = tr.scorer.score(guider, extra, lambda p, it: tr._set_cond(guider, p, it), noises=noises)
            tr.lora.restore()
            return rep, []

    ref = []
    tw = FakeTwin(ref) if twin else None
    tr = engine.Trainer(cfg, model, items, {p: p for p in prompts}, out, budget, twin=tw)
    ref.append(tr)
    return tr, guider, extra, out, tw


def run(tr, guider, extra):
    tr.run(guider, extra)
    return json.load(open(os.path.join(tr.out_dir, "summary.json")))


# --- checks ------------------------------------------------------------------------

def identity_first():
    import crown.scoring as scoring
    tr, guider, extra, out, _ = build(seconds=20)
    orig = scoring.HoldoutScorer.score

    def boom(*a, **k):
        raise RuntimeError("first forward exploded")
    scoring.HoldoutScorer.score = boom
    try:
        tr.run(guider, extra)
        raise AssertionError("run should have raised")
    except RuntimeError:
        pass
    finally:
        scoring.HoldoutScorer.score = orig
    assert os.path.exists(os.path.join(out, "last.safetensors")), "no artifact before the first screen"


def restore_on_error():
    from crown.scoring import HoldoutScorer
    tr, guider, extra, out, _ = build(seconds=20)
    tr.scorer = HoldoutScorer(tr.holdout_items, None, tr.device)     # run() would build it; we screen directly
    orig = tr.scorer.score

    def boom(*a, **k):
        raise ValueError("Non-finite denoising MSE")
    tr.scorer.score = boom
    cand = Candidate("x", 1, {n: (torch.randn_like(u), torch.randn_like(d)) for n, (u, d) in tr.lora.state().items()}, tr.lora.scale)
    try:
        tr._screen(guider, extra, cand)
    except ValueError:
        pass
    for i, n in enumerate(tr.lora.names):
        w = tr.lora.wrappers[n]
        assert w.up is tr.lora.params[2 * i] and w.down is tr.lora.params[2 * i + 1], f"wrapper {n} left on the candidate"
    tr.scorer.score = orig


def per_member_soup():
    tr, guider, extra, out, _ = build(seconds=60, max_members=2)
    s = run(tr, guider, extra)
    members = s["members"]
    assert len(members) == 2, f"phase 2 did not run (members={len(members)}); lengthen the horizon"
    assert members[1]["best_member"] == 1, f"member 1's best is member {members[1]['best_member']}'s candidate (the old defect)"
    soups = [c for c in tr.selector.cands if c.tag == "soup"]
    assert soups, "no soup candidate"
    r = tr.lora.rank
    assert next(iter(soups[0].state.values()))[0].shape[1] == 2 * r, "soup is not a rank-concat of two states"


def train_on_holdout_final():
    """v11 --train-on-holdout: every image is trained on (holdout included) and the latest member state ships (EMA
    preferred at that step), whatever the biased holdout screens say; the default split is unchanged when off."""
    tr0, *_ = build(seconds=5)
    n_all = len(tr0.items)
    assert len(tr0.train_items) < n_all and not tr0.train_on_holdout, "default must keep the holdout out of training"
    tr, guider, extra, out, _ = build(seconds=40, train_on_holdout=True)
    assert len(tr.train_items) == n_all, f"train_items {len(tr.train_items)} != all {n_all}"
    s = run(tr, guider, extra)
    assert s["n_train"] == n_all, f"summary n_train {s['n_train']} != {n_all}"
    pool = [c for c in tr.selector.cands if c.member >= 0 and c.step > 0 and not c.unloadable]
    top_step = max(c.step for c in pool)
    assert s["final"]["step"] == top_step, f"shipped step {s['final']['step']} is not the latest {top_step}"
    tags_at_top = {c.tag for c in pool if c.step == top_step}
    if "ema" in tags_at_top:
        assert s["final"]["picked"] == "ema", f"shipped {s['final']['picked']} at the latest step, EMA exists"
    assert "train_on_holdout" in s, "summary does not record the override"


def train_on_holdout_soup():
    """v11 --train-on-holdout with --member-slices 2: the artefact is the exact rank-concat soup of BOTH members' final
    states (no selection), it is screened once as a failure guard, and the summary lists the finals."""
    tr, guider, extra, out, _ = build(seconds=75, max_members=1, member_slices=2, train_on_holdout=True)
    s = run(tr, guider, extra)
    assert len(s["members"]) == 2, f"expected 2 members, got {len(s['members'])}"
    assert s["final"]["picked"] == "soup_final" and s["final"]["member"] == -1, f"shipped {s['final']}"
    rec = s["train_on_holdout"]
    assert rec["shipped"].startswith("m-1:soup_final@") and len(rec["finals"]) == 2, rec
    assert {f.split(":")[0] for f in rec["finals"]} == {"m0", "m1"}, rec["finals"]
    sf = [c for c in tr.selector.cands if c.tag == "soup_final"]
    assert len(sf) == 1 and sf[0].screen is not None, "final soup missing or not screened"
    r = tr.lora.rank
    assert next(iter(sf[0].state.values()))[0].shape[1] == 2 * r, "final soup is not a rank-concat of two states"
    assert os.path.exists(os.path.join(out, "last.safetensors")), "no artifact"


def member_passes_plan():
    """v12 --member-passes P with --train-on-holdout: K is planned from the budget and P, every member trains exactly
    its planned steps (equal lengths), no confirm stage and no in-member screens run, each member's final is
    EMA-guarded, and the artefact is the rank-concat soup of all K finals."""
    tr, guider, extra, out, _ = build(seconds=90, max_members=1, train_on_holdout=True, member_passes=0.5)
    s = run(tr, guider, extra)
    mp = s["member_passes"]
    assert mp["P"] == 0.5 and mp["K"] >= 2, f"planned K {mp}"
    assert len(s["members"]) == mp["K"], f"{len(s['members'])} members ran, K {mp['K']}"
    steps = [m["best_step"] for m in s["members"]]
    plan = mp["plan"]["steps_per_member"]
    assert all(st == plan for st in steps), f"member steps {steps} != planned {plan}"
    assert s.get("confirms_skipped") == "train-on-holdout" and not s["confirms"], "confirm stage ran under train-on-holdout"
    per_member_screens = {}
    for e in s["evals"]:
        if e["step"] == 0:
            continue                          # the run-level identity screen, not an in-member screen
        per_member_screens.setdefault(e["member"], []).append(e["step"])
    for m, st in per_member_screens.items():
        if m < 0:
            continue
        assert len(set(st)) == 1 and st[0] == plan, f"member {m} screened at {sorted(set(st))}, expected only its final {plan}"
        tags = {e["tag"] for e in s["evals"] if e["member"] == m and e["step"] > 0}
        assert tags == {"ema"}, f"member {m} end guard screened {tags}, expected only ema"
    assert s["final"]["picked"] == "soup_final" and s["final"]["member"] == -1, s["final"]
    r = tr.lora.rank
    sf = [c for c in tr.selector.cands if c.tag == "soup_final"][0]
    assert next(iter(sf.state.values()))[0].shape[1] == mp["K"] * r, "final soup rank != K x r"
    assert s["train_on_holdout"]["cut_members"] == [], s["train_on_holdout"]
    # off by default: no plan, the selector's pick ships
    tr2, g2, e2, out2, _ = build(seconds=20)
    s2 = run(tr2, g2, e2)
    assert "member_passes" not in s2 and "confirms_skipped" not in s2, "member-passes leaked into the default path"
    # without --train-on-holdout the flag is ignored (logged), not applied
    tr3, g3, e3, out3, _ = build(seconds=20, member_passes=0.5)
    s3 = run(tr3, g3, e3)
    assert "member_passes" not in s3, "--member-passes applied without --train-on-holdout"


def band_stop():
    """v12 L1 --band-stop: a band leaves the case cycle once the member's progress p reaches its stop (4 values = the
    band groups 0-3 / 4-9 / 10-13 / 14-15); the steps after that go to the open bands only. All stops at 1.0 draw the
    exact band sequence of the default path (same rng calls), and malformed values raise."""
    def drawn(**over):
        tr, guider, extra, out, _ = build(seconds=60, max_members=1, train_on_holdout=True, member_passes=0.5, **over)
        rec, orig = [], tr._cases_at

        def spy(cases, n_items, p):
            out_cases = orig(cases, n_items, p)
            rec.append((p, out_cases[-1][1]))
            return out_cases
        tr._cases_at = spy
        s = run(tr, guider, extra)
        return rec, s
    rec, s = drawn(band_stop="1,1,0.5,0.5")
    late = [b for p, b in rec if p >= 0.5]
    early = [b for p, b in rec if p < 0.5]
    assert late and all(b < 10 for b in late), f"bands >= 10 drawn after their stop: {sorted(set(late))}"
    assert any(b >= 10 for b in early), "bands 10-15 never drawn before their stop"
    assert s.get("band_stop_log"), "summary does not record the band-stop transitions"
    ref, _ = drawn()
    same, _ = drawn(band_stop="1,1,1,1")
    n = min(len(ref), len(same), 300)
    assert n > 50 and [b for _, b in ref[:n]] == [b for _, b in same[:n]], "all-open --band-stop changed the band draws"
    tr, *_ = build(seconds=5, band_stop="1,0.5,0")
    try:
        tr._band_stop()
        raise AssertionError("a malformed --band-stop did not raise")
    except ValueError:
        pass


def member_same_init():
    """v12 S2 --same-init with --member-passes and --train-on-holdout: every member after the first restarts from
    member 0's init seed (cfg.seed) instead of cfg.seed + 1000 k; the members' finals still differ (each member keeps
    its own training draws), and the soup of all K finals ships. Off by default: a fresh init seed per member."""
    def seeds_of(**over):
        tr, guider, extra, out, _ = build(seconds=90, max_members=1, train_on_holdout=True, member_passes=0.5, **over)
        seen, orig = [], tr.lora.reinit

        def spy(seed):
            seen.append(seed)
            return orig(seed)
        tr.lora.reinit = spy
        s = run(tr, guider, extra)
        return tr, s, seen
    tr, s, seen = seeds_of(same_init=True)
    K = s["member_passes"]["K"]
    assert K >= 2 and len(seen) == K - 1, f"K {K}, reinit calls {seen}"
    assert all(x == 0 for x in seen), f"--same-init re-initialised members with seeds {seen}, expected member 0's seed 0"
    finals = {}
    for c in tr.selector.cands:
        if c.member >= 0 and c.tag == "ema":
            finals[c.member] = c.state
    assert len(finals) == K, f"member finals {sorted(finals)} for K {K}"
    a, b = finals[0], finals[1]
    assert any(not torch.equal(a[n][0], b[n][0]) for n in a), "same-init members ended identical: the training draws are shared"
    assert s["final"]["picked"] == "soup_final", s["final"]
    tr2, s2, seen2 = seeds_of()
    K2 = s2["member_passes"]["K"]
    assert seen2 == [1000 * k for k in range(1, K2)], f"default re-init seeds {seen2}, expected a fresh seed per member"


class _PlainMod(torch.nn.Module):
    """krea2-style block modulation: a bare constant added to the activations (a Parameter, no Linear)."""

    def __init__(self, hidden):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(hidden), requires_grad=False)

    def forward(self, h):
        return h + self.lin


class _TinyFlowPlain(mock_run.TinyFlow):
    def __init__(self, dim, cond_dim=8, hidden=64):
        super().__init__(dim, cond_dim, hidden)
        self.mod = _PlainMod(hidden)

    def forward(self, x_t, sigma, cond):
        h = torch.cat([x_t.flatten(1), sigma[:, None], cond.expand(x_t.shape[0], -1)], 1)
        h = self.mod(torch.tanh(self.inp(h)))
        for b in self.blocks:
            h = torch.tanh(b(h)) + h
        return self.out(h).reshape(x_t.shape)


def plain_delta_soup():
    """v12 P --plain-delta on a model with a krea2-style plain modulation constant (mod.lin, not a Linear), under
    --train-on-holdout --member-passes: the delta trains (non-zero), every member starts it at zero, the shipped soup's
    delta is the mean of the members' final deltas, the artefact carries `diffusion_model.mod.lin.diff`, and the
    delta changes the forward. Off by default: no plain state, no .diff key. A regex matching nothing raises."""
    from safetensors.torch import load_file
    from crown import lora as L
    orig = mock_run.TinyFlow
    mock_run.TinyFlow = _TinyFlowPlain
    try:
        tr, guider, extra, out, _ = build(seconds=90, max_members=1, train_on_holdout=True, member_passes=0.5,
                                          plain_delta=r"(^|\.)mod\.lin$")
        s = run(tr, guider, extra)
        assert s["plain_delta"]["tensors"] == 1 and s["plain_delta"]["numel"] == 64, s["plain_delta"]
        K = s["member_passes"]["K"]
        finals = {c.member: c for c in tr.selector.cands if c.member >= 0 and c.tag == "ema"}
        assert len(finals) == K >= 2, f"finals {sorted(finals)}, K {K}"
        vecs = [finals[m].state[L.PLAIN_KEY][0].float() for m in sorted(finals)]
        assert all(v.abs().max() > 0 for v in vecs), "a member's plain delta stayed at zero"
        sf = [c for c in tr.selector.cands if c.tag == "soup_final"][0]
        assert torch.allclose(sf.state[L.PLAIN_KEY][0].float(), sum(vecs) / K, atol=1e-6), "soup delta != mean of the members'"
        sd = L.export_state_dict(sf.state, sf.scale)
        assert "diffusion_model.mod.lin.diff" in sd and tuple(sd["diffusion_model.mod.lin.diff"].shape) == (64,), sorted(sd)[:5]
        art = load_file(os.path.join(tr.out_dir, engine.C.OUTPUT_LORA_NAME))
        assert "diffusion_model.mod.lin.diff" in art and art["diffusion_model.mod.lin.diff"].abs().max() > 0, "artefact lacks the .diff"
        dm = tr.model.model.diffusion_model
        x, sig, cnd = torch.randn(1, 4, 8, 8), torch.tensor([0.5]), torch.randn(1, 8)
        tr.lora.set_tensors(sf.state, sf.scale)
        with torch.no_grad():
            y_on = dm(x, sig, cnd)
            tr.lora.plain.enabled = False
            y_off = dm(x, sig, cnd)
            tr.lora.plain.enabled = True
        tr.lora.restore()
        assert not torch.allclose(y_on, y_off), "the plain delta does not reach the forward"
        assert dm.mod.lin.abs().max() == 0, "the base parameter was modified in place"
        # off by default on the same model
        tr2, g2, e2, out2, _ = build(seconds=40, max_members=1, train_on_holdout=True, member_passes=0.5)
        run(tr2, g2, e2)
        assert tr2.lora.plain is None and all(L.PLAIN_KEY not in c.state for c in tr2.selector.cands if c.state is not None)
        art2 = load_file(os.path.join(tr2.out_dir, engine.C.OUTPUT_LORA_NAME))
        assert not any(k.endswith(".diff") for k in art2), "a .diff key without --plain-delta"
        try:
            build(seconds=10, plain_delta=r"no_such_param$")
            raise AssertionError("a regex matching nothing did not raise")
        except ValueError:
            pass
    finally:
        mock_run.TinyFlow = orig
        L._PLAIN = None


class _GainNorm(torch.nn.Module):
    """ideogram4-style 1-D gain stored as `.weight` (not a Linear)."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden), requires_grad=False)

    def forward(self, h):
        return h * self.weight


class _TinyFlowGainBias(mock_run.TinyFlow):
    def __init__(self, dim, cond_dim=8, hidden=64):
        super().__init__(dim, cond_dim, hidden)
        self.norm = _GainNorm(hidden)

    def forward(self, x_t, sigma, cond):
        h = torch.cat([x_t.flatten(1), sigma[:, None], cond.expand(x_t.shape[0], -1)], 1)
        h = self.norm(torch.tanh(self.inp(h)))
        for b in self.blocks:
            h = torch.tanh(b(h)) + h
        return self.out(h).reshape(x_t.shape)


def plain_delta_gain_bias():
    """v12 P extended (ideogram4 ipD): a 1-D gain stored as `.weight` and a LoRA-targeted Linear's bias both train
    under --plain-delta; the Linear's 2-D weight is never taken even when the regex matches it; the gain ships on the
    module key (`diffusion_model.norm.diff`), the bias on its own name (`diffusion_model.blocks.0.bias.diff`); the
    bias delta reaches the forward as an output offset equal to patching the bias; base tensors stay untouched."""
    from safetensors.torch import load_file
    from crown import lora as L
    orig = mock_run.TinyFlow
    mock_run.TinyFlow = _TinyFlowGainBias
    try:
        tr, guider, extra, out, _ = build(seconds=90, max_members=1, train_on_holdout=True, member_passes=0.5,
                                          plain_delta=r"^(norm[.]weight|blocks[.]0[.](weight|bias))$")
        lay = {n: (off, n_) for n, _, off, n_ in tr.lora.plain.layout}
        assert sorted(lay) == ["blocks.0.bias", "norm.weight"], f"plain layout {sorted(lay)} (blocks.0.weight must stay the LoRA's)"
        s = run(tr, guider, extra)
        sf = [c for c in tr.selector.cands if c.tag == "soup_final"][0]
        v = sf.state[L.PLAIN_KEY][0].float()
        vg = v[lay["norm.weight"][0]:lay["norm.weight"][0] + 64]
        vb = v[lay["blocks.0.bias"][0]:lay["blocks.0.bias"][0] + 64]
        assert vg.abs().max() > 0 and vb.abs().max() > 0, "the gain or the bias delta stayed at zero"
        art = load_file(os.path.join(tr.out_dir, engine.C.OUTPUT_LORA_NAME))
        assert "diffusion_model.norm.diff" in art and "diffusion_model.blocks.0.bias.diff" in art, sorted(k for k in art if k.endswith(".diff"))
        dm = tr.model.model.diffusion_model
        b0 = dm.blocks[0]
        x = torch.randn(3, 64)
        tr.lora.set_tensors({k: v2 for k, v2 in sf.state.items()}, sf.scale)
        tr.lora.set_enabled(False)
        tr.lora.plain.enabled = True
        with torch.no_grad():
            y_hook = b0(x)
            y_ref = torch.nn.functional.linear(x, b0.weight, (b0.bias.float() + vb).to(b0.bias.dtype))
        tr.lora.set_enabled(True)
        tr.lora.restore()
        assert torch.allclose(y_hook, y_ref, atol=1e-6), "the bias output offset != patching the bias"
        assert torch.equal(dm.norm.weight, torch.ones(64)), "the base gain was modified in place"
    finally:
        mock_run.TinyFlow = orig
        L._PLAIN = None


def member_slices_three():
    """v10 --member-slices 3: three members from fresh seeds, each ends at its own slice with its own best, the slices
    are about equal, and the soup is the exact rank-concat of all three bests."""
    tr, guider, extra, out, _ = build(seconds=75, max_members=1, member_slices=3)
    s = run(tr, guider, extra)
    members = s["members"]
    assert len(members) == 3, f"expected 3 members, got {len(members)}"
    for k, m in enumerate(members):
        assert m["best_member"] == k, f"member {k}'s best is member {m['best_member']}'s candidate"
    secs = s["member_seconds"]
    mean = sum(secs) / len(secs)
    assert all(0.5 * mean <= x <= 1.7 * mean for x in secs), f"slices are not about equal: {secs}"
    assert s.get("member_slices") == 3, "summary does not record member_slices"
    soups = [c for c in tr.selector.cands if c.tag == "soup"]
    assert soups, "no soup candidate"
    r = tr.lora.rank
    assert next(iter(soups[0].state.values()))[0].shape[1] == 3 * r, "soup is not a rank-concat of three states"
    assert s["final"]["picked"] in {"raw", "ema", "avg2", "soup"}, f"unexpected pick {s['final']}"


def nonfinite_skip():
    tr, guider, extra, out, _ = build(seconds=25)
    calls = {"n": 0}
    orig = engine.flow_prediction_mse

    def flaky(noisy, den, target, sigma):
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            raise ValueError("Non-finite denoising MSE")
        return orig(noisy, den, target, sigma)
    engine.flow_prediction_mse = flaky
    try:
        s = run(tr, guider, extra)
    finally:
        engine.flow_prediction_mse = orig
    assert s.get("nonfinite", 0) > 0, "non-finite steps were not counted"
    assert s["steps"] > 20, "training did not continue past the non-finite steps"
    assert os.path.exists(os.path.join(out, "last.safetensors"))


def unloadable_veto():
    tr, guider, extra, out, tw = build(seconds=40)
    # veto the first candidate the CONFIRM stage sends to the twin (the parity call is not a confirm)
    state = {"armed": False, "done": False}
    orig_confirm = tr._confirm

    def confirm(g, e, c):
        state["armed"] = not state["done"]
        try:
            return orig_confirm(g, e, c)
        finally:
            state["armed"] = False
    tr._confirm = confirm

    def veto(st):
        if state["armed"] and not state["done"]:
            state["done"] = True
            return True
        return False
    tw.veto_current = veto
    s = run(tr, guider, extra)
    vetoed = s.get("twin_unloadable") or []
    assert vetoed, "the twin veto never fired"
    fin = s["final"]
    assert not any(v["tag"] == fin["picked"] and v["step"] == fin["step"] for v in vetoed), "a vetoed candidate was shipped"
    assert all(not c.unloadable or c is not tr.selector.pick() for c in tr.selector.cands)


def twin_failure_continues():
    """A twin exception at the parity meter (before training) or at the 1st / 2nd twin call of the confirm
    stage must drop the twin and let the run train and finish on the hook path, the twin's confirms
    discarded. Before the guard, the parity failure ended the run with only the identity saved.
    The confirm stage is forced to run (the mock clock alone may leave it no time), so every failure
    point is reached on every run; a point that is not reached fails the check."""
    for where, nth in (("parity", 1), ("confirm", 1), ("confirm", 2)):
        tr, guider, extra, out, tw = build(seconds=40)
        state = {"stage": "parity", "confirm_calls": 0, "fired": False}
        orig_stage = getattr(tr, "_confirm_stage", None)

        def stage(g, e, top, _orig=orig_stage, _tr=tr, _st=state):
            _st["stage"] = "confirm"
            fits = _tr.budget.fits
            _tr.budget.fits = lambda seconds: True
            try:
                return _orig(g, e, top)
            finally:
                _tr.budget.fits = fits
        if orig_stage is not None:       # the pre-guard engine (415b8f3) has no _confirm_stage: the regression
            tr._confirm_stage = stage    # run then fails on the parity failure itself, which is the defect
        orig = tw.score

        def flaky(st, scale, noises=1, _orig=orig, _st=state, _where=where, _nth=nth):
            if _st["stage"] == "confirm":
                _st["confirm_calls"] += 1
            k = 1 if _st["stage"] == "parity" else _st["confirm_calls"]
            if not _st["fired"] and _st["stage"] == _where and k == _nth:
                _st["fired"] = True
                raise RuntimeError(f"twin exploded ({_where} call {_nth})")
            return _orig(st, scale, noises=noises)
        tw.score = flaky
        s = run(tr, guider, extra)
        label = f"twin failure at {where} call {nth}"
        assert state["fired"], f"{label}: the failure point was never reached ({state['confirm_calls']} confirm calls)"
        dropped = s.get("twin_dropped") or []
        assert len(dropped) == 1 and dropped[0]["where"] == where, f"{label}: twin_dropped = {dropped}"
        assert tr.twin is None, f"{label}: the twin is still attached"
        assert s["steps"] > 0 and s["members"], f"{label}: no training happened (steps {s['steps']})"
        assert "final" in s, f"{label}: the run did not reach the final pick"
        confirmed = [c for c in tr.selector.cands if c.confirm is not None]
        assert confirmed, f"{label}: nothing was confirmed on the hook path"
        twin_confirms = sum(1 for c in s["confirms"] if c.get("via") == "twin")
        expected = 0 if where == "parity" else nth - 1
        assert twin_confirms == expected, f"{label}: {twin_confirms} twin confirms recorded, expected {expected}"
        assert os.path.exists(os.path.join(out, "last.safetensors")), f"{label}: no artifact"


def _rep(per_case_scores, per_image=(1.0,)):
    per_case = {("img", "text", b, 0): v for b, v in enumerate(per_case_scores)}
    return ScoreReport(float(sum(per_case_scores) / len(per_case_scores)), list(per_image), [1.0], [1.0], per_case)


def selector_rules():
    sel = Selector()
    a = Candidate("raw", 100, {}, 1.0, screen=_rep([1.0, 1.0, 1.0, 1.0]), confirm=_rep([0.90, 0.91, 0.92, 0.93]))
    b = Candidate("ema", 100, {}, 1.0, screen=_rep([1.0, 1.0, 1.0, 1.0]), confirm=_rep([0.89, 0.90, 0.91, 0.92]))
    c = Candidate("raw", 200, {}, 1.0, screen=_rep([0.8, 0.8, 0.8, 0.8]), confirm=_rep([0.95, 0.96, 0.97, 0.98]))
    for x in (a, b, c):
        sel.add(x)
    p = sel.pick(use_confirm=True)
    assert p is b, f"tiebreak picked {p.tag}@{p.step}, expected ema@100 (better of the same step)"
    top = sel.top(3)
    assert len(top) == 2 and {(t.member, t.step) for t in top} == {(0, 100), (0, 200)}, "top() did not dedupe per point"
    c.unloadable = True
    assert c not in sel.top(3) and sel.best() is not c, "unloadable candidate still ranked"


def selector_trajectory():
    """The 1-SE 'earliest' preference orders the checkpoints of ONE member's run: a soup (member -1, step =
    the members' total) or another member (its own step count) wins only by score (P1 z-image, 2026-09-26)."""
    best = _rep([0.899, 0.900, 0.901, 0.902])
    near = _rep([0.899, 0.902, 0.900, 0.903])          # paired diffs 0, +.002, -.001, +.001: mean .0005 <= SE .00065
    assert near.paired_diff(best)[0] <= near.paired_diff(best)[1], "test data: `near` must lie within one SE of `best`"
    sel = Selector()
    soup = Candidate("soup", 3271, {}, 1.0, screen=best, confirm=best, member=-1)
    m1 = Candidate("raw", 1587, {}, 1.0, screen=near, confirm=near, member=1)
    for x in (soup, m1):
        sel.add(x)
    p = sel.pick(use_confirm=True)
    assert p is soup, f"picked {p.tag}@{p.step} (member {p.member}) over the better soup"
    sel = Selector()
    late = Candidate("ema", 1684, {}, 1.0, screen=best, confirm=best, member=0)
    early = Candidate("ema", 1347, {}, 1.0, screen=near, confirm=near, member=0)
    other = Candidate("raw", 900, {}, 1.0, screen=near, confirm=near, member=1)
    for x in (late, early, other):
        sel.add(x)
    p = sel.pick(use_confirm=True)
    assert p is early, f"picked {p.tag}@{p.step} (member {p.member}); expected the earlier point of the best's own run"


def pick_tiebreak_latest():
    import crown.selector as sel_mod
    best = _rep([0.899, 0.900, 0.901, 0.902])
    near = _rep([0.899, 0.902, 0.900, 0.903])          # within one SE of best (see selector_trajectory)
    worse = _rep([0.950, 0.951, 0.952, 0.953])         # significantly worse
    assert near.paired_diff(best)[0] <= near.paired_diff(best)[1], "test data: `near` must lie within one SE of `best`"

    def fresh():
        sel = Selector()
        cands = {"early": Candidate("ema", 1000, {}, 1.0, screen=near, confirm=near, member=0),
                 "best": Candidate("ema", 1347, {}, 1.0, screen=best, confirm=best, member=0),
                 "late": Candidate("ema", 1684, {}, 1.0, screen=near, confirm=near, member=0),
                 "later_worse": Candidate("raw", 2000, {}, 1.0, screen=worse, confirm=worse, member=0),
                 "other_member": Candidate("raw", 3000, {}, 1.0, screen=near, confirm=near, member=1)}
        for c in cands.values():
            sel.add(c)
        return sel, cands

    saved = sel_mod.PICK_TIEBREAK
    try:
        assert saved == "earliest", f"module default is {saved!r}, want 'earliest' (the shipped rule)"
        sel, c = fresh()
        assert sel.pick(use_confirm=True) is c["early"], "default rule did not take the earliest state within one SE"
        sel_mod.PICK_TIEBREAK = "latest"
        sel, c = fresh()
        p = sel.pick(use_confirm=True)
        assert p is c["late"], f"latest rule picked {p.tag}@{p.step} (member {p.member}); want ema@1684 of member 0"
        sel_mod.PICK_TIEBREAK = "sideways"
        sel, _ = fresh()
        try:
            sel.pick(use_confirm=True)
            raise AssertionError("an unknown tie-break value did not raise")
        except ValueError:
            pass
    finally:
        sel_mod.PICK_TIEBREAK = saved


def cadence_cap():
    tr, guider, extra, out, _ = build(seconds=25, adaptive_cadence=True)
    s = run(tr, guider, extra)
    steps = {e.get("step") for e in s["evals"] if e.get("step", 0) > 0}
    assert len(steps) >= 3, f"only {len(steps)} eval points with adaptive cadence on a short horizon"


def trailing_ignores_0():
    sel = Selector()
    sel.add(Candidate("identity", 0, {}, 1.0, screen=_rep([1.0, 1.0, 1.0, 1.0])))
    sel.add(Candidate("ema", 100, {}, 1.0, screen=_rep([0.5, 0.5, 0.5, 0.5])))
    assert sel.trailing_improvements(0) == 0, "identity->first point counted as an improvement"
    sel.add(Candidate("ema", 200, {}, 1.0, screen=_rep([0.4, 0.4, 0.4, 0.4])))
    assert sel.trailing_improvements(0) == 1


def _sim():
    spec2 = importlib.util.spec_from_file_location("sim_budget", os.path.join(HERE, "sim_budget.py"))
    m = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(m)
    return m


def screen_passes_schedule():
    """--screen-passes: screens land on pass multiples, never closer than the share cap allows."""
    tr, guider, extra, out, _ = build(seconds=20, screen_passes="0.5,1,1.5,2,3", screen_max_share=0.3)
    ps = tr._pass_steps()
    tr.budget.observe_step(1.0)
    tr.budget.observe_eval(1.0)                       # cheap screens: the schedule itself governs
    s1 = tr._next_screen_step(0)
    assert s1 == round(0.5 * ps), f"first screen {s1} != 0.5 pass ({ps} steps/pass)"
    s2 = tr._next_screen_step(s1)
    assert s2 == round(1.0 * ps), f"second screen {s2} != 1 pass"
    tr.budget.eval_time = 1000.0                      # very expensive screens: the share cap must widen the gap
    s3 = tr._next_screen_step(s2)
    gap_min = 1000.0 * tr._screens_per_point() * 0.7 / (0.3 * 1.0)
    assert s3 - s2 >= gap_min - 1, f"gap {s3 - s2} below the share cap {gap_min:.0f}"


def seed2_fits_or_declines():
    """--seed2 (virtual clock, measured profiles) over independent simulator draws: it never starts a second
    seed that cannot finish, never breaks the 2-minute margin, and opens on boss z-image where the arithmetic
    allows it in at least one draw. (Whether it opens in a given draw depends on when the noisy plateau is
    detected: 5 of 10 draws on boss z-image, 2026-09-26; a single-draw assertion was a coin flip.)"""
    sim = _sim()
    pol = {"name": "t", "screen_passes": sim.SCHED, "seed2": True}
    opened = 0
    for fam, h, ntr, nho, seeds in (("z-image", 1.0, 32, 4, range(5)), ("krea2", 0.75, 20, 3, range(3))):
        for seed in seeds:
            r = sim.Sim(fam, h, ntr, nho, pol, seed=seed).run()
            s2 = r["seed2"]
            assert not (s2 and s2.get("skipped")), f"{fam} draw {seed}: seed2 started but did not finish: {s2}"
            assert r["slack_min"] >= 2.0, f"{fam} draw {seed}: ends {r['slack_min']} min before the deadline (margin broken)"
            opened += 1 if (s2 and fam == "z-image") else 0
    assert opened >= 1, "seed2 never opened on boss z-image in 5 draws although the arithmetic allows it"


def requant_matches_twin():
    """ideogram4 --train-base requant == the twin's evaluator-base rule (docs/v9_gpu_batch.md change 17)."""
    from crown import twin
    from crown.assets import evaluator_grid_ideogram4

    g = torch.Generator().manual_seed(0)
    raw = {}
    for i in range(3):
        w = torch.randn(64, 32, generator=g) * 0.05
        s = w.abs().amax(dim=1) / 448.0
        raw[f"b{i}.lin.weight"] = (w / s[:, None]).to(torch.float8_e4m3fn)
        raw[f"b{i}.lin.weight_scale"] = s.float()
        raw[f"b{i}.lin.bias"] = (torch.randn(64, generator=g) * 0.3 + 1.0).to(torch.bfloat16)
        raw[f"b{i}.norm.weight"] = (torch.rand(32, generator=g) * 0.4 + 0.8).to(torch.bfloat16)
    raw["plain.weight"] = (torch.randn(8, 8, generator=g) * 2).to(torch.float8_e4m3fn)   # fp8 without a scale
    out = evaluator_grid_ideogram4(raw)
    tw, _ = twin.rebuild_ideogram4_evaluator_base(raw, torch.device("cpu"))
    assert not any(k.endswith("weight_scale") for k in out), "scale keys survived"
    for k, v in out.items():
        assert v.dtype == torch.bfloat16, f"{k}: {v.dtype}, want bf16"
        if k.endswith("lin.weight"):
            assert torch.equal(v.float(), tw[k].float()), f"{k}: differs from the twin's evaluator base"
        elif k == "plain.weight":
            assert torch.equal(v.float(), raw[k].float()), "an unscaled fp8 tensor changed"
        else:
            assert torch.equal(v, raw[k]), f"{k}: a bf16 tensor was changed (the evaluator keeps it as stored)"


# --- v11 sigma head ------------------------------------------------------------------

from crown import sigma_head as SH  # noqa: E402


class FakeMLP(torch.nn.Module):
    """Comfy's MLPEmbedder: in_layer -> SiLU -> out_layer."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.in_layer = mock_run.HookLinear(in_dim, hidden)
        self.silu = torch.nn.SiLU()
        self.out_layer = mock_run.HookLinear(hidden, hidden)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


class TinyFluxVec(torch.nn.Module):
    """A velocity model with flux's modulation structure: vec = time_in(emb(sigma)) + guidance_in(emb(g)) + vector_in(y),
    read by a modulated hidden layer and a rank-1 readout along c. guidance_in is solved so that guidance_in(emb(g)) =
    (g - 1) c at g in {1, 1.5, ..., 3.5}: guidance above 1 adds a bias that grows with g (the probe's over-guidance)."""

    def __init__(self, dim, cond_dim=8, hidden=32, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.params = types.SimpleNamespace(guidance_embed=True)
        self.time_in = FakeMLP(SH.EMB_DIM, hidden)
        self.guidance_in = FakeMLP(SH.EMB_DIM, hidden)
        self.vector_in = FakeMLP(cond_dim, hidden)
        self.inp = mock_run.HookLinear(dim, hidden)
        self.mod = mock_run.HookLinear(hidden, 2 * hidden)
        self.out = mock_run.HookLinear(hidden, dim)
        self.read = mock_run.HookLinear(hidden, dim)
        with torch.no_grad():
            gs = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
            hg = torch.nn.functional.silu(self.guidance_in.in_layer(SH.timestep_embedding_ref(gs, SH.EMB_DIM)))
            c = torch.randn(hidden, generator=gen)
            x = torch.cat([hg, torch.ones(len(gs), 1)], 1).double()
            sol = torch.linalg.lstsq(x, ((gs - 1)[:, None] * c[None]).double()).solution
            self.guidance_in.out_layer.weight.copy_(sol[:-1].T.float())
            self.guidance_in.out_layer.bias.copy_(sol[-1].float())
            a = torch.randn(dim, generator=gen)
            self.read.weight.copy_(a[:, None] * c[None] / c.square().sum())
            self.read.bias.zero_()
            self.mod.weight.mul_(0.1)
            self.mod.bias.zero_()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_t, sigma, cond, guidance):
        emb = SH.timestep_embedding_ref
        vec = (self.time_in(emb(sigma.float(), SH.EMB_DIM)) + self.guidance_in(emb(guidance.float(), SH.EMB_DIM))
               + self.vector_in(cond.expand(x_t.shape[0], -1)))
        h = self.inp(x_t.flatten(1))
        scale, shift = self.mod(torch.nn.functional.silu(vec)).chunk(2, -1)
        h = torch.tanh(h * (1 + scale) + shift)
        return (self.out(h) + self.read(vec)).reshape(x_t.shape)


class FakeFluxGuider:
    """CFGGuider stand-in reading Comfy-style conds [[cond, {"guidance": g, ...}]]."""

    def __init__(self, model):
        self.model, self.conds, self.inner_model = model, None, object()

    def __call__(self, noisy, t, **extra):
        cond, opts = self.conds["positive"][0]
        g = torch.full((noisy.shape[0],), float(opts["guidance"]))
        v = self.model.model.diffusion_model(noisy, t, cond, g)
        return noisy - t[:, None, None, None] * v


def build_flux(seconds=40, n_images=12, twin=False, sigma_head="flux-guidance", **over):
    torch.manual_seed(0)
    Cc, H, W = 4, 8, 8
    prompts = ["", "a red cube", "a blue sphere", "a green cone"]
    cond_vecs = {p: (torch.zeros(1, 8) if p == "" else torch.randn(1, 8)) for p in prompts}
    items = []
    for i in range(n_images):
        cap = prompts[1 + i % 3]
        x = torch.randn(1, Cc, H, W) * 0.5 + 0.3 * cond_vecs[cap].mean()
        items.append({"name": f"{i:03d}.png", "caption": cap, "sha256": f"{i:064x}", "scaled": x, "holdout": i % 4 == 0})
    dm = TinyFluxVec(Cc * H * W)
    model = mock_run.FakeModel(dm)
    guider = FakeFluxGuider(model)
    cfg = types.SimpleNamespace(
        family="flux", seed=0, include=None, exclude=None, rank=4, alpha=4.0, ckpt=False, ckpt_stride=1,
        lr=3e-3, lr_final_frac=0.1, warmup_steps=5, weight_decay=0.0, grad_clip=1.0, lora_plus_ratio=1.0, ema=0.99,
        eval_share=0.25, confirm_top=3, confirm_noises=4, phase2=True, max_members=1,
        polish=False, polish_lr_frac=0.3, band_power=0.0, plateau_window=2, screen_ema_only=False,
        empty_prompt_frac=0.5, select_metric="mean", eval_every=0, holdout_names="", flip=False, adaptive_cadence=False,
        sigma_head_grid="1.5,2,2.5,3", sigma_head_noises=2)
    if sigma_head is not None:
        cfg.sigma_head = sigma_head
    for k, v in over.items():
        setattr(cfg, k, v)
    out = tempfile.mkdtemp(prefix="crown-sh-")
    budget = Budget(time.time() + seconds, kill_margin_s=0.0, publish_reserve_s=2.0)
    os.environ["GOD_TRAIN_LOGS"] = "0"
    extra = {"model_options": {}, "seed": 42}
    raw_conds = {p: [[cond_vecs[p], {"guidance": 3.5}]] for p in prompts}

    class FakeFluxTwin:
        """Evaluator-path stand-in: the fake loader's key check, then the hook numerics at the requested draws."""
        source = "mock"

        def __init__(self, tr_ref):
            self.tr_ref, self.base, self.calls = tr_ref, model, []

        def _load(self):
            pass

        def release(self):
            pass

        def score(self, state, scale, noises=1, noise_offset=0):
            from crown.twin import apply_lora_like_evaluator
            _, missing = apply_lora_like_evaluator(model, state, scale)
            if missing:
                return None, missing
            self.calls.append((noises, noise_offset))
            return self.tr_ref[0]._sh_score(guider, extra, state, scale, noises, offset=noise_offset), []

    ref = []
    tw = FakeFluxTwin(ref) if twin else None
    tr = engine.Trainer(cfg, model, items, raw_conds, out, budget, twin=tw)
    ref.append(tr)
    return tr, guider, extra, out, tw


def _flux_ready(tr, guider, extra, seed=1, up_std=0.01):
    """Trainer state after run() has set the scorer, plus a non-trivial candidate (small random factors) with its
    screen, whose measured time feeds the budget as run() would (the 60 s prior before any screen is for GPUs)."""
    from crown.scoring import HoldoutScorer
    for it in tr.items:
        it["scaled"] = it["scaled"].to(tr.device)
    tr.scorer = HoldoutScorer(tr.holdout_items, None, tr.device)
    g = torch.Generator().manual_seed(seed)
    st = {n: (torch.randn(u.shape, generator=g) * up_std, d.detach().cpu().clone()) for n, (u, d) in tr.lora.state().items()}
    cand = Candidate("ema", 100, st, tr.lora.scale)
    tr._screen(guider, extra, cand)          # observes the screen time
    return cand


def _rank(state, name):
    return int(state[name][1].shape[0])


def sigma_head_exact():
    from crown.lora import LoraWrapper
    torch.manual_seed(0)
    hidden, rank, scale, g0 = 3072, 16, 1.0, 3.5
    ti, gi = FakeMLP(SH.EMB_DIM, hidden), FakeMLP(SH.EMB_DIM, hidden)
    gen = torch.Generator().manual_seed(1)
    cand = {}
    for name, lin in (("time_in.in_layer", ti.in_layer), ("time_in.out_layer", ti.out_layer),
                      ("guidance_in.in_layer", gi.in_layer), ("guidance_in.out_layer", gi.out_layer)):
        o, i = lin.weight.shape
        cand[name] = (torch.randn(o, rank, generator=gen) * 0.01, torch.randn(rank, i, generator=gen) / i ** 0.5)
        w = LoraWrapper(scale)
        w.up, w.down = cand[name]
        lin.weight_function = [w]
    chosen = [g0] * 3 + [2.0] * 7 + [1.5] * 6
    sigmas = [(b + 0.5) / 16 for b in range(16)]
    emb = SH.timestep_embedding_ref
    e_sig = SH.sigma_embeddings(emb, sigmas, torch.float32, "cpu")
    gvals = sorted(set(chosen))
    row = {g: i for i, g in enumerate(gvals)}
    e_g = SH.guidance_embeddings(emb, gvals, torch.float32, "cpu")
    tp, gp = SH.mlp_f64(ti, "cpu"), SH.mlp_f64(gi, "cpu")
    Hm = SH.mlp_hidden(tp, e_sig)
    G = SH.mlp_out(gp, e_g)
    D = torch.stack([G[row[g]] - G[row[g0]] for g in chosen])
    up, down, diag = SH.closed_form(Hm, D, scale)
    assert up is not None and diag["ok"], f"closed form refused: {diag}"
    assert diag["rank"] <= 16, f"rank {diag['rank']} > 16"
    u0, d0 = cand["time_in.out_layer"]
    ti.out_layer.weight_function[0].up = torch.cat([u0, up], 1)            # rank-concat, as the export stores it
    ti.out_layer.weight_function[0].down = torch.cat([d0, down], 0)
    with torch.no_grad():
        patched = ti(e_sig) + gi(e_g[row[g0]].unsqueeze(0))
        ti.out_layer.weight_function[0].up, ti.out_layer.weight_function[0].down = u0, d0
        target = ti(e_sig) + gi(e_g[[row[g] for g in chosen]])
        plain = ti(e_sig) + gi(e_g[row[g0]].unsqueeze(0))
    err = (patched - target).abs().max(dim=1).values
    shift = (target - plain).norm(dim=1)
    rel = [(patched - target)[b].norm().item() / shift[b].item() for b in range(3, 16)]
    print(f"    exact: kappa(H) {diag['kappa']:.3g}, rank {diag['rank']}, fp64 residual {diag['residual_rel']:.2e} "
          f"(bound {diag['tol']:.2e}), fp32 max|vec error| {err.max().item():.2e} (moved bands; kept bands {err[:3].max().item():.2e}), "
          f"max error/shift {max(rel):.2e}, |vec| max {target.abs().max().item():.2f}")
    assert err.max().item() <= 5e-5, f"fp32 vec error {err.max().item():.2e} > 5e-5"
    assert max(rel) <= 1e-3, f"error/shift {max(rel):.2e}"
    # the evaluator's realisation: bf16 weights, fp16 merge cast to bf16 (LoraWrapper), bf16 compute
    for m in (ti, gi):
        m.to(torch.bfloat16)
    eb, gb = e_sig.to(torch.bfloat16), e_g.to(torch.bfloat16)
    with torch.no_grad():
        ti.out_layer.weight_function[0].up = torch.cat([u0, up], 1)
        ti.out_layer.weight_function[0].down = torch.cat([d0, down], 0)
        pb = (ti(eb) + gi(gb[row[g0]].unsqueeze(0))).double()
        ti.out_layer.weight_function[0].up, ti.out_layer.weight_function[0].down = u0, d0
        tb = (ti(eb) + gi(gb[[row[g] for g in chosen]])).double()
        plb = (ti(eb) + gi(gb[row[g0]].unsqueeze(0))).double()
    relb = [((pb - tb)[b].norm() / (tb - plb)[b].norm()).item() for b in range(3, 16)]
    print(f"    bf16 realisation: error/shift max {max(relb):.3f} median {sorted(relb)[len(relb) // 2]:.3f} (1.0 = no better than no shift)")
    assert max(relb) < 1.0, "bf16 realisation no better than no shift"


def sigma_head_stage_ships():
    tr, guider, extra, out, _ = build_flux(seconds=120)
    cand = _flux_ready(tr, guider, extra)
    got = tr._sigma_head(guider, extra, cand)
    rec = tr.summary["sigma_head"]
    assert got is not None, f"not shipped: {rec.get('skipped')}"
    assert rec.get("adopted") and rec["reproduction"]["ok"] and rec["delivery"]["ok"], rec
    moved = [g for g in rec["g_per_band"] if g != 3.5]
    assert len(moved) >= 8, f"only {len(moved)} bands moved on an over-guided model: {rec['g_per_band']}"
    r_sh = rec["delivery"]["rank"]
    assert 1 <= r_sh <= 16 and _rank(got.state, SH.TARGET) == tr.lora.rank + r_sh, "the correction is not rank-concatenated"
    assert rec["delivery"]["vec"]["max_err_over_shift"] < 1e-3, rec["delivery"]["vec"]
    assert rec["fresh"]["mean"] < -rec["fresh"]["se"], rec["fresh"]
    for n in tr.lora.names:
        if n != SH.TARGET:
            assert got.state[n][0] is cand.state[n][0], f"{n} changed"
    from crown.twin import verify_loadable
    ok, missing = verify_loadable(tr.model, got.state, got.scale)
    assert ok, missing
    assert tr._guidance is None, "guidance override left set"
    for i, n in enumerate(tr.lora.names):
        assert tr.lora.wrappers[n].up is tr.lora.params[2 * i], f"wrapper {n} left on a scored state"
    print(f"    ships: bands {rec['g_per_band']}, fit {rec['fit_gain']['pct']:+.2f}%, fresh {rec['fresh']['pct']:+.2f}% "
          f"(SE {rec['fresh']['se'] / rec['fresh']['unpatched'] * 100:.3f}%), kappa {rec['delivery']['kappa']:.3g}, "
          f"residual {rec['delivery']['residual_rel']:.1e}, rank {r_sh}")


def sigma_head_new_key():
    tr, guider, extra, out, _ = build_flux(seconds=120, exclude="time_in[.]out_layer")
    assert SH.TARGET not in tr.lora.names and SH.TARGET in tr._sh.extra, "the inert wrapper was not attached"
    w = tr._sh.extra[SH.TARGET]
    assert w in tr.model.model.diffusion_model.time_in.out_layer.weight_function
    cand = _flux_ready(tr, guider, extra)
    got = tr._sigma_head(guider, extra, cand)
    rec = tr.summary["sigma_head"]
    assert got is not None, f"not shipped: {rec.get('skipped')}"
    assert rec["delivery"].get("new_key") and SH.TARGET in got.state and SH.TARGET not in cand.state
    assert rec["reproduction"]["ok"], rec["reproduction"]
    assert w.up is None and w.down is None, "the extra wrapper was left routed"
    from crown.twin import verify_loadable
    ok, missing = verify_loadable(tr.model, got.state, got.scale)
    assert ok, missing


def sigma_head_twin_fresh():
    tr, guider, extra, out, tw = build_flux(seconds=120, twin=True)
    cand = _flux_ready(tr, guider, extra)
    cand.confirm, _ = tw.score(cand.state, cand.scale, noises=4)        # the pick's twin confirm (draws 0..3)
    tw.calls.clear()
    got = tr._sigma_head(guider, extra, cand)
    rec = tr.summary["sigma_head"]
    assert got is not None, f"not shipped: {rec.get('skipped')}"
    assert rec["verify_via"] == "twin" and rec["budget"]["reused"]["unpatched_fresh"], rec["budget"]
    assert not rec["budget"]["reused"]["base_fit"], "a twin confirm must not stand in for the hook fit"
    assert tw.calls == [(2, 2)], f"twin calls {tw.calls}, want one patched score at draws [2, 4)"


def sigma_head_rejects():
    tr, guider, extra, out, _ = build_flux(seconds=120)
    cand = _flux_ready(tr, guider, extra)
    orig = tr._sh_fresh

    def worse(g, e, c, st, unp):
        p, u = orig(g, e, c, st, unp)
        p.per_case = {k: u.per_case[k] + 1e-3 for k in u.per_case}   # patched worse than unpatched on every fresh case
        return p, u
    tr._sh_fresh = worse
    got = tr._sigma_head(guider, extra, cand)
    rec = tr.summary["sigma_head"]
    assert got is None and not rec.get("adopted") and rec["fresh"]["adopt"] is False, rec.get("fresh")
    assert "fresh" in rec["skipped"], rec["skipped"]
    assert tr._sh_candidate is not None, "the built patched state is not kept for the paired readout"
    # the same through a full run: the unpatched pick ships
    tr, guider, extra, out, _ = build_flux(seconds=100)
    orig2 = tr._sh_fresh

    def worse2(g, e, c, st, unp):
        p, u = orig2(g, e, c, st, unp)
        p.per_case = {k: u.per_case[k] + 1e-3 for k in u.per_case}
        return p, u
    tr._sh_fresh = worse2
    s = run(tr, guider, extra)
    assert not s["final"]["picked"].endswith("+sh") and not s["sigma_head"]["shipped"], (s["final"], s["sigma_head"].get("skipped"))
    from safetensors.torch import load_file
    sd = load_file(os.path.join(out, "last.safetensors"))
    assert sd[f"diffusion_model.{SH.TARGET}.lora_down.weight"].shape[0] == tr.lora.rank, "a patched state was shipped"


def sigma_head_unloadable():
    tr, guider, extra, out, _ = build_flux(seconds=40)

    def fake_head(g, e, picked):
        up, down = picked.state[SH.TARGET]
        st = dict(picked.state)
        st[SH.TARGET] = (torch.cat([up, torch.zeros(up.shape[0], 2)], 1), torch.cat([down, torch.zeros(2, down.shape[1])], 0))
        tr.summary["sigma_head"] = {"mode": "flux-guidance", "shipped": False, "adopted": True}
        return Candidate(picked.tag + "+sh", picked.step, st, picked.scale, screen=picked.screen, member=picked.member)
    tr._sigma_head = fake_head
    orig = tr._final_loadable

    def loadable(state, scale):
        if _rank(state, SH.TARGET) > tr.lora.rank:
            return False, ["lora key not loaded: refused by the check"]
        return orig(state, scale)
    tr._final_loadable = loadable
    s = run(tr, guider, extra)
    assert not s["final"]["picked"].endswith("+sh"), s["final"]
    assert s["sigma_head"]["loadable"]["ok"] is False and s["sigma_head"]["shipped"] is False, s["sigma_head"]
    assert s["final_loadable"]["ok"], s["final_loadable"]


def sigma_head_budget_skip():
    tr, guider, extra, out, _ = build_flux(seconds=120)
    cand = _flux_ready(tr, guider, extra)
    calls = {"n": 0}
    orig = tr.scorer.score

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    tr.scorer.score = counting
    tr.budget.deadline = time.time() + 3.0          # a few seconds left: the stage cannot fit
    got = tr._sigma_head(guider, extra, cand)
    rec = tr.summary["sigma_head"]
    assert got is None and rec["skipped"].startswith("budget"), rec.get("skipped")
    assert calls["n"] == 0, f"{calls['n']} scores ran although the stage did not fit"


def sigma_head_non_flux():
    tr, guider, extra, out, _ = build(seconds=20, sigma_head="flux-guidance")
    assert tr._sh is not None and not tr._sh.active and "not flux" in tr._sh.reason, tr._sh.reason
    assert tr._sigma_head_reserve() == 0.0
    s = run(tr, guider, extra)
    assert "not flux" in s["sigma_head"]["skipped"] and not s["sigma_head"]["shipped"], s["sigma_head"]
    assert not s["final"]["picked"].endswith("+sh")


def sigma_head_run_flux():
    tr, guider, extra, out, _ = build_flux(seconds=100)
    assert tr._sigma_head_reserve() > 0.0
    s = run(tr, guider, extra)
    rec = s["sigma_head"]
    assert "bands" in rec, f"the stage did not run: {rec.get('skipped')}"
    from safetensors.torch import load_file
    sd = load_file(os.path.join(out, "last.safetensors"))
    r = sd[f"diffusion_model.{SH.TARGET}.lora_down.weight"].shape[0]
    if rec.get("shipped"):
        assert s["final"]["picked"].endswith("+sh") and r > tr.lora.rank, (s["final"], r)
    else:
        assert not s["final"]["picked"].endswith("+sh") and r == tr.lora.rank, (s["final"], r)
    assert s["final_loadable"]["ok"]
    print(f"    full run: shipped={rec.get('shipped')} ({rec.get('skipped', 'adopted')}), steps {s['steps']}, "
          f"stage {rec.get('seconds')}s, fresh {rec.get('fresh', {}).get('pct')}")


def sigma_head_default_off():
    for kw in ({}, {"sigma_head": "off"}):
        tr, guider, extra, out, _ = build(seconds=15, **kw)
        assert tr._sh is None and tr._sigma_head_reserve() == 0.0 and tr._guidance is None
        s = run(tr, guider, extra)
        assert "sigma_head" not in s, s.get("sigma_head")
        assert all(len(k) == 2 for k in tr._processed), f"conditioning cache keys changed: {list(tr._processed)[:3]}"
    tr, guider, extra, out, _ = build_flux(seconds=15, sigma_head=None)
    assert tr._sh is None and tr._sigma_head_reserve() == 0.0
    wf = tr.model.model.diffusion_model.time_in.out_layer.weight_function
    assert len(wf) == 1 and wf[0] is tr.lora.wrappers[SH.TARGET], "an extra wrapper was attached with the head off"


if __name__ == "__main__":
    for name, fn in [("identity_first", identity_first), ("restore_on_error", restore_on_error), ("per_member_soup", per_member_soup),
                     ("member_slices_three", member_slices_three), ("train_on_holdout_final", train_on_holdout_final), ("train_on_holdout_soup", train_on_holdout_soup), ("member_passes_plan", member_passes_plan), ("member_same_init", member_same_init), ("band_stop", band_stop), ("plain_delta_soup", plain_delta_soup), ("plain_delta_gain_bias", plain_delta_gain_bias),
                     ("nonfinite_skip", nonfinite_skip), ("unloadable_veto", unloadable_veto), ("selector_rules", selector_rules), ("selector_trajectory", selector_trajectory),
                     ("pick_tiebreak_latest", pick_tiebreak_latest), ("cadence_cap", cadence_cap), ("trailing_ignores_0", trailing_ignores_0),
                     ("screen_passes_schedule", screen_passes_schedule), ("seed2_fits_or_declines", seed2_fits_or_declines),
                     ("twin_failure_continues", twin_failure_continues), ("requant_matches_twin", requant_matches_twin),
                     ("sigma_head_exact", sigma_head_exact), ("sigma_head_stage_ships", sigma_head_stage_ships),
                     ("sigma_head_new_key", sigma_head_new_key), ("sigma_head_twin_fresh", sigma_head_twin_fresh),
                     ("sigma_head_rejects", sigma_head_rejects), ("sigma_head_unloadable", sigma_head_unloadable),
                     ("sigma_head_budget_skip", sigma_head_budget_skip), ("sigma_head_non_flux", sigma_head_non_flux),
                     ("sigma_head_run_flux", sigma_head_run_flux), ("sigma_head_default_off", sigma_head_default_off)]:
        if len(sys.argv) > 1 and not any(a in name for a in sys.argv[1:]):
            continue                      # optional name filter: python tools/mock_checks.py sigma_head
        check(name, fn)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    sys.exit(1 if bad else 0)
