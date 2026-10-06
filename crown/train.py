"""Orchestration: arguments -> assets -> latents -> conditioning -> model -> train.

Invoked by scripts/image_trainer.py inside the validator's container; can also
be run directly for local experiments.
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["flux", "z-image", "qwen-image", "ideogram4", "krea2"])
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset", required=True, help="zip or directory of images + .txt captions")
    p.add_argument("--out", required=True)
    p.add_argument("--deadline-ts", type=float, required=True, help="unix time at which the validator kills the container")
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
    p.add_argument("--baked-dir", default=os.environ.get("CROWN_ASSETS", "/opt/crown/assets"))
    # LoRA
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="default = rank (scale 1.0)")
    p.add_argument("--include", default=None, help="regex over Linear module names; default = every Linear")
    p.add_argument("--exclude", default=None)
    # optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-final-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-plus-ratio", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--ckpt", dest="ckpt", action="store_true", default=True)
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    p.add_argument("--ckpt-stride", type=int, default=1)
    p.add_argument("--exact-merge", action="store_true",
                   help="plain-fp8 bases: train through the evaluator's post-merge stochastic fp8 rounding (seeded by the "
                        "state-dict key), straight-through gradient; a no-op on bf16/fp16 bases")
    p.add_argument("--fast-lora", action="store_true",
                   help="train through the merged-forward / rank-r-gradient Function (crown.lora._MergedLoraLinear): "
                        "same forward values, no full weight-gradient GEMM, no merged weights kept for backward")
    p.add_argument("--fast-lora-lowrank", action="store_true",
                   help="v12 Q (with --fast-lora): the backward's input gradient uses dy (W + scale up down) with the base "
                        "weight cast WITHOUT re-merging the LoRA (rank-r term in fp32); the forward stays bitwise the merged "
                        "path. Saves the second full-weight merge per step; dx differs only by the merge's bf16 rounding")
    p.add_argument("--text-enc-dtype", default=None, choices=[None, "fp16", "bf16", "fp32"])
    # holdout / selection / phase 2
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--holdout-min", type=int, default=3)
    p.add_argument("--train-on-holdout", action="store_true", default=False,
                   help="v11 (z-image hypothesis): also train on the holdout images and ship the final state; the "
                        "holdout screens become biased failure guards")
    p.add_argument("--member-passes", type=float, default=0.0,
                   help="v12: plan K = round(budget / (P passes per member)) fresh-seed members of exactly P passes each "
                        "(steps = P x n_train x cases per image), ship the soup of their finals; 0 = off. Needs "
                        "--train-on-holdout (no holdout pick exists at a fixed horizon); takes precedence over --member-slices")
    p.add_argument("--same-init", action="store_true", default=False,
                   help="v12 S2: every soup member starts from the SAME LoRA init (member 0's down, up = 0); the members "
                        "still differ by their own noise / sigma / data draws. Off (default): a fresh init seed per member. "
                        "A weight average keeps a coordinated multi-layer change only when the members share one basin")
    p.add_argument("--plain-delta", default="",
                   help="v12 P: regex over the diffusion model's parameter names; every matching NON-Linear parameter "
                        "(not .weight / .bias; e.g. krea2's per-block modulation constant mod.lin and RMSNorm .scale) "
                        "trains a full-rank fp32 delta that ships as a `.diff` key. Empty (default): LoRA only")
    p.add_argument("--eval-share", type=float, default=0.15, help="target fraction of wall-clock spent on holdout screens")
    p.add_argument("--confirm-top", type=int, default=3)
    p.add_argument("--confirm-noises", type=int, default=4, help="evaluator noise draws per stratum for the confirm stage")
    p.add_argument("--phase2", dest="phase2", action="store_true", default=True)
    p.add_argument("--no-phase2", dest="phase2", action="store_false")
    p.add_argument("--max-members", type=int, default=2)
    p.add_argument("--member-slices", type=int, default=0,
                   help="v10: K>1 trains K members from fresh seeds, member k for (time left)/(K-k), each with its own "
                        "screens, then soups their bests (the generic model-soup budget); 0 = plateau-driven phase 2")
    p.add_argument("--twin", dest="twin", action="store_true", default=True,
                   help="score the confirm stage through the evaluator twin on ideogram4/qwen-image")
    p.add_argument("--no-twin", dest="twin", action="store_false")
    p.add_argument("--identity-only", action="store_true", help="load, export the identity LoRA, exit (fallback path)")
    p.add_argument("--polish", action="store_true",
                   help="when a member plateaus and a second member cannot fit, restart from its best state with a low decaying LR "
                        "for the time left instead of training on past the optimum")
    p.add_argument("--polish-lr-frac", type=float, default=0.3, help="polish peak LR as a fraction of --lr")
    p.add_argument("--plateau-window", type=int, default=3, help="eval points without a real improvement that count as a plateau")
    p.add_argument("--screen-ema-only", action="store_true",
                   help="screen only the EMA candidate at intermediate eval points (raw at the member end), doubling the eval cadence")
    p.add_argument("--empty-prompt-frac", type=float, default=0.5,
                   help="share of training steps with the empty prompt; 0.5 alternates exactly like the score's 50/50 mix")
    p.add_argument("--select-metric", default="mean", choices=["mean", "p75", "worst"],
                   help="how the selector ranks candidates; mean = the evaluator's own reduction")
    p.add_argument("--pick-tiebreak", default="earliest", choices=["earliest", "latest"],
                   help="1-SE rule: earliest (shipped) or latest candidate of the best trajectory within one SE (v10 test)")
    p.add_argument("--eval-every", type=int, default=0, help="fixed eval cadence in steps (0 = derive from --eval-share)")
    p.add_argument("--holdout-names", default="", help="comma-separated image names to hold out (overrides the stratified choice)")
    p.add_argument("--train-base", default="cache", choices=["cache", "evaluator", "requant"],
                   help="ideogram4: train on the cache's per-row fp8 base (dequantised) or on the evaluator's per-tensor file")
    p.add_argument("--flip", action="store_true", help="train on a horizontally flipped twin of every training image as well (architecture v6 L3)")
    p.add_argument("--screen-passes", default="", help="screen schedule in passes over the (image, band) cases, e.g. 0.5,1,1.5,2")
    p.add_argument("--screen-max-share", type=float, default=0.30, help="cap on the wall-clock share of screens under --screen-passes")
    p.add_argument("--seed2", action="store_true", help="after the plateau: a clean second seed for s* steps + soup (v7)")
    p.add_argument("--adaptive-cadence", action="store_true",
                   help="space screens out while the holdout curve is still descending (late-optimum families)")
    p.add_argument("--band-power", type=float, default=0.0,
                   help="0 = uniform over sigma bands (like the score); p>0 repeats band b (loss_b/median)^p times per pass, "
                        "loss_b being the latest holdout per-band loss - importance sampling toward where the score mass is")
    p.add_argument("--band-stop", default="",
                   help="v12 L1: per-band training horizon as fractions of the member's progress p (planned steps under "
                        "--member-passes, else the clock): band b leaves the case cycle once p >= stop_b. 4 values = the "
                        "band groups 0-3 / 4-9 / 10-13 / 14-15, or 16 values; fitted from a scored trajectory "
                        "(research/v11_gpu/fit_band_stop.py on the T1 readout), never typed. Empty (default): every band to the end")
    p.add_argument("--sigma-head", default="off", choices=["off", "flux-guidance"],
                   help="v11: flux only - fit a per-band guidance g_b on the internal holdout after the pick and ship it as an exact "
                        "rank<=16 delta on time_in.out_layer iff it reproduces and beats the pick on fresh draws; any other "
                        "family or failure is a logged no-op")
    p.add_argument("--sigma-head-grid", default="1.5,2,2.5,3", help="guidance values scored against the evaluator's 3.5")
    p.add_argument("--sigma-head-noises", type=int, default=2,
                   help="evaluator noise draws per stratum for the fit ([0,k)); the adoption test uses the fresh draws [k,2k)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse(argv)
    if cfg.alpha is None:
        cfg.alpha = float(cfg.rank)
    if cfg.text_enc_dtype is None:
        cfg.text_enc_dtype = "bf16" if cfg.family == "qwen-image" else "fp16"
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(vars(cfg), open(os.path.join(cfg.out, "crown_config.json"), "w"), indent=1)   # not "config.json": HF tooling would read that as a model config

    from crown import comfy_boot, data, engine
    from crown.assets import Assets
    from crown.budget import Budget

    budget = Budget(cfg.deadline_ts)
    mm = comfy_boot.boot(cfg.comfy_root, text_enc_dtype=cfg.text_enc_dtype)
    if cfg.family == "flux":
        from crown import flux_autograd

        flux_autograd.apply()
    import torch

    engine.log(f"family={cfg.family} budget {budget.remaining() / 60:.1f} min; device {mm.get_torch_device()}")

    if cfg.identity_only:
        # lean fallback (v6.1 B3): only the diffusion weights are needed to write a loadable identity
        # LoRA; dataset, VAE and text encoders stay untouched so a failure there cannot repeat here
        from crown.lora import Lora, save_lora, select_targets

        assets = Assets(cfg.family, cfg.model_dir, cfg.baked_dir)
        model = engine.load_diffusion_model(assets.diffusion_sd(), cfg.family)
        targets = select_targets(model.model.diffusion_model, cfg.include, cfg.exclude) or select_targets(model.model.diffusion_model, None, None)
        lora = Lora(targets, cfg.rank, cfg.alpha, mm.get_torch_device(), cfg.seed)
        tmp = os.path.join(cfg.out, engine.C.OUTPUT_LORA_NAME + ".tmp")
        save_lora(tmp, lora.state(), lora.scale, metadata={"crown": "identity-only"})
        os.replace(tmp, os.path.join(cfg.out, engine.C.OUTPUT_LORA_NAME))
        engine.log(f"identity-only artifact written ({len(targets)} targets)")
        return

    items = data.load_items(cfg.dataset, cfg.trigger_word)
    hold = data.choose_holdout(items, cfg.holdout_frac, cfg.holdout_min)
    if getattr(cfg, "holdout_names", ""):
        wanted = {n.strip() for n in cfg.holdout_names.split(",") if n.strip()}
        hold = [i for i, it in enumerate(items) if it["name"] in wanted]
        if len(hold) != len(wanted):
            raise SystemExit(f"--holdout-names: {sorted(wanted - {items[i]['name'] for i in hold})} not in the dataset")
        chosen = set(hold)                      # choose_holdout already flagged its own split; re-flag
        for i, it in enumerate(items):
            it["holdout"] = i in chosen
    engine.log(f"dataset: {len(items)} images, holdout {len(hold)}: {[items[i]['name'] for i in hold]}")

    assets = Assets(cfg.family, cfg.model_dir, cfg.baked_dir)
    assets.train_base = getattr(cfg, "train_base", "cache")
    if assets.train_base != "cache":
        engine.log(f"training base: {assets.train_base}")

    t = time.time()
    kind, payload = assets.vae()
    enc = engine.LatentEncoder(kind, payload)
    for it in items:
        it["latent"] = enc.encode(it["image"])
    if getattr(cfg, "flip", False):
        twins = data.add_flips(items, enc.encode)
        items.extend(twins)
        engine.log(f"flip: {len(twins)} flipped training twins added (holdout untouched)")
    for it in items:
        it["image"] = None
    enc.free()
    engine.log(f"latents ({kind}) in {time.time() - t:.0f}s; shape {tuple(items[0]['latent'].shape)}")

    t = time.time()
    clip = engine.load_clip(assets.text_encoder_sds(), cfg.family)
    raw_conds = {}
    for prompt in sorted({it["caption"] for it in items} | {""}):
        raw_conds[prompt] = engine.encode_prompt(clip, prompt, cfg.family)
    del clip
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    engine.log(f"{len(raw_conds)} prompts encoded in {time.time() - t:.0f}s")

    t = time.time()
    model = engine.load_diffusion_model(assets.diffusion_sd(), cfg.family, assets.diffusion_meta())
    base_fp8 = getattr(assets, "base_fp8", False)
    assets.release()
    gc.collect()
    for it in items:
        it["scaled"] = model.model.process_latent_in(it["latent"].clone()).float().cpu()
        it["latent"] = None
    engine.log(f"model in {time.time() - t:.0f}s: {type(model.model).__name__} "
               f"{sum(p.numel() for p in model.model.diffusion_model.parameters()) / 1e9:.2f}B")

    twin = None
    if cfg.twin and (cfg.family in ("ideogram4", "qwen-image") or base_fp8):
        from crown.twin import EvaluatorTwin

        try:
            t = time.time()
            twin = EvaluatorTwin(cfg.family, assets, [it for it in items if it["holdout"]], raw_conds, model.load_device, cfg.baked_dir)
            engine.log(f"evaluator twin ready ({twin.source}) in {time.time() - t:.0f}s")
        except Exception as e:  # the twin is an instrument, never a reason to fail the run
            engine.log(f"evaluator twin unavailable: {type(e).__name__}: {e}")
            twin = None
    trainer = engine.Trainer(cfg, model, items, raw_conds, cfg.out, budget, twin=twin)
    guider = engine.make_guider(model, raw_conds[""])
    lat0 = items[0]["scaled"]
    guider.sample(torch.zeros_like(lat0), lat0, engine.TrainSampler(trainer), engine.evaluator_schedule(),
                  disable_pbar=True, seed=engine.C.EVAL_SAMPLER_SEED)
    engine.log("done")


if __name__ == "__main__":
    main()
