# crown — SN56 image-tournament trainer

Our submission for the Gradients (SN56) image tournament. Built from the
validator's own evaluator source, not from any miner's reproduction. See
`../docs/research_evaluator_and_field.md` for every fact this design rests on
and `../docs/strategy_v3_crown.md` for why these levers and not others.

## What it does

```
validator CLI ─► scripts/image_trainer.py ─► crown/train.py
                                              │  load base / text encoders / VAE  (assets.py, evaluator formats)
                                              │  preprocess + hash images          (data.py, bit-identical to evaluator)
                                              │  stratified holdout                (data.py)
                                              │  encode latents, process_latent_in
                                              │  encode prompts once
                                              └► engine.Trainer inside CFGGuider.sample
                                                    identity LoRA saved first        (an artifact always exists)
                                                    train: uniform (image, band) cycle, 50/50 caption/empty,
                                                           wall-clock cosine LR, EMA
                                                    screen candidates raw/ema/avg2 at 1 evaluator noise draw
                                                    plateau ─► second member from a new seed ─► soup
                                                    confirm top-k at more draws ─► 1-SE pick ─► save
                                              ─► publish /app/checkpoints/{task}/{repo}/last.safetensors (+summary.json)
```

The loss is the evaluator's loss (`scoring.py` reproduces
`validator/evaluation/denoising_mse.py`; the unit test in this repo's history
shows 0.0 difference on 256 cases). Candidates are compared on the same noise
draws, so selection uses a paired standard error rather than a guess.

## Layout

```
crown/contract.py       every path / constant the validator dictates, with sources
crown/scoring.py        case seeds, sigmas, flow MSE, HoldoutScorer, paired diff
crown/data.py           evaluator preprocessing, stratified holdout
crown/assets.py         per-family weights in evaluator format (parity notes inline)
crown/lora.py           LoRA params, Comfy weight-function hook, export, soup
crown/budget.py         measured wall-clock controller
crown/selector.py       racing + 1-SE rule + plateau test
crown/engine.py         the loop
crown/train.py          orchestration + CLI
crown/flux_autograd.py  out-of-place Flux blocks (upstream uses in-place adds)
scripts/image_trainer.py  validator entrypoint, recipes, publish, fallback
ops/docker/*.dockerfile   evaluator-pinned stack; bakes the two evaluator VAEs
tools/local_run.sh        production-faithful docker run (/cache ro, 110g, 24 cpu, no net)
tools/prepare_cache.py    build ./cache like the validator's downloader does
tools/check_lora_keys.py  G0: our export loads with zero unmatched keys
tools/local_evaluate.py   G-PARITY: score a LoRA with the validator's evaluator, offline
```

## Running locally (needs one H100-class GPU)

```
# 1. cache for one task (dataset zip + base model + text encoder)
python tools/prepare_cache.py --cache ./cache --task-dir ../research/datasets/5c1de437_krea2 \
    --model krea/Krea-2-Raw --model-type krea2            # HF_TOKEN for gated repos
# 2. train exactly as the validator would
GOD_TRAIN_LOGS=1 tools/local_run.sh 5c1de437-e407-418a-a896-210c3d128150 krea/Krea-2-Raw krea2 0.75
# 3. G0 — does it load like the evaluator loads it?
docker run --rm --gpus all -v $PWD/cache:/cache:ro -v $PWD/outputs:/out crown-image-trainer \
    python tools/check_lora_keys.py --family krea2 --model-dir /cache/models/krea--Krea-2-Raw --lora /out/<task>/<repo>/last.safetensors
# 4. G-PARITY — score with the validator's evaluator image (built from research/god_upstream)
python tools/local_evaluate.py --family krea2 --base <raw.safetensors> --lora <last.safetensors> --dataset <holdout dir> --out eval.json
```

## Gates before any lever ships

| gate | passes when |
| --- | --- |
| G0 | 5/5 families produce `last.safetensors` under a short budget; loader reports 0 unmatched keys; evaluator repeatability check passes |
| G-PARITY | internal holdout == `local_evaluate.py` score on the same images: < 0.05% on krea2/z-image/flux; gap *measured* on qwen-image/ideogram4 |
| G-L1 | phase-2 (plateau → member → soup) beats "ship phase-1 best" on evaluator score, per family, on held-out datasets |
| G-L3 | confirm-stage argmin differs from 1-noise argmin often enough to matter, and the 1-SE pick is never significantly worse |

## Flags that matter

`--phase2/--no-phase2`, `--max-members`, `--confirm-noises`, `--confirm-top`,
`--eval-share` (target fraction of wall-clock on screens; cadence adapts to the
measured eval/step costs), `--holdout-frac/--holdout-min`, `--include/--exclude`
(LoRA target regex; default = every Linear), `--ckpt/--no-ckpt`.

## FLUX note

The validator cache for `rayonlabs/FLUX.1-dev` is a plain snapshot: a *bare*
`flux1-dev.safetensors` (no `model.diffusion_model.` prefix) plus
`ae.safetensors`, and **no text-encoder weights**. `assets.py` therefore loads
the bare model, takes the VAE from `ae.safetensors`, and reads `clip_l` +
`t5xxl_fp16` from the baked assets (`comfyanonymous/flux_text_encoders` at the
revision the evaluator resolves). The 0921 winner's code assumes a full
checkpoint here and produces no artifact on this cache — see
`../docs/research_evaluator_and_field.md` §4.3.

## Evaluator twin (`crown/twin.py`) — L2a

On ideogram4 and qwen-image the evaluator's numerics differ from any training
hook (different base quantisation, stochastic-rounding merge seeded by
`crc32(key)`, fp8 activations). Instead of re-implementing them, the twin loads
a second copy of the base **exactly as the evaluator does** (ideogram4: the
baked `Comfy-Org` per-tensor `comfy_quant` file the evaluator hard-codes;
qwen-image: the same plain-fp8 file), applies each candidate with
`comfy.sd.load_lora_for_models` (what `nodes.LoraLoader` calls — unmatched
keys are caught the way the evaluator catches them), and scores under
`CFGGuider.sample`. It is used for:

- the **parity meter** — identity state scored by hook vs twin at the same
  draws; the gap is written to `summary.json["parity"]`;
- the **confirm stage** — top candidates scored the evaluator's way before the
  1-SE pick;
- the **final loadability check** — every family, every run, on the real
  loader path (`verify_loadable`), with fallback to the best loadable candidate.

`--no-twin` disables it. Memory: +9.3 GB (ideogram4) / +20 GB (qwen-image).

## Not yet in this tree (see strategy v3)

- L2b: *training* under the evaluator's fp8 numerics (straight-through
  requantise + `QuantLinearFunc`). The twin already gives exact selection;
  L2b would additionally move the optimum. Deferred until measured.
- L5 N-aware schedule horizon (the plateau controller covers most of it).
- L7 text-encoder parity.
