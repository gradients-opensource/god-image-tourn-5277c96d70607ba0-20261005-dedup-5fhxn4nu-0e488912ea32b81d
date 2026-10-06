"""Validator entrypoint (ops/docker/*.dockerfile ENTRYPOINT).

Receives the validator's CLI (docs/miner.md "Image Trainer"), resolves cache
paths, launches the trainer with the family recipe, and publishes the artifact
at the exact path the uploader reads. Never exits without trying to leave a
loadable LoRA behind.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

START = time.time()
sys.path.insert(0, "/app")
from crown import contract as C  # noqa: E402

# Per-family recipe. Structural choices (which layers, rank) — everything
# adaptive (budget, selection, phase 2) is decided at runtime from measurements.
# Gradient checkpointing stays ON for every family until step time and peak memory are
# measured per family; turning it off is a speed optimisation, not a correctness one.
RECIPES = {
    "krea2":      ['--rank', '64', '--lr', '2e-4', '--train-on-holdout', '--member-passes', '2.5', '--fast-lora', '--no-ckpt', '--alpha', '32'],   # v12 adopted (apply_adopted.py): --train-on-holdout --member-passes 2.5 --fast-lora --no-ckpt --rank 64 --alpha 32
    "ideogram4":  ['--rank', '64', '--lr', '2e-4', '--include', 'layers[.][0-9]+[.](attention|adaln_modulation|feed_forward)', '--no-ckpt', '--train-base', 'requant', '--fast-lora', '--screen-passes', '0.5,1,1.5,2,2.5,3,3.5,4,5,6,8,10,12,16', '--train-on-holdout', '--member-passes', '2.2', '--alpha', '32'],   # fast-lora + screens on pass counts (I4D 2026-09-26, R1 rehearsal c7460d80): -0.03 % vs without (t -1.8, rule "not worse" met), -16 GB peak (66.4 -> 50.4 GB), -0.23 % vs the champion   # requant: measured -1.03% vs the dequantised base (2026-09-25, af4138aa), closes the gap to the champion   # block layers only: the evaluator re-quantises every patched tensor; the 7 embedder/final-layer tensors alone cost 0.7% (probe_requant 2026-09-24)   # twin released between scores -> fits: 0.68 s/step, peak 64.7 GB (smoke 2026-09-24); the OOM guard re-enables checkpointing if a task needs it   # v12 adopted (apply_adopted.py): --train-on-holdout --member-passes 2.2 --rank 64 --alpha 32
    "z-image":    ['--rank', '16', '--lr', '3e-4', '--train-on-holdout', '--member-passes', '4.5', '--fast-lora', '--no-ckpt'],   # v12 adopted (apply_adopted.py): --train-on-holdout --member-passes 4.5 --fast-lora --no-ckpt
    "flux":       ['--rank', '16', '--lr', '3e-4', '--fast-lora', '--no-ckpt', '--screen-passes', '0.5,1,1.5,2,2.5,3,3.5,4,5,6,8,10,12,16', '--train-on-holdout', '--member-passes', '2.5'],   # NO --exact-merge: the validator scores on an A100, where Comfy loads an fp8 flux checkpoint as bf16 (unet_dtype: no fp8 compute, 24 GB fits) and merges without stochastic rounding; re-scored A100-style (FA 2026-09-26) the exact-merge arm E2 was +0.85 % vs D (t +15) and lost to the champion by 0.60 %, while D beat it by 0.25 %   # fast-lora + no-ckpt + screens on pass counts: G1 smoke 1.01 s/step, peak 53.5 GB; the OOM guard re-enables checkpointing if a task needs it   # v12 adopted (apply_adopted.py): --train-on-holdout --member-passes 2.5
    "qwen-image": ['--rank', '64', '--lr', '1e-4', '--include', '.', '--train-on-holdout', '--fast-lora', '--no-ckpt', '--alpha', '22.63', '--fast-lora-lowrank'],   # the public champion recipe's target set (240 vs 846 Linears): 1.47 -> ~1.05 s/step; Round 2 can be a qwen knockout   # v12 adopted (apply_adopted.py): --train-on-holdout --fast-lora --no-ckpt --include . --rank 64 --alpha 22.63 --fast-lora-lowrank
}
# Data-size rules (v12 addenda KN / ZN / FN, docs/v12_phase_max_maximum_form.md): at fixed member passes a
# member's steps scale with the number of given images, so a small dataset gets few steps and wants a larger
# step (muA reading: progress ~ step x steps). (family, max given images, extra args appended after the recipe;
# the last value wins.) The cut n_given <= 13 is N <= 15 under the validator's 10 % hidden draw. Each row ships
# on its own pre-registered verdict (paired on two fresh splits; an owner override is stated on its row):
#   KN krea2: kpR's alpha 64 (2x kp64's step); kpR - kp64 -0.1338 % (se 0.0251, t -5.32, 25 images)
#   ZR z-image: zp2x's alpha 32 over the round-1 range (2x zp's step at rank 16); zp2x - zp -0.0494 % (se 0.0270, t -1.83, 21 images) [shipped by OWNER OVERRIDE 10-05; the measured verdict was STAYS]
SIZE_RULES = [
    ("krea2", 13, ["--alpha", "64"]),
    ("z-image", 27, ["--alpha", "32"]),
]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def count_images(dataset):
    """The number of images in the task's dataset zip (the given images); None if it cannot be read."""
    import zipfile
    try:
        with zipfile.ZipFile(dataset) as z:
            return sum(1 for n in z.namelist() if n.lower().endswith(IMAGE_EXTS)
                       and "__MACOSX" not in n and not os.path.basename(n).startswith("."))
    except Exception:  # noqa: BLE001 - an unreadable zip leaves the recipe as it is
        return None


def size_overrides(family, dataset):
    """Extra trainer args chosen from the dataset size (SIZE_RULES); [] when no rule applies or the size is unknown."""
    n = count_images(dataset)
    if n is None:
        return []
    for fam, max_n, extra in SIZE_RULES:
        if fam == family and n <= max_n:
            return list(extra)
    return []


COMMON = ["--holdout-frac", "0.10", "--holdout-min", "3", "--eval-share", "0.15",
          "--confirm-top", "3", "--confirm-noises", "4", "--ema", "0.99", "--phase2", "--max-members", "2"]


def logs_enabled():
    # on by default: the validator keeps container logs, and one line per 25 steps is the only
    # diagnostic a failed task leaves behind (set GOD_TRAIN_LOGS=0 to silence)
    return os.environ.get("GOD_TRAIN_LOGS", "1").strip().lower() not in {"0", "false", "no", "off"}


def say(*a):
    print(*a, flush=True)


def resolve_model_dir(model):
    from crown.assets import resolve_model_dir as r

    return r(model)


def run_trainer(args, family, model_dir, dataset, work, deadline, extra):
    cmd = [sys.executable, "/app/crown/train.py", "--family", family, "--model-dir", model_dir, "--dataset", dataset,
           "--out", work, "--deadline-ts", str(deadline)] + RECIPES[family] + COMMON + extra
    cmd += shlex.split(os.environ.get("CROWN_EXTRA_ARGS", ""))   # local experiments only; unset on the validator
    if args.trigger_word:
        cmd += ["--trigger-word", args.trigger_word]
    if logs_enabled():
        say("running:", " ".join(cmd))
    # the child must be gone before the validator's own clock (deadline) kills the container:
    # whatever it saved by then is already in /app/checkpoints (see main), so a kill loses nothing
    limit = max(30.0, deadline - time.time() - 60.0)
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        say(f"trainer exceeded its wall-clock ({limit:.0f}s); terminating it, the published artifact stands")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 124


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    a, unknown = p.parse_known_args()
    if unknown:
        say(f"ignoring unknown validator arguments: {unknown}")
    os.environ.setdefault("GOD_TRAIN_LOGS", "1")

    family = a.model_type.lower().replace("_", "-")
    if family not in RECIPES:
        say(f"unsupported model type {a.model_type}")
        sys.exit(2)
    dataset = str(C.dataset_zip(a.task_id))
    if not os.path.exists(dataset) and os.path.exists(a.dataset_zip):
        dataset = a.dataset_zip
    model_dir = resolve_model_dir(a.model)
    out_final = str(C.output_dir(a.task_id, a.expected_repo_name))
    # the trainer writes straight into the validator's checkpoints volume (atomic tmp + os.replace per
    # save), so an artifact is on the volume from the identity save onwards: a timeout kill, a crash
    # or a hang after that point still leaves the best-so-far where the uploader reads it
    os.makedirs(out_final, exist_ok=True)
    work = out_final
    deadline = START + a.hours_to_complete * 3600.0

    extra = size_overrides(family, dataset)
    say(f"data-size rule: {family}, {count_images(dataset)} given images -> {' '.join(extra) or 'the recipe as it is'}")
    rc = run_trainer(a, family, model_dir, dataset, work, deadline, extra)
    src = os.path.join(work, C.OUTPUT_LORA_NAME)

    # Fallback: nothing produced (crash before the identity save) and time remains -> lean identity-only
    # run (diffusion weights only: no dataset, VAE or text encoders)
    if not os.path.exists(src) and (deadline - time.time()) > 3 * 60:
        say(f"trainer rc={rc} with no artifact; running the identity-only fallback")
        rc2 = run_trainer(a, family, model_dir, dataset, work, deadline, ["--identity-only"])
        say(f"identity-only rc={rc2}")

    if os.path.exists(src):
        # the validator uploads only when the container exits 0: a published artifact is a success
        # whatever the trainer's own exit status was
        say(f"published {src} (trainer rc={rc})")
        sys.exit(0)
    say("no checkpoint produced")
    sys.exit(1)


if __name__ == "__main__":
    main()
