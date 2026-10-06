"""Dataset loading — preprocessing bit-identical to the evaluator — and holdout
construction.

Preprocessing mirrors validator/evaluation/image_test_data.py exactly (LANCZOS
to a 1024 long side, centre-crop to multiples of 16, RGB, sha256 of bytes+size)
so that a held-out image here is scored on the same tensor the evaluator would
build for it.
"""

import hashlib
import io
import math
import random
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

from . import contract as C


def adjust_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if min(w, h) <= 0:
        raise ValueError("Invalid image dimensions")
    size = (1024, int(h / w * 1024)) if w > h else (int(w / h * 1024), 1024)
    image = image.resize(size, Image.Resampling.LANCZOS)
    cw, ch = (v // 16 * 16 for v in size)
    left, top = (size[0] - cw) // 2, (size[1] - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).convert("RGB")


def image_digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()


def read_rows(source):
    """(name, raw_bytes, caption_or_None) for every image in a zip or directory."""
    source = Path(source)
    rows = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in C.IMAGE_EXTENSIONS:
                cap = path.with_suffix(".txt")
                rows.append((str(path.relative_to(source)), path.read_bytes(), cap.read_text(encoding="utf-8") if cap.exists() else None))
    else:
        with zipfile.ZipFile(source) as z:
            names = z.namelist()
            for name in sorted(names):
                if name.lower().endswith(C.IMAGE_EXTENSIONS):
                    cap = str(PurePosixPath(name).with_suffix(".txt"))          # zip paths are posix (v6.1 F8)
                    rows.append((name, z.read(name), z.read(cap).decode("utf-8", errors="replace") if cap in names else None))
    if not rows:
        raise ValueError("Empty image dataset")
    return rows


def load_items(source, trigger_word=None):
    """Decode, preprocess and hash every image. Captions are guaranteed by the
    validator's split (only image/text pairs enter a task), but we never crash
    on a missing one: the evaluator would score that image with its caption, and
    the closest thing we have is the trigger word or an empty string."""
    items, dropped = [], []
    for name, raw, caption in read_rows(source):
        try:
            image = adjust_image(Image.open(io.BytesIO(raw)))
            if min(image.size) < 16:
                raise ValueError(f"degenerate size {image.size}")
        except Exception as e:  # noqa: BLE001 - an undecodable training image must not end the run (v6.1 F2)
            dropped.append(f"{name}: {type(e).__name__}: {e}")
            continue
        # the evaluator conditions on the caption exactly as stored (no strip); only a blank one is missing (v6.1 F5)
        cap = caption if caption and caption.strip() else (trigger_word or "")
        items.append({"name": name, "image": image, "caption": cap, "sha256": image_digest(image),
                      "aspect": image.size[0] / image.size[1], "cap_len": len(cap.split())})
    if dropped:
        print(f"[data] {len(dropped)} unusable image(s) dropped: {dropped[:3]}", flush=True)
    if len(items) < 2:
        raise ValueError(f"fewer than 2 usable images ({len(items)}); dropped: {dropped[:3]}")
    # identical images (same digest) would let one copy train while its twin sits in the holdout and
    # would collide in the paired per-case keys: keep the first of each (v6.1 m12)
    seen, unique = set(), []
    for it in items:
        if it["sha256"] in seen:
            continue
        seen.add(it["sha256"])
        unique.append(it)
    if len(unique) != len(items):
        print(f"[data] {len(items) - len(unique)} duplicate image(s) dropped", flush=True)
    return unique


# --- holdout ------------------------------------------------------------------

def _strata_key(item):
    # aspect: portrait / square / landscape ; caption length: short / long (median split later)
    a = item["aspect"]
    shape = "L" if a > 1.1 else ("P" if a < 0.9 else "S")
    return shape


def add_flips(items, encode):
    """Architecture v6 L3: a horizontally flipped twin of every TRAINING image (holdout images are
    scored as they are). `encode(image) -> latent`; the twin carries the caption and a distinct
    digest. Returns the twins (the caller appends them)."""
    from PIL import ImageOps
    twins = []
    for it in items:
        if it.get("holdout") or it.get("image") is None:
            continue
        twins.append({"name": it["name"] + "#flip", "image": None, "caption": it["caption"],
                      "sha256": hashlib.sha256(b"flip:" + it["sha256"].encode()).hexdigest(), "holdout": False, "flip_of": it["name"],
                      "latent": encode(ImageOps.mirror(it["image"]))})
    return twins


def choose_holdout(items, frac=C.TEST_SPLIT_FRACTION, minimum=3):
    """Stratified holdout that mirrors the evaluator's own split size.

    The evaluator holds out ceil(0.10 * N_total) of the original set; we see the
    remaining 90%, so ceil(frac * len(items)) reproduces the same order of
    magnitude. `minimum` is the floor below which a paired standard error across
    images is meaningless. The draw is stratified over aspect shape and caption
    length so three images cannot all be one corner of the set, and it is seeded
    from the dataset's own content so the same dataset always yields the same
    split (a property the champion's fixed seed also had, without the stratification).
    """
    n = len(items)
    n_hold = min(n - 1, max(minimum, math.ceil(frac * n)))
    seed = int(hashlib.sha256("".join(sorted(i["sha256"] for i in items)).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)

    med = sorted(i["cap_len"] for i in items)[n // 2]
    buckets = {}
    for idx, it in enumerate(items):
        buckets.setdefault((_strata_key(it), it["cap_len"] >= med), []).append(idx)
    for b in buckets.values():
        rng.shuffle(b)
    # round-robin over buckets, largest first, so every populated stratum is represented
    order = sorted(buckets.values(), key=len, reverse=True)
    hold = []
    while len(hold) < n_hold:
        progressed = False
        for b in order:
            if b and len(hold) < n_hold:
                hold.append(b.pop()); progressed = True
        if not progressed:
            break
    hold = set(hold)
    for idx, it in enumerate(items):
        it["holdout"] = idx in hold
    return sorted(hold)
