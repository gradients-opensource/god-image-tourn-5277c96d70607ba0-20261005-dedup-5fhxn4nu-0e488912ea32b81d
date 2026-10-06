"""CPU check of the asset resolution against every flux repo layout the validator can present
(and two hypothetical ones), using tiny fake safetensors and a patched file-size rule:

  snapshot_bare     root bare transformer + ae.safetensors + diffusers dirs   (rayonlabs)
  snapshot_multi    three root files (bf16 / fp8 / nf4) + dirs               (PixelWave; evaluator rule ambiguous -> logged)
  allinone_fp8      single root file, model.diffusion_model.* in fp8 + vae + text encoders (MonochromeManga)
  bare_single       single root bare bf16 transformer, nothing else          (fluxunchained after normalisation)
  diffusers_only    no root file, transformer/ folder only                   (hypothetical: no loadable base -> clean error)
  scaled_fp8        single root file with a scaled_fp8 marker                 (hypothetical: must not be silently cast)

Asserts: the diffusion dict is the bare transformer keys; fp8 dtype is preserved and flagged (base_fp8);
the VAE always comes from the baked rayonlabs ae; text encoders from the baked pair; ambiguous or missing
roots produce the documented log line / a FileNotFoundError (never a wrong file).

T5 fallback (the build bakes t5xxl_fp8_e4m3fn_scaled only when t5xxl_fp16 did not fit its clock):

  t5_fp8_fallback         no fp16 T5, scaled fp8 T5 baked -> fp8 T5 dequantised to the fp16 file's layout, logged
  t5_fp8_plain            the same with a plain (unscaled) fp8 file -> exact upcast
  t5_none_fails           neither T5 baked, rayonlabs as modeled above (no text_encoder dirs) and fluxunchained ->
                          the same FileNotFoundError as before the fallback existed
  t5_fp16_unchanged       fp16 T5 baked next to an unreadable fp8 file -> the baked pair exactly as before; the fp8
                          file is never opened
  t5_repo_before_fp8      the real rayonlabs snapshot (text_encoder/ + sharded text_encoder_2/) -> the repo's own T5
                          (bf16; equal to t5xxl_fp16 once cast, on sampled tensors) wins over the fp8 file
  t5_fp8_before_embedded  MonochromeManga (embeds the plain fp8 T5) -> the baked scaled fp8 T5 wins
  t5_fp8_name_matches_build  the trainer's file names are the ones fetch_assets.sh writes, fp8 only without fp16

    python tools/check_layouts.py     (exit 1 on any failure)
"""
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from crown import assets as A  # noqa: E402

RESULTS = []
BIG = set()   # paths that pretend to be > 5 GiB

_real_getsize = os.path.getsize


def fake_getsize(p):
    return 6 * 1024 ** 3 if os.path.abspath(p) in BIG else _real_getsize(p)


A.os.path.getsize = fake_getsize   # assets.py calls os.path.getsize through its own os import


def bare(dtype=torch.bfloat16, prefix=""):
    return {f"{prefix}double_blocks.0.img_attn.qkv.weight": torch.zeros(4, 4, dtype=dtype),
            f"{prefix}single_blocks.0.linear1.weight": torch.zeros(4, 4, dtype=dtype),
            f"{prefix}img_in.weight": torch.zeros(4, 4, dtype=dtype)}


def write(path, sd, big=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(sd, path)
    if big:
        BIG.add(os.path.abspath(path))


T5_W = torch.tensor([[0.5, -1.0], [2.0, 0.25]])   # on the e4m3 grid, so the fp8 copy is exact
T5_SCALE = 0.01
# the file name the build writes (ops/docker/fetch_assets.sh), spelled out rather than read from crown.contract so the
# checks also run against a trainer without the fallback (base control) and t5_fp8_name_matches_build ties the two
FP8_T5_FILE = "t5xxl_fp8_e4m3fn_scaled.safetensors"
FETCH_ASSETS = os.path.join(os.path.dirname(HERE), "ops", "docker", "fetch_assets.sh")


def fake_t5_fp8_scaled():
    """t5xxl_fp8_e4m3fn_scaled's format: scaled_fp8 marker, fp8 Linear weight + f32 scalar scale_weight, fp16 rest."""
    return {"scaled_fp8": torch.zeros(0, dtype=torch.float8_e4m3fn),
            "encoder.x.weight": T5_W.to(torch.float8_e4m3fn),
            "encoder.x.scale_weight": torch.tensor(T5_SCALE, dtype=torch.float32),
            "encoder.final_layer_norm.weight": torch.full((2,), 0.75, dtype=torch.float16),
            "shared.weight": torch.full((2, 2), 3.0, dtype=torch.float16)}


def fake_t5_fp8_plain():
    """t5xxl_fp8_e4m3fn's format: every tensor fp8, no scales, no marker."""
    return {"encoder.x.weight": T5_W.to(torch.float8_e4m3fn),
            "encoder.final_layer_norm.weight": torch.full((2,), 0.75).to(torch.float8_e4m3fn),
            "shared.weight": torch.full((2, 2), 3.0).to(torch.float8_e4m3fn)}


def baked_dir(t5="fp16"):
    """t5: which T5 files are baked, '+'-joined: fp16, fp8 (scaled), fp8plain (plain content under the fp8 name),
    fp8junk (unreadable bytes under the fp8 name: a path that opens it fails), none."""
    d = tempfile.mkdtemp(prefix="baked-")
    write(os.path.join(d, "ae.safetensors"), {"decoder.conv_in.weight": torch.zeros(2, 2)}, big=False)
    write(os.path.join(d, "clip_l.safetensors"), {"text_model.x.weight": torch.zeros(2, 2)}, big=False)
    parts = t5.split("+")
    if "fp16" in parts:
        write(os.path.join(d, "t5xxl_fp16.safetensors"), {"encoder.x.weight": torch.zeros(2, 2)}, big=False)
    fp8 = os.path.join(d, FP8_T5_FILE)
    if "fp8" in parts:
        write(fp8, fake_t5_fp8_scaled(), big=False)
    if "fp8plain" in parts:
        write(fp8, fake_t5_fp8_plain(), big=False)
    if "fp8junk" in parts:
        with open(fp8, "wb") as f:
            f.write(b"\xff" * 64)
    return d


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True)); print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False)); print(f"FAIL {name}: {type(e).__name__}: {e}")


def resolve(model_dir):
    a = A.Assets("flux", model_dir, baked_dir())
    sd = a.diffusion_sd()
    kind, vae = a.vae()
    tes = a.text_encoder_sds()
    return a, sd, kind, vae, tes


def expect_bare(sd, dtype):
    assert set(sd) == set(bare()), f"unexpected keys {sorted(sd)[:3]}"
    assert all(v.dtype == dtype for v in sd.values()), {k: v.dtype for k, v in sd.items()}


def snapshot_bare():
    d = tempfile.mkdtemp(prefix="rayon-")
    write(os.path.join(d, "flux1-dev.safetensors"), bare())
    write(os.path.join(d, "ae.safetensors"), {"decoder.conv_in.weight": torch.ones(2, 2)}, big=False)
    os.makedirs(os.path.join(d, "transformer")); os.makedirs(os.path.join(d, "vae"))
    a, sd, kind, vae, tes = resolve(d)
    expect_bare(sd, torch.bfloat16)
    assert not a.base_fp8
    assert kind == "comfy" and float(vae["decoder.conv_in.weight"].sum()) == 0.0, "VAE must be the baked rayonlabs ae, not the repo copy"
    assert len(tes) == 2


def snapshot_multi():
    d = tempfile.mkdtemp(prefix="pixel-")
    write(os.path.join(d, "pixelwave_bf16.safetensors"), bare())
    write(os.path.join(d, "pixelwave_fp8.safetensors"), bare(torch.float8_e4m3fn))
    write(os.path.join(d, "pixelwave_nf4.safetensors"), {"x": torch.zeros(2, 2)})
    os.makedirs(os.path.join(d, "transformer"))
    # the bf16 file must win by size: make it the largest
    _sizes = {os.path.abspath(os.path.join(d, "pixelwave_bf16.safetensors")): 24 * 1024 ** 3,
              os.path.abspath(os.path.join(d, "pixelwave_fp8.safetensors")): 12 * 1024 ** 3,
              os.path.abspath(os.path.join(d, "pixelwave_nf4.safetensors")): 6 * 1024 ** 3}
    A.os.path.getsize = lambda p: _sizes.get(os.path.abspath(p), fake_getsize(p))
    try:
        a, sd, kind, vae, tes = resolve(d)
    finally:
        A.os.path.getsize = fake_getsize
    expect_bare(sd, torch.bfloat16)
    assert not a.base_fp8


def allinone_fp8():
    d = tempfile.mkdtemp(prefix="mono-")
    sd_all = bare(torch.float8_e4m3fn, prefix="model.diffusion_model.")
    sd_all.update({"vae.decoder.conv_in.weight": torch.ones(2, 2), "text_encoders.clip_l.transformer.x.weight": torch.ones(2, 2),
                   "text_encoders.t5xxl.transformer.x.weight": torch.ones(2, 2)})
    write(os.path.join(d, "FLUX-DEV_MonochromeManga.safetensors"), sd_all)
    a, sd, kind, vae, tes = resolve(d)
    expect_bare(sd, torch.float8_e4m3fn)
    assert a.base_fp8, "fp8 base not flagged (the twin would not be built)"
    assert float(vae["decoder.conv_in.weight"].sum()) == 0.0, "VAE must be the baked rayonlabs ae, not the embedded one"
    assert len(tes) == 2 and float(tes[0]["text_model.x.weight"].sum()) == 0.0, "text encoders must be the baked pair"


def bare_single():
    d = tempfile.mkdtemp(prefix="unch-")
    write(os.path.join(d, "fluxunchained-dev-fp16.safetensors"), bare())
    a, sd, kind, vae, tes = resolve(d)
    expect_bare(sd, torch.bfloat16)
    assert kind == "comfy" and len(tes) == 2


def diffusers_only():
    d = tempfile.mkdtemp(prefix="diff-")
    os.makedirs(os.path.join(d, "transformer"))
    write(os.path.join(d, "transformer", "diffusion_pytorch_model.safetensors"), bare())
    try:
        resolve(d)
        raise AssertionError("a repo without a root checkpoint must fail loudly (identity fallback), not load a wrong file")
    except FileNotFoundError:
        pass


def scaled_fp8():
    d = tempfile.mkdtemp(prefix="scaled-")
    sd_s = bare(torch.float8_e4m3fn)
    sd_s["scaled_fp8"] = torch.zeros(0, dtype=torch.float8_e4m3fn)
    sd_s["double_blocks.0.img_attn.qkv.scale_weight"] = torch.ones(1)
    write(os.path.join(d, "flux-scaled.safetensors"), sd_s)
    a, sd, kind, vae, tes = resolve(d)
    assert a.base_fp8 and "scaled_fp8" in sd, "scaled-fp8 marker must survive to Comfy's loader (never cast away)"


# --- T5 fallback ---------------------------------------------------------------------------------------------------

def te_logged(model_dir, bd):
    """(text encoder dicts, the [assets] lines they printed); an exception carries its log in .log"""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            tes = A.Assets("flux", model_dir, bd).text_encoder_sds()
    except Exception as e:
        e.log = buf.getvalue()
        raise
    return tes, buf.getvalue()


def layout_rayon_modeled():          # as snapshot_bare: root file + ae + empty diffusers dirs, no text encoders
    d = tempfile.mkdtemp(prefix="rayon-")
    write(os.path.join(d, "flux1-dev.safetensors"), bare())
    write(os.path.join(d, "ae.safetensors"), {"decoder.conv_in.weight": torch.ones(2, 2)}, big=False)
    os.makedirs(os.path.join(d, "transformer")); os.makedirs(os.path.join(d, "vae"))
    return d


def layout_unchained():              # fluxunchained after the validator's standalone normalisation
    d = tempfile.mkdtemp(prefix="unch-")
    write(os.path.join(d, "fluxunchained-dev-fp16.safetensors"), bare())
    return d


def layout_mono():                   # FLUX-MonochromeManga: all-in-one fp8, embeds clip_l and the plain fp8 T5
    d = tempfile.mkdtemp(prefix="mono-")
    sd_all = bare(torch.float8_e4m3fn, prefix="model.diffusion_model.")
    sd_all.update({"vae.decoder.conv_in.weight": torch.ones(2, 2), "text_encoders.clip_l.transformer.x.weight": torch.ones(2, 2),
                   "text_encoders.t5xxl.transformer.x.weight": torch.ones(2, 2).to(torch.float8_e4m3fn)})
    write(os.path.join(d, "FLUX-DEV_MonochromeManga.safetensors"), sd_all)
    return d


def layout_rayon_real():             # the validator's full snapshot of rayonlabs/FLUX.1-dev (and PixelWave): bf16 CLIP-L
    d = layout_rayon_modeled()       # in text_encoder/, bf16 T5 in two shards + index in text_encoder_2/
    write(os.path.join(d, "text_encoder", "model.safetensors"), {"text_model.x.weight": torch.full((2, 2), 5.0, dtype=torch.bfloat16)}, big=False)
    te2 = os.path.join(d, "text_encoder_2")
    write(os.path.join(te2, "model-00001-of-00002.safetensors"), {"encoder.x.weight": torch.full((2, 2), 6.0, dtype=torch.bfloat16)}, big=False)
    write(os.path.join(te2, "model-00002-of-00002.safetensors"), {"shared.weight": torch.full((2, 2), 7.0, dtype=torch.bfloat16)}, big=False)
    with open(os.path.join(te2, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {"encoder.x.weight": "model-00001-of-00002.safetensors",
                                  "shared.weight": "model-00002-of-00002.safetensors"}}, f)
    return d


def expect_fp8_t5(tes, log, plain=False):
    assert "[assets] fallback: fp8 T5" in log, f"fp8 fallback not logged: {log!r}"
    assert len(tes) == 2 and float(tes[0]["text_model.x.weight"].sum()) == 0.0, "CLIP-L must be the baked clip_l"
    t5 = tes[1]
    assert set(t5) == {"encoder.x.weight", "encoder.final_layer_norm.weight", "shared.weight"}, f"scale keys / marker left: {sorted(t5)}"
    assert all(v.dtype == torch.float16 for v in t5.values()), {k: v.dtype for k, v in t5.items()}
    want = T5_W if plain else T5_W * T5_SCALE
    assert torch.equal(t5["encoder.x.weight"], want.to(torch.float16)), (t5["encoder.x.weight"], want)
    assert torch.equal(t5["encoder.final_layer_norm.weight"], torch.full((2,), 0.75, dtype=torch.float16))
    assert torch.equal(t5["shared.weight"], torch.full((2, 2), 3.0, dtype=torch.float16))


def t5_fp8_fallback():
    tes, log = te_logged(layout_unchained(), baked_dir("fp8"))
    expect_fp8_t5(tes, log)
    # the modeled rayonlabs layout has no text_encoder dirs: the fp8 T5 is used there too
    tes, log = te_logged(layout_rayon_modeled(), baked_dir("fp8"))
    expect_fp8_t5(tes, log)


def t5_fp8_plain():
    tes, log = te_logged(layout_unchained(), baked_dir("fp8plain"))
    expect_fp8_t5(tes, log, plain=True)


def t5_none_fails():
    for layout in (layout_rayon_modeled, layout_unchained):
        bd = baked_dir("none")
        try:
            te_logged(layout(), bd)
        except FileNotFoundError as e:
            want = f"flux text encoders not baked under {bd} and not embedded in the checkpoint"
            assert str(e) == want, f"{layout.__name__}: {e} != {want}"
            assert "fp8" not in e.log and "fallback: text encoders embedded in the checkpoint" in e.log, e.log
        else:
            raise AssertionError(f"{layout.__name__}: no T5 anywhere must fail loudly (identity fallback), as before")


def t5_fp16_unchanged():
    for layout in (layout_rayon_modeled, layout_unchained, layout_mono, layout_rayon_real):
        bd = baked_dir("fp16+fp8junk")        # opening the fp8 file would raise
        tes, log = te_logged(layout(), bd)
        assert log == "[assets] text encoders: baked clip_l + t5xxl\n", f"{layout.__name__}: {log!r}"
        files = [A.load_file(os.path.join(bd, n)) for n in A.C.BAKED_FLUX_TE]
        assert len(tes) == 2 and all(set(t) == set(f) and all(torch.equal(t[k], f[k]) and t[k].dtype == f[k].dtype for k in f)
                                     for t, f in zip(tes, files)), f"{layout.__name__}: not the baked pair as stored"


def t5_repo_before_fp8():
    tes, log = te_logged(layout_rayon_real(), baked_dir("fp8junk"))
    assert log == "[assets] fallback: text encoders from the task repo's text_encoder/text_encoder_2\n", log
    assert float(tes[0]["text_model.x.weight"][0, 0]) == 5.0 and float(tes[1]["encoder.x.weight"][0, 0]) == 6.0
    assert float(tes[1]["shared.weight"][0, 0]) == 7.0, "both text_encoder_2 shards must be read"


def t5_fp8_before_embedded():
    tes, log = te_logged(layout_mono(), baked_dir("fp8"))
    expect_fp8_t5(tes, log)
    assert "embedded" not in log, log


def t5_fp8_name_matches_build():
    """The trainer looks for the files the build writes: fetch_assets.sh fetches each baked text-encoder name from the
    pinned revision, and tries the fp8 T5 only when the fp16 one is absent."""
    src = open(FETCH_ASSETS, encoding="utf-8").read()
    assert A.C.BAKED_FLUX_T5_FP8 == FP8_T5_FILE, (A.C.BAKED_FLUX_T5_FP8, FP8_T5_FILE)
    for name in A.C.BAKED_FLUX_TE + (A.C.BAKED_FLUX_T5_FP8,):
        want = f"fetch {name} "
        assert any(ln.strip().startswith(want) and f"/resolve/${{REV}}/{name}" in ln for ln in src.splitlines()), \
            f"fetch_assets.sh does not fetch {name} from the pinned revision"
    guard = f'if [ ! -f "$DEST/{A.C.BAKED_FLUX_TE[1]}" ]; then'
    assert guard in src and src.index(guard) < src.index(f"fetch {A.C.BAKED_FLUX_T5_FP8} "), "fp8 T5 must be fetched only without the fp16 one"


if __name__ == "__main__":
    for name, fn in [("snapshot_bare", snapshot_bare), ("snapshot_multi", snapshot_multi), ("allinone_fp8", allinone_fp8),
                     ("bare_single", bare_single), ("diffusers_only", diffusers_only), ("scaled_fp8", scaled_fp8),
                     ("t5_fp8_fallback", t5_fp8_fallback), ("t5_fp8_plain", t5_fp8_plain), ("t5_none_fails", t5_none_fails),
                     ("t5_fp16_unchanged", t5_fp16_unchanged), ("t5_repo_before_fp8", t5_repo_before_fp8),
                     ("t5_fp8_before_embedded", t5_fp8_before_embedded), ("t5_fp8_name_matches_build", t5_fp8_name_matches_build)]:
        check(name, fn)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} layout checks passed")
    sys.exit(1 if bad else 0)
