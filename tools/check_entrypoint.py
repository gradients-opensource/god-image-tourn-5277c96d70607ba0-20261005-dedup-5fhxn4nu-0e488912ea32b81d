"""CPU checks for the validator entrypoint (scripts/image_trainer.py) after v6.1:

  publish_exit0     a trainer that publishes last.safetensors then exits 1 -> entrypoint exits 0
  timeout_kill      a trainer that hangs is killed before the deadline; the file it wrote stays published
  fallback_identity a trainer that produces nothing -> the identity-only fallback is invoked once
  nothing_exit1     nothing produced by either run -> exit 1
  size_rule         v12 KN / ZN / FN: every SIZE_RULES row fires at <= its size and not above, nothing else
  recipes_parse     every family's argv, exactly as run_trainer builds it, parses in crown/train.py and the
                    recipe's switches land in the config (a typo in RECIPES would otherwise surface on the validator)

The real trainer is replaced by a tiny fake script; cache/checkpoint roots are redirected to a temp dir.

    python tools/check_entrypoint.py        (exit 1 on any failure)
"""
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("image_trainer", os.path.join(ROOT, "scripts", "image_trainer.py"))
it = importlib.util.module_from_spec(spec)
spec.loader.exec_module(it)

RESULTS = []
FAKE = os.path.join(tempfile.gettempdir(), "crown_fake_trainer.py")
open(FAKE, "w").write('''
import os, sys, time
mode = os.environ["FAKE_MODE"]
out = sys.argv[sys.argv.index("--out") + 1]
if "--identity-only" in sys.argv:
    open(os.path.join(out, "last.safetensors"), "wb").write(b"identity" * 300)
    sys.exit(0)
if mode == "publish_then_crash":
    open(os.path.join(out, "last.safetensors"), "wb").write(b"trained" * 300)
    sys.exit(1)
if mode == "hang":
    open(os.path.join(out, "last.safetensors"), "wb").write(b"partial-best" * 300)
    time.sleep(3600)
if mode == "nothing":
    sys.exit(3)
if mode == "nothing_even_identity":
    sys.exit(3)
''')


def setup(mode, hours):
    tmp = tempfile.mkdtemp(prefix="crown-ep-")
    it.C.CHECKPOINTS_ROOT = Path(tmp) / "checkpoints"
    it.C.output_dir = lambda t, r: it.C.CHECKPOINTS_ROOT / t / r
    it.C.dataset_zip = lambda t: Path(tmp) / "nope.zip"
    it.resolve_model_dir = lambda m: tmp
    os.environ["FAKE_MODE"] = mode
    it.START = time.time()
    real_argv = sys.argv
    sys.argv = ["image_trainer", "--task-id", "t1", "--model", "x/y", "--dataset-zip", os.path.join(tmp, "nope.zip"),
                "--model-type", "krea2", "--expected-repo-name", "r1", "--hours-to-complete", str(hours), "--future-flag", "1"]
    return tmp, real_argv


def run_main():
    try:
        it.main()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True)); print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False)); print(f"FAIL {name}: {type(e).__name__}: {e}")


def with_fake(mode, hours, fn):
    tmp, argv = setup(mode, hours)
    orig_popen = it.subprocess.Popen

    def popen(cmd, *a, **k):
        # swap /app/crown/train.py for the fake, keep every argument
        cmd = [sys.executable, FAKE] + cmd[2:]
        return orig_popen(cmd, *a, **k)
    it.subprocess.Popen = popen
    try:
        return fn(tmp, run_main())
    finally:
        it.subprocess.Popen = orig_popen
        sys.argv = argv


def publish_exit0():
    def body(tmp, rc):
        f = it.C.output_dir("t1", "r1") / "last.safetensors"
        assert f.exists() and f.read_bytes().startswith(b"trained"), "published file missing"
        assert rc == 0, f"exit code {rc} after a published artifact"
    with_fake("publish_then_crash", 0.5, body)


def timeout_kill():
    def body(tmp, rc):
        f = it.C.output_dir("t1", "r1") / "last.safetensors"
        assert f.exists() and f.read_bytes().startswith(b"partial-best"), "the hung trainer's artifact is not published"
        assert rc == 0, f"exit code {rc}"
    t0 = time.time()
    # hours 0.026 -> deadline 94 s; limit = max(30, 94-60) = 34 s: the kill must come at ~34 s, not at the deadline
    with_fake("hang", 0.026, body)
    assert time.time() - t0 < 60, "the hung trainer was not killed before the deadline"


def fallback_identity():
    def body(tmp, rc):
        f = it.C.output_dir("t1", "r1") / "last.safetensors"
        assert f.exists() and f.read_bytes().startswith(b"identity"), "identity fallback did not publish"
        assert rc == 0
    with_fake("nothing", 0.5, body)


def nothing_exit1():
    def body(tmp, rc):
        assert rc == 1, f"exit code {rc} with nothing produced"
    # identity fallback also produces nothing: patch the fake to exit on --identity-only too
    src = open(FAKE).read().replace('if "--identity-only" in sys.argv:\n    open', 'if "--identity-only" in sys.argv and os.environ["FAKE_MODE"] != "nothing_even_identity":\n    open')
    open(FAKE, "w").write(src)
    with_fake("nothing_even_identity", 0.5, body)


def recipes_parse():
    import types

    from crown import train

    class _Done:
        def wait(self, timeout=None):
            return 0

    captured = []
    orig_popen = it.subprocess.Popen
    it.subprocess.Popen = lambda cmd, *a, **k: captured.append(cmd) or _Done()
    extra = os.environ.pop("CROWN_EXTRA_ARGS", None)   # local experiments only: must not leak into this check
    try:
        for fam, recipe in it.RECIPES.items():
            captured.clear()
            it.run_trainer(types.SimpleNamespace(trigger_word=None), fam, "/m", "/d.zip", "/o", time.time() + 3600, [])
            cfg = train.parse(captured[0][2:])
            assert cfg.family == fam, f"{fam}: family {cfg.family}"
            assert cfg.exact_merge == ("--exact-merge" in recipe), f"{fam}: exact_merge {cfg.exact_merge}"
            assert cfg.fast_lora == ("--fast-lora" in recipe), f"{fam}: fast_lora {cfg.fast_lora}"
            assert cfg.ckpt == ("--no-ckpt" not in recipe), f"{fam}: ckpt {cfg.ckpt}"
            if "--screen-passes" in recipe:
                want = recipe[recipe.index("--screen-passes") + 1]
                assert cfg.screen_passes == want, f"{fam}: screen_passes {cfg.screen_passes!r}"
                sched = [float(x) for x in want.split(",")]
                assert sched == sorted(sched) and sched[0] > 0, f"{fam}: schedule not increasing and positive: {want}"
            print(f"  {fam}: exact_merge={cfg.exact_merge} fast_lora={cfg.fast_lora} ckpt={cfg.ckpt} "
                  f"screen_passes={getattr(cfg, 'screen_passes', '') or '-'}")
    finally:
        it.subprocess.Popen = orig_popen
        if extra is not None:
            os.environ["CROWN_EXTRA_ARGS"] = extra


def size_rule():
    """v12 KN / ZN / FN: every SIZE_RULES row (family, max given images, extra args) fires at <= max images and not
    above; families without a row, and an unreadable zip, keep the recipe as it is; the built argv parses in
    crown/train.py with the row's values winning over the recipe's (table-driven: it checks whatever rows ship)."""
    import types
    import zipfile

    from crown import train
    tmp = tempfile.mkdtemp(prefix="crown-size-")

    def zip_with(n, name):
        p = os.path.join(tmp, name)
        with zipfile.ZipFile(p, "w") as z:
            for i in range(n):
                z.writestr(f"{i:03d}.png", b"x")
                z.writestr(f"{i:03d}.txt", "a caption")
            z.writestr("__MACOSX/._000.png", b"x")
        return p

    assert it.count_images(zip_with(9, "nine.zip")) == 9, "image count wrong"
    assert it.size_overrides("krea2", os.path.join(tmp, "missing.zip")) == [], "missing zip overridden"
    ruled = {fam for fam, _, _ in it.SIZE_RULES}
    for fam in it.RECIPES:
        if fam not in ruled:
            assert it.size_overrides(fam, zip_with(5, f"{fam}_small.zip")) == [], f"{fam} overridden without a rule"

    class _Done:
        def wait(self, timeout=None):
            return 0
    captured = []
    orig_popen = it.subprocess.Popen
    it.subprocess.Popen = lambda cmd, *a, **k: captured.append(cmd) or _Done()
    extra_env = os.environ.pop("CROWN_EXTRA_ARGS", None)
    try:
        for fam, max_n, extra in it.SIZE_RULES:
            at, over = zip_with(max_n, f"{fam}_at.zip"), zip_with(max_n + 1, f"{fam}_over.zip")
            assert it.size_overrides(fam, at) == list(extra), f"{fam} at {max_n}: {it.size_overrides(fam, at)}"
            assert it.size_overrides(fam, over) == [], f"{fam} at {max_n + 1} overridden"
            captured.clear()
            it.run_trainer(types.SimpleNamespace(trigger_word=None), fam, "/m", at, "/o", time.time() + 3600,
                           it.size_overrides(fam, at))
            cfg = train.parse(captured[0][2:])
            for k in range(0, len(extra), 2):
                key = extra[k].lstrip("-").replace("-", "_")
                assert float(getattr(cfg, key)) == float(extra[k + 1]), f"{fam}: {key} {getattr(cfg, key)} != {extra[k + 1]}"
            print(f"  {fam}: <= {max_n} given images -> {' '.join(extra)} (parsed)")
    finally:
        it.subprocess.Popen = orig_popen
        if extra_env is not None:
            os.environ["CROWN_EXTRA_ARGS"] = extra_env

if __name__ == "__main__":
    for name, fn in [("recipes_parse", recipes_parse), ("publish_exit0", publish_exit0), ("timeout_kill", timeout_kill), ("fallback_identity", fallback_identity), ("nothing_exit1", nothing_exit1), ("size_rule", size_rule)]:
        check(name, fn)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} entrypoint checks passed")
    sys.exit(1 if bad else 0)
