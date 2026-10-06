"""Boot ComfyUI as a library, in training mode.

The evaluator imports the same ComfyUI revision with default CLI arguments. We
only change memory-management flags (which do not touch numerics) and the text
encoder dtype (fp16 on H100 is Comfy's own default for non-fp8 encoders).
"""

import os
import sys

_STATE = {}


def boot(comfy_root=None, text_enc_dtype="fp16"):
    if _STATE:
        return _STATE["mm"]
    root = os.path.abspath(comfy_root or os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.cli_args as cli

    a = cli.args
    a.disable_dynamic_vram = True
    a.highvram = True
    a.disable_smart_memory = True
    a.bf16_text_enc = text_enc_dtype == "bf16"
    a.fp16_text_enc = text_enc_dtype == "fp16"
    a.fp32_text_enc = text_enc_dtype == "fp32"
    a.fp8_e4m3fn_text_enc = False
    a.fp8_e5m2_text_enc = False

    import comfy.model_management as mm

    mm.in_training = True
    # NOT installed in training: crown.philox.install_evaluator_gpu (the validator's A100 - no fp8 compute, so an
    # fp8 flux base loads as bf16) was measured as a training setting by the pre-registered E3 arm (2026-09-26):
    # D's flags on the A100's own bf16 model vs D on fp8 storage, both scored A100-style: +0.081 % (t +1.97),
    # rejected by its rule (keep only if not worse than D by more than one SE). band_eval --eval-gpu a100 keeps
    # the emulation as the measuring instrument.
    _STATE["mm"] = mm
    return mm
