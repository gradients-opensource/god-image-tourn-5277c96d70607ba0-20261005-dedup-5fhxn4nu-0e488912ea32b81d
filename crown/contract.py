"""The validator contract, as read from G.O.D (2026-09-23).

Every constant here is dictated by the validator or the evaluator, not chosen
by us. Sources are cited inline. Change nothing without re-reading the source.
"""

from pathlib import Path

# --- container paths (trainer/constants.py, docs/miner.md) ---------------------
CACHE_ROOT = Path("/cache")                      # mounted read-only
CACHE_MODELS = CACHE_ROOT / "models"             # {model_id with '/' -> '--'}
CACHE_DATASETS = CACHE_ROOT / "datasets"         # {task_id}_tourn.zip
CACHE_HF = CACHE_ROOT / "hf_cache"               # text encoders for krea2/ideogram4
CHECKPOINTS_ROOT = Path("/app/checkpoints")      # the only writable output path
WORK_ROOT = Path("/tmp/crown")                   # scratch inside the container

# --- families (validator/evaluation/image_flow_adapter.py FAMILIES) -------------
FAMILIES = ("flux", "z-image", "qwen-image", "ideogram4", "krea2")

# Evaluator text-encoder / VAE identities (image_flow_adapter.py). We bake the
# two VAEs the cache does not provide in evaluator format; text encoders come
# from the cache (see assets.py for the documented parity gap).
BAKED_DIR = Path("/opt/crown/assets")
BAKED_VAE = {
    "krea2": "qwen_image_vae.safetensors",       # Comfy-Org/Krea-2/vae/
    "qwen-image": "qwen_image_vae.safetensors",  # Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/ (same bytes)
    "ideogram4": "flux2-vae.safetensors",        # Comfy-Org/Ideogram-4/vae/
}
# FLUX: the validator's downloader (trainer/containers/downloader.py, G.O.D fb8cff2) never stages the evaluator's
# text encoders, only tokenizer configs. What the task cache holds is the repo's own: a full snapshot for
# diffusers-style repos (rayonlabs/FLUX.1-dev and PixelWave: text_encoder/ CLIP-L + text_encoder_2/ T5, bf16), the
# single root file for standalone repos (FLUX-MonochromeManga: CLIP-L + plain fp8 T5 embedded; fluxunchained-dev:
# none). The evaluator encodes prompts with comfyanonymous/flux_text_encoders {t5xxl_fp16, clip_l}; we bake those two.
BAKED_FLUX_TE = ("clip_l.safetensors", "t5xxl_fp16.safetensors")
# Build-clock fallback for the 9.8 GB fp16 T5 (ops/docker/fetch_assets.sh bakes it only when t5xxl_fp16 is absent): the
# same repo revision's scaled fp8 T5, 5.2 GB. Its 168 Linear weights are fp8 e4m3 x a per-tensor scale_weight; its norms,
# embeddings and relative-attention bias are the fp16 file's own tensors. assets.dequantize_t5_fp8 turns it back into
# the fp16 file's layout (same 220 keys and shapes), so Comfy loads it exactly like the evaluator's file.
BAKED_FLUX_T5_FP8 = "t5xxl_fp8_e4m3fn_scaled.safetensors"
FLUX_VAE_FILE = "ae.safetensors"                 # root of the model snapshot == evaluator's rayonlabs/FLUX.1-dev/ae.safetensors
# ideogram4: the evaluator ignores the task model_id and loads this file (image_artifacts.prepare_base).
# The cache ships a differently-quantised copy (per-row scales); this one is per-tensor comfy_quant.
# Baked so the twin scorer can load the model exactly as the evaluator does.
BAKED_IDEOGRAM4_BASE = "ideogram4_fp8_scaled.safetensors"   # Comfy-Org/Ideogram-4/diffusion_models/
CLIP_TYPES = {"krea2": "KREA2", "ideogram4": "IDEOGRAM4", "qwen-image": "QWEN_IMAGE", "z-image": "LUMINA2", "flux": "FLUX"}
FAMILY_GUIDANCE = {"flux": 3.5}                  # FlowFamily.guidance
FAMILY_SHIFT = {"qwen-image": 3.0}               # FlowFamily.shift (ModelSamplingAuraFlow)

# --- scoring (validator/evaluation/denoising_mse.py, evaluators/diffusion.py) ----
EVAL_STRATA = 16
EVAL_NOISES = 16                                 # DEFAULT_NOISES; no production override exists
EVAL_MASTER_SEED = 42
EVAL_TEXT_WEIGHT = 0.5                           # DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT
EVAL_BATCH = 2                                   # IMAGE_EVAL_BATCH_SIZE default
EVAL_CFG = 1.0
EVAL_SAMPLER_SEED = 42
# The evaluator's GPU (G.O.D validator/evaluation: run_evaluation_basilica_image -> gpu_models = BASILICA_GPU_MODELS
# = ["A100"], 80 GB; the runpod backend asks for "A100" too). Training runs on an H100 (gpu_requirements.py:
# IMAGETASK -> H100_1X). Both facts change numbers the evaluator computes: torch.randint's kernel geometry (the
# stochastic-rounding noise of every patched fp8 weight) and supports_fp8_compute (sm80: comfy_quant layers are
# emulated - dequantised weights, bf16 activations). crown.philox.install_evaluator_gpu reproduces both.
EVAL_GPU = {"name": "A100", "sm_count": 108, "threads_per_sm": 2048, "fp8_compute": False}

# --- dataset (validator/tasks/datasets/preparation.py, core/constants/datasets.py)
TEST_SPLIT_FRACTION = 0.10                       # ceil(N * 0.10) held out, disjoint, unseeded shuffle
MIN_IMAGE_TEXT_PAIRS = 10
MAX_IMAGE_TEXT_PAIRS = 50
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# --- output (docs/miner.md, image_artifacts.select_lora) ----------------------
OUTPUT_LORA_NAME = "last.safetensors"            # exactly one -> selected unambiguously
OUTPUT_SUBDIR = "checkpoints"                    # HF_REPO_SUBFOLDER the uploader uses

# --- timing --------------------------------------------------------------------
# The validator waits `hours_to_complete` then kills the container. Everything
# after the kill is lost, so the run must be *finished and written* before it.
KILL_MARGIN_S = 180.0                            # never train into the last 3 minutes
PUBLISH_RESERVE_S = 90.0                         # time to save + copy the final artifact


def cached_model_dir(model_id: str) -> Path:
    return CACHE_MODELS / model_id.replace("/", "--")


def dataset_zip(task_id: str) -> Path:
    return CACHE_DATASETS / f"{task_id}_tourn.zip"


def output_dir(task_id: str, expected_repo_name: str) -> Path:
    return CHECKPOINTS_ROOT / task_id / expected_repo_name
