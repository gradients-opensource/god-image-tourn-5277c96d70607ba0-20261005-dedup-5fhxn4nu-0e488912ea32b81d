# The base image by digest (python:3.11-slim as pulled on 2026-09-30, created 2026-09-19): a moving tag can change the
# Debian release and the Python patch level between two cold builds of the same commit.
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

# Numerical stack pinned to the G.O.D image evaluator (ops/docker/validator-diffusion.dockerfile
# in rayonlabs/G.O.D): same ComfyUI revision, same torch build, same requirement pins. Training
# against any other stack would optimise a different function than the one we are scored on.
# EVERY Python distribution is pinned by ops/docker/constraints.txt (the `pip freeze --all` of the verified cold build):
# the validator rebuilds this image with nocache for every task and ComfyUI's requirements.txt leaves ~25 packages
# unpinned, so without the constraints an upstream release could change or break the build between two tasks.
#
# STEP ORDER. The validator cold-builds this image with the legacy builder (docker-py /build, no cache,
# no build args) under a 30-minute cap for every task. On docker 29's containerd image store every step
# after a large layer pays a fixed commit cost (measured 2026-09-26 on the build VPS: ~35 s per step,
# even for ARG/COPY, once the torch layer exists), so every cheap step (ARG, ENV, WORKDIR, ENTRYPOINT,
# COPY) comes BEFORE the large layers, the import check shares the requirements RUN, and the ~10 GB asset
# pull is the last step. The image content is unchanged.
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ARG FLUX_TE_REV=6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5
ARG BUILD_CAP_S=1800
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 \
    COMFY_ROOT=/opt/ComfyUI PYTHONPATH=/app CROWN_ASSETS=/opt/crown/assets
WORKDIR /app
ENTRYPOINT ["python", "/app/scripts/image_trainer.py"]

# build_t0: when the first RUN started - the asset step predicts the end of the build from it
RUN mkdir -p /opt/crown && date +%s > /opt/crown/build_t0 \
    && apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /app/checkpoints

RUN git init /opt/ComfyUI && cd /opt/ComfyUI \
    && git remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git fetch --depth 1 origin "${COMFYUI_COMMIT}" && git checkout FETCH_HEAD && rm -rf .git

COPY ops/docker/fetch_assets.sh /opt/crown/fetch_assets.sh
COPY ops/docker/constraints.txt /opt/crown/constraints.txt
COPY crown /app/crown
COPY scripts /app/scripts
COPY tools /app/tools

# Official PyPI is an explicit extra index: a cold build must not depend on one index's contents.
RUN pip install --timeout 120 --retries 3 --no-cache-dir -c /opt/crown/constraints.txt \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
RUN pip install --timeout 120 --retries 3 --no-cache-dir -c /opt/crown/constraints.txt -r /opt/ComfyUI/requirements.txt \
        diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 \
        pydantic==2.13.4 accelerate==1.6.0 pillow numpy \
    && python -c "import torch, diffusers, transformers; assert torch.__version__.startswith('2.9.1')" \
    && python -c "import comfy_kitchen"

# INTEGRITY DECLARATION. Five public, unmodified files are baked because they are the
# *evaluator's own inputs* (validator/evaluation/image_flow_adapter.py FAMILIES) and the task
# cache does not provide them in that form:
#   - qwen_image_vae.safetensors, flux2-vae.safetensors: the VAEs the evaluator encodes with
#     (the cache ships diffusers-layout VAEs, a different tensor than the one scored against);
#   - ae.safetensors: the FLUX VAE the evaluator encodes with (rayonlabs/FLUX.1-dev);
#   - clip_l.safetensors, t5xxl_fp16.safetensors: FLUX text encoders, which the validator's
#     downloader does not stage at all (it fetches only tokenizer configs). When the build clock
#     cannot fit t5xxl_fp16 (9.8 GB), the same repo revision's t5xxl_fp8_e4m3fn_scaled.safetensors
#     (5.2 GB) is baked in its place; the trainer dequantises it to fp16.
# The Ideogram-4 base the evaluator hard-codes is NOT baked: the evaluator twin rebuilds it from the
# task cache's copy of the same weights (crown/twin.py) and uses it only to score candidates.
# No dataset and no private artifact is bundled. Trained outputs derive solely from the
# validator-provided base model and dataset.
# The pulls run smallest-first and NEVER fail the build: before each file the script predicts the end of
# the build (elapsed since build_t0 + the download at the measured rate + the gzip commit of this layer at
# the measured rate) and skips the file if that passes the 30-minute cap minus a margin. A missing file
# degrades the trainer at run time instead.
RUN BUILD_CAP_S=${BUILD_CAP_S} FLUX_TE_REV=${FLUX_TE_REV} bash /opt/crown/fetch_assets.sh
