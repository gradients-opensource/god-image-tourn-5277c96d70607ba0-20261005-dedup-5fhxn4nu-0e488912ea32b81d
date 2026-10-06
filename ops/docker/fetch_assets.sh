#!/usr/bin/env bash
# Build-time asset pulls that can never cost the build.
#
# The validator cold-builds this image with no cache under a 30-minute cap for EVERY task (G.O.D
# DOCKER_BUILD_TIMEOUT_MINUTES), and on docker 29's containerd image store every layer is gzip-compressed
# when its step is committed. The asset layer's commit is the largest single cost of the whole build
# (measured 2026-09-26 on the build VPS: 10.3 GB committed in ~17 min = 0.83 x that machine's single-core
# `gzip -6` rate on the same kind of data). So before each file this script predicts when the build would
# end with it:
#     elapsed since the build's first RUN (/opt/crown/build_t0)
#   + its download at the rate measured on the files before it
#   + the commit of every byte of this layer at COMMIT_FACTOR x the gzip rate measured on a landed asset
# and skips the file when that prediction passes the cap minus LEAD_S (the validator's clock starts before
# our first RUN: FROM pull, context upload, ARG/ENV steps) and SAFETY_S (image export after the last commit,
# error in the predictions). Each download is also cut at the latest moment that still leaves its commit
# inside the cap. Files go smallest first; a skipped or failed file is simply absent and the trainer degrades
# at run time (no flux text encoders -> the task repo's own; no flux VAE -> the repo's diffusers VAE).
# Only when the fp16 T5 ends up absent is the scaled fp8 T5 tried, under the same rule; the log line "T5 BAKED:"
# names what the image carries.
# The ideogram4 evaluator base is not pulled: the twin rebuilds it from the task cache
# (crown.twin.rebuild_ideogram4_evaluator_base). Exit status is always 0.
set -u
DEST=${ASSET_DIR:-/opt/crown/assets}
REV=${FLUX_TE_REV:-6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5}
CAP_S=${BUILD_CAP_S:-1800}
LEAD_S=60
SAFETY_S=120
COMMIT_FACTOR=0.8            # containerd's commit rate / `gzip -6` rate, 0.83 measured, rounded down
PROBE_BYTES=64000000         # gzip-rate probe on the first landed asset
mkdir -p "$DEST"
now() { date +%s; }
t0=$(cat /opt/crown/build_t0 2>/dev/null || now)
deadline=$(( t0 + CAP_S - LEAD_S - SAFETY_S ))
layer=0                      # bytes of assets already in this layer
dl_bytes=0; dl_ms=0          # download rate measurement
commit_bps=0                 # predicted commit rate (bytes/s), 0 until measured
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
echo "asset step at t=$(( $(now) - t0 ))s since the first RUN; deadline t=$(( deadline - t0 ))s (cap ${CAP_S}s - ${LEAD_S}s - ${SAFETY_S}s)"

remote_size() {  # the final Content-Length after redirects, empty if unknown
  curl -sSIL --max-time 30 "$1" 2>/dev/null | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{if (n != "") print n}'
}

probe_gzip() {  # single-core gzip -6 rate on the first PROBE_BYTES of a landed file -> commit_bps
  local f=$1 s e ns
  s=$(date +%s%N)
  head -c "$PROBE_BYTES" "$f" | gzip -6 > /dev/null
  e=$(date +%s%N)
  ns=$(( e - s ))
  [ "$ns" -gt 0 ] || return 0
  commit_bps=$(awk -v b="$PROBE_BYTES" -v ns="$ns" -v k="$COMMIT_FACTOR" 'BEGIN{printf "%d", k * b / (ns / 1e9)}')
  echo "gzip -6 probe: $(( PROBE_BYTES * 1000 / (ns / 1000000) / 1000000 )) MB/s -> predicted commit $(( commit_bps / 1000000 )) MB/s"
}

fetch() {  # name url
  local name=$1 url=$2 size t tm dl commit end left rc
  size=$(remote_size "$url")
  if [ -z "$size" ]; then echo "SKIPPED $name: size unknown (HEAD failed); trainer will degrade for this asset"; return 0; fi
  t=$(now); tm=$(now_ms)
  commit=0
  [ "$commit_bps" -gt 0 ] && commit=$(( (layer + size) / commit_bps ))
  dl=0
  [ "$dl_bytes" -gt 0 ] && dl=$(( size * dl_ms / dl_bytes / 1000 ))
  end=$(( t + dl + commit ))
  if [ "$end" -gt "$deadline" ]; then
    echo "SKIPPED $name ($(( size / 1000000 )) MB): predicted end t=$(( end - t0 ))s > deadline t=$(( deadline - t0 ))s (download ${dl}s + commit ${commit}s); trainer will degrade for this asset"
    return 0
  fi
  left=$(( deadline - t - commit ))
  if [ "$left" -le 10 ]; then echo "SKIPPED $name: no time left for its download"; return 0; fi
  if timeout "$left" curl -fsSL --retry 3 --retry-delay 2 -o "$DEST/$name.part" "$url"; then
    mv -f "$DEST/$name.part" "$DEST/$name"
    dl_bytes=$(( dl_bytes + size )); dl_ms=$(( dl_ms + $(now_ms) - tm ))
    layer=$(( layer + size ))
    [ "$commit_bps" -eq 0 ] && probe_gzip "$DEST/$name"
    echo "fetched $name ($(( size / 1000000 )) MB) at t=$(( $(now) - t0 ))s"
  else
    rc=$?
    rm -f "$DEST/$name.part"
    echo "MISSING $name (exit $rc) at t=$(( $(now) - t0 ))s; trainer will degrade for this asset"
  fi
  return 0
}
fetch clip_l.safetensors               "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${REV}/clip_l.safetensors"
fetch qwen_image_vae.safetensors       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"
fetch flux2-vae.safetensors            "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors"
fetch ae.safetensors                   "https://huggingface.co/rayonlabs/FLUX.1-dev/resolve/main/ae.safetensors"
fetch t5xxl_fp16.safetensors           "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${REV}/t5xxl_fp16.safetensors"
# FALLBACK only when the fp16 T5 (the evaluator's file, 9.8 GB) is absent - skipped by the prediction or failed: the same
# revision's scaled fp8 T5 (5.2 GB) under the same prediction rule. The trainer dequantises it to fp16 (crown/assets.py).
if [ ! -f "$DEST/t5xxl_fp16.safetensors" ]; then
  echo "FALLBACK: t5xxl_fp16.safetensors is not baked; trying t5xxl_fp8_e4m3fn_scaled.safetensors under the same prediction"
  fetch t5xxl_fp8_e4m3fn_scaled.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${REV}/t5xxl_fp8_e4m3fn_scaled.safetensors"
fi
if [ -f "$DEST/t5xxl_fp16.safetensors" ]; then
  echo "T5 BAKED: fp16 (t5xxl_fp16.safetensors, the evaluator's file)"
elif [ -f "$DEST/t5xxl_fp8_e4m3fn_scaled.safetensors" ]; then
  echo "T5 BAKED: FALLBACK fp8 (t5xxl_fp8_e4m3fn_scaled.safetensors; the trainer dequantises it to fp16)"
else
  echo "T5 BAKED: NONE (flux falls back to the task repo's own T5, else the checkpoint's embedded T5, else the identity LoRA)"
fi
echo "assets present (predicted commit of this layer: $(( commit_bps > 0 ? layer / commit_bps : 0 ))s):"
ls -la --block-size=M "$DEST" | tail -n +2 | awk '{print $5, $9}'
exit 0
