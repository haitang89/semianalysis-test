#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs results/env "$HF_CACHE"

echo "== gpu"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

echo "== docker"
docker --version
docker pull "$IMAGE"
docker inspect -f '{{index .RepoDigests 0}}' "$IMAGE" | tee results/env/image_digest.txt
docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" -L

echo "== model download in the background: $MODEL"
nohup docker run --rm --network host \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  --entrypoint hf "$IMAGE" download "$MODEL" \
  > logs/model_download.log 2>&1 &
echo "follow it with: tail -f logs/model_download.log"
