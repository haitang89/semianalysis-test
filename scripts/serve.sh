#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
NAME="${SERVER_CONTAINER:-vllm}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TRACES="${TRACES_DIR:-$HOME/traces}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
PREFIX_CACHING="${PREFIX_CACHING:-on}"

case "$MODE" in
  eager) extra=(--enforce-eager) ;;
  compiled) extra=() ;;
  stop) docker rm -f "$NAME"; exit 0 ;;
  *) echo "usage: $0 eager | compiled | stop" >&2; exit 64 ;;
esac

if [ "$PREFIX_CACHING" = "off" ]; then extra+=(--no-enable-prefix-caching); fi
mkdir -p "$TRACES"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --network host --ipc host \
  -v "$HF_CACHE:/root/.cache/huggingface" -v "$TRACES:/traces" \
  "$IMAGE" "$MODEL" --language-model-only --max-model-len "$MAX_LEN" "${extra[@]}" \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/traces", "torch_profiler_record_shapes": true}' >/dev/null
echo "started $NAME in $MODE mode, prefix caching $PREFIX_CACHING, traces in $TRACES"
echo "wait for it with: until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done"
