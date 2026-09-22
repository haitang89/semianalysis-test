#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
NAME="${BENCH_CONTAINER:-bench}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

build_args() {
  local digest sha dirty=0
  digest="$(docker inspect -f '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null | sed 's/.*@//' || true)"
  sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]; then dirty=1; fi
  args=(
    --name "$NAME" --gpus all --network host --ipc host
    -v "$ROOT:/work" -v "$HF_CACHE:/root/.cache/huggingface"
    -w /work -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1
    -e "BENCH_IMAGE=$IMAGE" -e "BENCH_IMAGE_DIGEST=$digest"
    -e "BENCH_GIT_SHA=$sha" -e "BENCH_GIT_DIRTY=$dirty"
    --entrypoint sleep "$IMAGE" infinity
  )
}

case "${1:-}" in
  start)
    build_args
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d "${args[@]}" >/dev/null
    echo "started $NAME from $IMAGE"
    ;;
  exec)
    shift
    docker exec "$NAME" "$@"
    ;;
  stop)
    docker rm -f "$NAME"
    ;;
  cmd)
    build_args
    printf 'docker run -d'
    printf ' %q' "${args[@]}"
    printf '\n'
    ;;
  *)
    echo "usage: $0 start | exec <command...> | stop | cmd" >&2
    exit 64
    ;;
esac
