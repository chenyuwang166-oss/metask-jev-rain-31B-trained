#!/usr/bin/env bash
# Start the service in the foreground. MODEL must hold the base files and adapter/ (see README.md).
#   bash serve.sh --model /abs/model --state-dir /abs/state --run-id NAME [--port 8000] [--bind 127.0.0.1]
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_BATCH_INVARIANT=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "${PYTHON:-python3}" -B "$PKG/launch.py" "$@"
