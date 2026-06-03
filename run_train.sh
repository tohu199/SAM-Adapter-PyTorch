#!/usr/bin/env bash
# Single-GPU training (README uses multi-GPU torch.distributed.run).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
  --config "${CONFIG:-configs/cod-sam-vit-l.yaml}" \
  "$@"
