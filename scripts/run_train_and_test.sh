#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sam3
source /venv/main/bin/activate
resume=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume=(--resume)
fi
for split in train test; do
  python -m r2s_sam3 run \
    --dataset-root /workspace/r2s100k \
    --split "$split" \
    --checkpoint /workspace/sam3/checkpoints/sam3.pt \
    --output "/workspace/sam3/runs/sam3_${split}" \
    "${resume[@]}"
done
