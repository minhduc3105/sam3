#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv-r2s-sam3
.venv-r2s-sam3/bin/python -m pip install --upgrade pip
# The pinned SAM3 revision imports pkg_resources at runtime; setuptools 82+ removed it.
.venv-r2s-sam3/bin/python -m pip install 'setuptools<82'
.venv-r2s-sam3/bin/python -m pip install torch torchvision
# Install a CUDA-compatible torch build for the server first if needed.
.venv-r2s-sam3/bin/python -m pip install 'git+https://github.com/facebookresearch/sam3.git@f1647e8d0c1d280991f4d1b7d974c2704537fe66' pillow einops pycocotools psutil 'numpy>=1.26,<2'
.venv-r2s-sam3/bin/python -c 'from sam3 import build_sam3_image_model; import r2s_sam3; print("SAM3 environment ready")'
