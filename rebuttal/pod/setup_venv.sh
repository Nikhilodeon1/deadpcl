#!/usr/bin/env bash
# Build the Python environment in /tmp (not in the small persistent home disk).
# Re-run after every pod replacement; takes a few minutes.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

PY="${PYTHON:-python3}"
# V100 is compute capability 7.0. CUDA 12.6 wheels of torch 2.6-2.8 still ship it.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.6,<2.9}"
PKGS="numpy pandas scikit-learn scipy matplotlib xgboost"

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "python3 is older than 3.10 ($("$PY" --version 2>&1))." >&2
  echo "Install a newer one with uv (pip install uv; uv python install 3.11) and rerun with PYTHON=<path>." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv "$RB_VENV" --python "$PY"
  uv pip install --python "$RB_VENV/bin/python" "$TORCH_SPEC" --index-url "$TORCH_INDEX"
  uv pip install --python "$RB_VENV/bin/python" $PKGS
else
  "$PY" -m venv "$RB_VENV"
  "$RB_VENV/bin/python" -m pip install --upgrade pip
  "$RB_VENV/bin/python" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
  "$RB_VENV/bin/python" -m pip install $PKGS
fi

"$RB_VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda"); print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
else:
    print("WARNING: no GPU visible to torch. If the driver is old, try TORCH_INDEX=https://download.pytorch.org/whl/cu121")
PY
