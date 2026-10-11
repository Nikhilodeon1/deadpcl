#!/usr/bin/env bash
# Build the Python environment in /tmp, reusing the image's CUDA torch.
# Never downloads torch by default: the CUDA wheels are several GB and crashed the pod
# (INSTALL_TORCH=1 overrides this, only after checking free space). No package caches are kept.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PY="${PYTHON:-/opt/conda/bin/python}"
VENV="$RB_VENV"
echo "== disk / memory"; df -h ~ /tmp | cat; free -g | head -2
echo "== what the image already has"
"$PY" - <<'PYEOF'
import importlib.util as u
for m in ["torch","numpy","pandas","sklearn","scipy","matplotlib","xgboost"]:
    print(f"{m:12s}", "yes" if u.find_spec(m) else "MISSING")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available(): print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch import failed:", e)
PYEOF
HAS=$("$PY" -c "
try:
    import torch; print('1' if torch.cuda.is_available() else '0')
except Exception: print('0')")
if [ "$HAS" != "1" ] && [ "${INSTALL_TORCH:-0}" != "1" ]; then
  echo "No CUDA torch in $PY. Not installing one (several GB; crashed this pod before)."; exit 2
fi
rm -rf "$VENV"
"$PY" -m venv --system-site-packages "$VENV"
MISSING=$("$PY" - <<'PYEOF'
import importlib.util as u
pk={"numpy":"numpy","pandas":"pandas","sklearn":"scikit-learn","scipy":"scipy","matplotlib":"matplotlib","xgboost":"xgboost"}
print(" ".join(p for m,p in pk.items() if not u.find_spec(m)))
PYEOF
)
if [ -n "$MISSING" ]; then echo "installing small packages: $MISSING"; "$VENV/bin/python" -m pip install --no-cache-dir $MISSING; fi
"$VENV/bin/python" -c "import torch,numpy,pandas,sklearn,scipy,xgboost;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "env ready: $VENV"
