#!/usr/bin/env bash
# Source this file (". rebuttal/pod/env.sh") at the start of every shell on the pod.
#
# Layout assumed:
#   - the repository is cloned into persistent storage (home directory); small
#     JSON outputs also live there
#   - the Python environment, pip cache, dataset caches and checkpoints live in
#     /tmp, which is large but is wiped when the pod is replaced
# Nothing here depends on where the repository was cloned.

_rb_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RB_REPO_ROOT="$(cd "$_rb_here/../.." && pwd)"

export RB_VENV="${RB_VENV:-/tmp/rb_venv}"
export RB_WORK="${RB_WORK:-/tmp/rb_work}"                # ephemeral: caches, checkpoints
export RB_OUT="${RB_OUT:-$RB_REPO_ROOT/rebuttal/outputs}" # persistent: small aggregates only
mkdir -p "$RB_WORK" "$RB_OUT"

# Paths read by config.py (all overridable there via environment variables).
export RESULTS_DIR="${RESULTS_DIR:-$RB_WORK/results}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RB_WORK/checkpoints}"
export CACHE_DIR="${CACHE_DIR:-$RB_WORK/cache}"
export PCL_TEST_MODE=0
export NUM_WORKERS="${NUM_WORKERS:-8}"

# Keep big temporary files off the small persistent disk.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1

# Dataset locations: one line each in ~/.rb_paths (outside the repository), e.g.
#   export PHYSIONET_DIR=/path/to/physionet2019   # contains training_setA, training_setB
#   export MIMIC_DIR=/path/to/mimiciv/3.1
#   export EICU_DIR=/path/to/eicu-crd/2.0
[ -f "$HOME/.rb_paths" ] && . "$HOME/.rb_paths"

[ -f "$RB_VENV/bin/activate" ] && . "$RB_VENV/bin/activate"

for _v in PHYSIONET_DIR MIMIC_DIR EICU_DIR; do
  [ -z "${!_v:-}" ] && echo "[env] note: $_v is not set (see ~/.rb_paths)" >&2
done
echo "[env] repo=$RB_REPO_ROOT venv=$RB_VENV work=$RB_WORK out=$RB_OUT" >&2
