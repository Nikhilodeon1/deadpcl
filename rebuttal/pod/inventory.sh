#!/usr/bin/env bash
# Read-only inventory of the pod: hardware, disks, dataset directories,
# existing checkpoints / result files / prediction dumps. Writes one text file.
# It does not read dataset contents, only lists names and sizes.
set -u
set -f   # no globbing: the find patterns below contain literal *s
. "$(dirname "${BASH_SOURCE[0]}")/env.sh" 2>/dev/null
OUT="${1:-$RB_OUT/pod_inventory.txt}"
SCAN_ROOTS="${SCAN_ROOTS:-$HOME /tmp /workspace /data /mnt /scratch}"
PRUNE='-path /proc -prune -o -path /sys -prune -o -path */node_modules -prune -o -path */.cache -prune -o -path /tmp/rb_venv -prune -o -path /tmp/pip_cache -prune'

{
echo "== date (UTC)"; date -u
echo "== cpu cores"; nproc
echo "== memory (GB)"; free -g
echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -5
echo "== python"; python3 --version 2>&1
echo "== disk"; df -h "$HOME" /tmp 2>&1
echo "== dataset env vars"
for v in PHYSIONET_DIR MIMIC_DIR EICU_DIR; do
  d="${!v:-}"
  if [ -n "$d" ] && [ -d "$d" ]; then echo "$v set, exists: $(du -sh "$d" 2>/dev/null | cut -f1)"; ls "$d" | head -20
  else echo "$v: unset or missing"; fi
done
echo "== candidate dataset directories (names only)"
for r in $SCAN_ROOTS; do [ -d "$r" ] && timeout 60 find "$r" -maxdepth 4 -type d \( -iname 'physionet*' -o -iname 'training_set*' -o -iname 'mimic*' -o -iname 'eicu*' \) 2>/dev/null; done | sort -u | head -60
echo "== checkpoint files (*.pt, *.pth): count and total size per directory"
for r in $SCAN_ROOTS; do [ -d "$r" ] && timeout 120 find "$r" $PRUNE -o -type f \( -name '*.pt' -o -name '*.pth' \) -printf '%h %s\n' 2>/dev/null; done \
  | awk '{n[$1]++; s[$1]+=$2} END {for (d in n) printf "%6d files %9.1f MB  %s\n", n[d], s[d]/1048576, d}' | sort -k5 | head -80
echo "== checkpoint file names (first 150)"
for r in $SCAN_ROOTS; do [ -d "$r" ] && timeout 120 find "$r" $PRUNE -o -type f \( -name '*.pt' -o -name '*.pth' \) -printf '%p  %s B  %TY-%Tm-%Td\n' 2>/dev/null; done | sort | head -150
echo "== per-sample prediction dumps (*.npz): count per directory"
for r in $SCAN_ROOTS; do [ -d "$r" ] && timeout 120 find "$r" $PRUNE -o -type f -name '*.npz' -printf '%h\n' 2>/dev/null; done | sort | uniq -c | head -40
echo "== result json/csv files (first 150)"
for r in $SCAN_ROOTS; do [ -d "$r" ] && timeout 120 find "$r" $PRUNE -o -type f \( -iname '*result*.json' -o -iname '*lambda*' -o -iname '*seed*' -o -iname '*sweep*' \) -printf '%p  %s B  %TY-%Tm-%Td\n' 2>/dev/null; done | sort | head -150
echo "== git clones present (names only)"
for r in $SCAN_ROOTS; do [ -d "$r" ] && timeout 60 find "$r" -maxdepth 4 -name .git -type d -printf '%h\n' 2>/dev/null; done | sort -u | head -30
} 2>&1 | sed "s#$HOME#~#g" > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
