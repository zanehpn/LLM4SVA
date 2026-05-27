#!/bin/bash
# vcs_check.sh — invoked INSIDE the vcs_lint container. Takes an absolute
# path to a .sv file (under /work/...), runs `vcs -sverilog -assert svaext`
# in a private scratch dir, and exits 0 iff no `^Error-[` lines were
# emitted. All build artifacts are written to /tmp so the bind-mounted
# /work source tree stays clean.
#
# Used by experiments/scripts/vcs_compile_check.py (the Python wrapper that
# mirrors verilator_compile_check).
set -u
SRC="${1:-}"
[ -z "$SRC" ] && { echo "usage: $0 <abs-path>" >&2; exit 2; }
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 2; }

WORK=$(mktemp -d /tmp/vcs_chk_XXXXXX)
cd "$WORK" || exit 2

export VCS_HOME=/usr/synopsys/vcs-L-2016.06
export PATH=$VCS_HOME/bin:$PATH
export LM_LICENSE_FILE=27000@lizhen
export LD_PRELOAD=/tmp/vcs_shim.so
ulimit -s unlimited

OUT=$(setarch x86_64 -R vcs -sverilog -assert svaext "$SRC" -o "$WORK/sim" 2>&1)
RC_HAS_ERROR=0
if echo "$OUT" | grep -q '^Error-\['; then
  RC_HAS_ERROR=1
  echo "$OUT" | grep -E '^Error-\[' | head -2
fi

cd /tmp
rm -rf "$WORK" 2>/dev/null
exit "$RC_HAS_ERROR"
