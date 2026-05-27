#!/bin/bash
# vcs_dynamic_check.sh — invoked INSIDE the vcs_lint container. Takes an
# absolute path to a TB .sv file (under /work/...), compiles + simulates,
# and prints whatever the sim writes to stdout. The Python wrapper parses
# the AGREEMENT line.
set -u
SRC="${1:-}"
[ -z "$SRC" ] && { echo "usage: $0 <abs-path>" >&2; exit 2; }
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 2; }

WORK=$(mktemp -d /tmp/vcs_dyn_XXXXXX)
cd "$WORK" || exit 2

export VCS_HOME=/usr/synopsys/vcs-L-2016.06
export PATH=$VCS_HOME/bin:$PATH
export LM_LICENSE_FILE=27000@lizhen
export LD_PRELOAD=/tmp/vcs_shim.so
ulimit -s unlimited

# Compile (silently, capture errors)
COMPILE_OUT=$(setarch x86_64 -R vcs -sverilog -assert svaext "$SRC" -o "$WORK/sim" 2>&1)
if echo "$COMPILE_OUT" | grep -q '^Error-\['; then
  echo "COMPILE_FAIL"
  echo "$COMPILE_OUT" | grep -E '^Error-\[' | head -2
  cd /tmp; rm -rf "$WORK"; exit 1
fi

# Simulate with wall-clock timeout
timeout 10 "$WORK/sim" 2>&1
RC=$?

cd /tmp
rm -rf "$WORK" 2>/dev/null
exit "$RC"
