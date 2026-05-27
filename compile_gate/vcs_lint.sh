#!/bin/bash
# vcs_lint.sh — wrapper that runs `vcs -sverilog -assert svaext` on a SV
# file inside the vcs_lint container with all the workarounds needed for
# VCS L-2016.06 to function on Ubuntu 18.04 / glibc 2.27.
#
# Usage:   vcs_lint.sh <file.sv>          # exit 0 = no errors
# Returns: 0 if VCS reports zero "Error-[..." lines, else 1.
#
# Setup needed before first use:
#   docker run -d --name vcs_lint \
#     --hostname lizhen --mac-address 02:42:ac:11:00:02 \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v ${REPO_ROOT}:/work \
#     --entrypoint /bin/bash phyzli/ubuntu18.04_xfce4_vnc4server_synopsys2016 \
#     -c "service ssh start; /usr/synopsys/11.9/amd64/bin/lmgrd -c \
#         /usr/local/flexlm/licenses/license.dat; sleep 5; tail -f /dev/null"
#   docker cp $(dirname $0)/vcs_shim.c vcs_lint:/tmp/vcs_shim.c
#   docker exec vcs_lint gcc -shared -fPIC -O2 -o /tmp/vcs_shim.so /tmp/vcs_shim.c -ldl

set -u
SRC="${1:-}"
[ -z "$SRC" ] && { echo "usage: $0 <file.sv>" >&2; exit 2; }
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 2; }

ABS=$(readlink -f "$SRC")
BASE=$(basename "$ABS")
WORK=/tmp/vcs_lint_$$
docker exec vcs_lint mkdir -p "$WORK" 2>/dev/null
docker cp "$ABS" "vcs_lint:$WORK/$BASE" >/dev/null

OUT=$(docker exec vcs_lint bash -c "
  cd $WORK
  export VCS_HOME=/usr/synopsys/vcs-L-2016.06
  export PATH=\$VCS_HOME/bin:\$PATH
  export LM_LICENSE_FILE=27000@lizhen
  export LD_PRELOAD=/tmp/vcs_shim.so
  ulimit -s unlimited
  setarch x86_64 -R vcs -sverilog -assert svaext '$BASE' -o sim 2>&1
")
docker exec vcs_lint rm -rf "$WORK" 2>/dev/null

if echo "$OUT" | grep -q '^Error-\['; then
  echo "$OUT" | grep -E '^Error-\[' | head -3
  exit 1
fi
exit 0
