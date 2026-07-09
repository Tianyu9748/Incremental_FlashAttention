#!/usr/bin/env bash
# check_sm80_build.sh — compile-only Ampere (sm_80) portability check for PRR's repair path.
#
# We only have Hopper (sm90) hardware, so this is a *compile-level* check, not a runtime
# measurement. It demonstrates that the two architecture-generic pieces of PRR's kernel
# stack — the online-softmax repair/merge kernel and FlashAttention-3's dense attention
# mainloop — build for Ampere, and it documents that the sparse block-table gather is
# currently instantiated only for Hopper (sm90).
#
# Usage:
#   source /home/wangtian/venv/fp8_llm/bin/activate
#   bash rebuttal/q1_portability/check_sm80_build.sh
#
# Override the FlashAttention-3 tree location with FA3_DIR if needed.

set -u

# Defaults to the repo this script lives in; override with FA3_DIR if needed.
FA3_DIR="${FA3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HOP="$FA3_DIR/hopper"
CUT="$FA3_DIR/csrc/cutlass"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJ_DIR="$(mktemp -d)"
RESULTS="$OUT_DIR/results.txt"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found on PATH" >&2; exit 2
fi
TORCH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')" || {
  echo "could not locate torch (activate the venv first)" >&2; exit 2; }
PYINC="$(python -c 'import sysconfig;print(sysconfig.get_path("include"))')"

INC=(-I"$HOP" -I"$CUT/include" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYINC")
COMMON=(-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
        -DNDEBUG -DCUTLASS_DEBUG_TRACE_LEVEL=0)

compile() {  # name  src  -> echoes PASS/FAIL and target arch
  local name="$1" src="$2" obj="$OBJ_DIR/$1.o" log="$OBJ_DIR/$1.log"
  printf '>>> [%s] %s  @ -arch=sm_80\n' "$name" "$(basename "$src")"
  if nvcc -c -arch=sm_80 "${COMMON[@]}" "${INC[@]}" "$src" -o "$obj" 2>"$log"; then
    local arch; arch="$(cuobjdump "$obj" 2>/dev/null | grep -m1 -i 'arch =' | tr -d ' ')"
    printf '    result: BUILDS for sm_80  (%s)\n' "${arch:-arch=?}"
    printf '%-28s BUILDS  %s\n' "$name" "${arch:-arch=?}" >>"$RESULTS"
  else
    printf '    result: FAILED  (see %s)\n' "$log"
    printf '%-28s FAILED\n' "$name" >>"$RESULTS"
    tail -8 "$log" | sed 's/^/      /'
  fi
}

: >"$RESULTS"
{
  echo "# PRR Ampere (sm_80) compile-only portability check"
  echo "# nvcc: $(nvcc --version | tail -1)"
  echo "# date: $(date -Is)"
  echo
} >>"$RESULTS"

echo "== Architecture-generic PRR pieces (expected: BUILD for Ampere) =="
compile repair_combine_online_softmax "$HOP/flash_fwd_combine.cu"
compile dense_attention_sm80         "$HOP/instantiations/flash_fwd_hdim128_bf16_sm80.cu"

echo
echo "== Hopper sparse gather instantiation coverage (informational) =="
n_sm80=$(ls "$HOP"/instantiations/flash_fwd_sparse*_sm80.cu 2>/dev/null | wc -l)
n_sm90=$(ls "$HOP"/instantiations/flash_fwd_sparse*_sm90.cu 2>/dev/null | wc -l)
echo "    sparse block-table sources:  sm80=$n_sm80   sm90=$n_sm90"
echo "sparse_sources sm80=$n_sm80 sm90=$n_sm90" >>"$RESULTS"
# The sm90 sparse source *compiles* under -arch=sm_80 (its Hopper tensor-core intrinsics
# are behind cutlass arch dispatch), but the build ships it only as sm90a because its fast
# path targets Hopper TMA/WGMMA. A functional Ampere sparse kernel = re-instantiating the
# gather on the existing sm80 mainloop (cp.async), not a redesign.
compile sparse_gather_sm90_under_sm80 "$HOP/instantiations/flash_fwd_sparse_hdim128_blockn64_bf16_sm90.cu"

echo
echo "Summary written to $RESULTS"
cat "$RESULTS"
rm -rf "$OBJ_DIR"
