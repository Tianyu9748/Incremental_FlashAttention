# Incremental Flash Attention for Speculative Sparse Attention

## Problem Statement

During LLM inference (decoding), a predictor speculatively selects the most likely important K/V blocks for sparse attention based on historical block importance scores. This avoids attending to all O(n) blocks.

The predictor occasionally misses important blocks. The current fallback re-runs full flash attention from scratch, discarding all computation from the speculative pass.

**Goal**: When missed blocks are identified, run FA only on those missed blocks and merge the result with the already-computed speculative output using the online softmax merge identity — no wasted recomputation.

---

## Assumption

The caller provides the missed K/V block indices as a **contiguous int array** `block_table[num_missed_blocks]` where each entry is an absolute block index into the K/V sequence. This is available before the correction pass is launched.

---

## Implementation Plan

The approach is **two-pass + combine**, reusing the existing `FlashAttnFwdCombine` kernel.

### Phase 1 — Speculative Pass (already exists, no changes)

Run FA normally over the speculatively selected blocks. This already produces `O_spec` and `LSE_spec`.

### Phase 2 — Missed-Block Pass (new)

Run FA over only the missed blocks by replacing the linear N-loop with indexed iteration over `block_table`.

**Step 1: Add block table fields to the params structs**

In `csrc/flash_attn/src/flash.h` (FA-2) and `hopper/flash.h` (FA-3), add to the forward params:

```cpp
int* block_table;       // pointer to contiguous array of KV block indices; null = use linear range
int  num_blocks;        // length of block_table
```

**Step 2: Modify the N-loop to use the block table**

In FA-2 (`csrc/flash_attn/src/flash_fwd_kernel.h`, function `compute_attn_1rowblock`), the loop:

```cpp
for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
```

becomes:

```cpp
for (int i = 0; i < (params.block_table ? params.num_blocks : n_block_max - n_block_min); ++i) {
    int n_block = params.block_table ? params.block_table[i] : n_block_max - 1 - i;
```

Apply the same change in FA-3 (`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`) inside the consumer warp-group's mainloop.

**Step 3: Expose in the Python API**

Add `block_table` and `num_blocks` (or derive `num_blocks` from tensor length) as optional arguments to `flash_attn_func` in `hopper/flash_attn_interface.py` and the corresponding `flash_api.cpp` binding.

### Phase 3 — Combine (already exists, no changes)

Call `FlashAttnFwdCombine` (`hopper/flash_fwd_combine_kernel.h`) on:
- Split 0: `(O_spec, LSE_spec)` from Phase 1
- Split 1: `(O_miss, LSE_miss)` from Phase 2

The kernel already implements the correct merge:
```
new_lse = log(exp(lse_spec) + exp(lse_miss))
new_O   = (exp(lse_spec)·O_spec + exp(lse_miss)·O_miss) / (exp(lse_spec) + exp(lse_miss))
```

### Files to Modify

| File | Change |
|---|---|
| `csrc/flash_attn/src/flash.h` | Add `block_table`, `num_blocks` to `Flash_fwd_params` |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | Replace linear N-loop with indexed iteration |
| `hopper/flash.h` | Add same fields to FA-3 params |
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | Replace linear N-loop in consumer mainloop |
| `hopper/flash_api.cpp` | Wire new params through pybind11 |
| `hopper/flash_attn_interface.py` | Expose `block_table` argument |

### Correctness Validation

For any query attending to blocks {B0, B2, B3, B5} where {B3} is the missed block:

```
FA_full({B0,B2,B3,B5})  ==  Combine( FA({B0,B2,B5}),  FA_indexed({B3}) )
```

Verify numerically in Python before touching CUDA code.

---

## Future Plan

### Option B — Single-Pass Warm-Start

Avoid the second kernel launch entirely by injecting the previous softmax state directly into the missed-block FA pass. This eliminates combine kernel overhead.

**The fundamental problem with warm-start**: FA writes `LSE = m·scale·ln2 + log(l)`, a single scalar fusing the running max `m` and running sum `l`. From LSE alone, `m` and `l` cannot be recovered independently, so the unnormalized accumulator `acc_o = O_final·l` cannot be reconstructed.

**Solution**: Store raw `row_max` and `row_sum` separately alongside O in the speculative pass output.

**Important**: `row_max` and `row_sum` are register-only during the kernel. `Softmax::finalize()` (`hopper/softmax.h:137`) overwrites `row_sum` in-place with the fused LSE before the epilogue runs — the raw values are destroyed. To expose them, new output pointers must be written *before* `finalize()` is called, and threaded through the epilogue. This is non-trivial additional work.

Extra outputs from the speculative pass (new, requires epilogue changes):
```cpp
float* ptr_row_max;   // [batch, nheads, seqlen_q] — raw running max, written before finalize()
float* ptr_row_sum;   // [batch, nheads, seqlen_q] — raw running sum, written before finalize()
```

Extra inputs to the incremental pass:
```cpp
Element* ptr_O_prev;        // previous normalized output
float*   ptr_row_max_prev;
float*   ptr_row_sum_prev;
bool     is_incremental;
```

Kernel warm-start (consumer warp-group, before the pipeline loop):
```cpp
if (is_incremental) {
    row_max = load(ptr_row_max_prev[q_idx]);
    row_sum = load(ptr_row_sum_prev[q_idx]);
    for (mi, ni): acc_o(mi,ni) = O_prev(mi,ni) * row_sum(mi);
}
```

The rest of the online softmax loop is unchanged — it handles `Is_first=false` naturally after warm-start. In FA-3, this init must happen inside the consumer warp-group before `pipeline.consumer_wait()`.

Pursue Option B only if Phase 2 + Phase 3 kernel launch overhead is measured to be significant relative to the missed-block compute.
