# Updating Log — Speculative Sparse Attention (plan.md)

This log records the incremental edits applied to implement the
speculative-sparse-attention plan from `plan.md`. Each stage is isolated
and reviewable on its own. No stage is started until the previous one
is accepted.

---

## Design decisions (locked)

- **New params fields:** `sparse_block_table` (`int*`) and
  `sparse_num_blocks` (`int`). Distinct from the existing paged-KV
  `block_table` (FA-2) / `page_table` (FA-3).
- **Paged-KV coexistence:** *mutually exclusive* in v1. At most one of
  paged-KV or `sparse_block_table` may be non-null per call.
- **Causal mask on the sparse path:** option (iii). The caller
  guarantees every entry in `sparse_block_table` refers to a block whose
  last K position is strictly less than the first Q position of the
  current M-block, so every sparse entry is fully-allowed. The sparse
  kernel path therefore runs with **no causal / local mask**. The
  diagonal block containing the query itself stays the responsibility of
  the Phase-1 dense speculative pass, which keeps default FA masking
  unchanged.
- **Host-side contract enforcement:** documentation only for now; no
  runtime assert on the host side. Revisit if needed.

---

## Stage 1 — Python numerical validation of the combine identity

**Goal:** before touching any CUDA code, prove in pure PyTorch + fp64
that the two-pass + combine identity holds:

    FA(A ∪ B) == Combine( FA(A), FA(B) )

This is the prerequisite check from `plan.md` line 84–92.

**File created:** `tests/test_sparse_combine_identity.py` (new file).

Contents:

- `reference_attention(q, k, v, scale) -> (O, LSE)` — dense scaled
  dot-product attention with natural-log LSE. 18 lines.
- `combine_partials(o1, lse1, o2, lse2) -> (O, LSE)` — online-softmax
  merge using the `lse_max` trick, mirroring what
  `FlashAttnFwdCombine` (`hopper/flash_fwd_combine_kernel.h`) does on
  GPU.
- `gather_blocks(x, block_indices, block_size)` — helper to assemble
  K/V tensors from a list of absolute block indices.
- `test_combine_identity_matches_dense` — parametrised over
  `seqlen_q ∈ {1, 4, 16}`, `block_size ∈ {32, 64, 128}`, and four
  A/B splits (`interleaved`, `spec_heavy`, `miss_heavy`,
  `contiguous_miss`). Asserts the combined output and LSE match the
  dense reference to 1e-10 absolute tolerance in fp64.
- `test_combine_is_order_invariant` — `Combine(A, B) == Combine(B, A)`.
- `test_combine_with_empty_partial_is_identity` — an empty partial
  (`lse = -inf`) leaves the other partial unchanged, which is the
  expected behavior when `sparse_num_blocks == 0`.

**Result:** 38 / 38 tests pass under
`source /HSC/users/wangtian/venv/llm/bin/activate && pytest
tests/test_sparse_combine_identity.py -v`. The combine identity is
numerically sound, so stages 2–5 can proceed with confidence that the
target math is correct.

**Files modified:** none. Only a new test file was added.

---

## Stage 2 — Param struct fields (no binding yet)

**Scope narrowed from the original sketch.** Stage 2 now adds only the
new `Flash_fwd_params` fields on both FA-2 and FA-3; pybind / Python
wrapper threading is deferred to the stage that first needs it
(stage 3 for FA-2, stage 4 for FA-3, stage 5 for orchestration).

Why narrower: `set_params_fprop` in both APIs zero-initializes the
struct via `params = {};` before populating fields, so new POD fields
default to `nullptr` / `0` with no other edits. Adding a binding arg
before any kernel reads the field is wasted surface and is harder to
review in isolation. Small, pure scaffolding first.

### File: `csrc/flash_attn/src/flash.h` (FA-2)

Added two fields to `Flash_fwd_params`, placed immediately after the
paged-KV block (line 105 in the original file, unchanged). The new
block:

```cpp
// Speculative sparse attention (see plan.md).
// When sparse_block_table != nullptr, the kernel iterates over the listed
// K/V block indices instead of the dense [n_block_min, n_block_max) range.
// Every entry must index a fully-historical block (strictly left of the
// current M-block's query range); the kernel then runs the no-mask fast
// path. sparse_block_table is mutually exclusive with the paged-KV
// block_table above — callers must set at most one.
int * __restrict__ sparse_block_table;
int sparse_num_blocks;
```

Placement: directly after `int page_block_size;` (was line 105). No
other lines touched.

### File: `hopper/flash.h` (FA-3)

Same change to the FA-3 `Flash_fwd_params`, placed immediately after
the paged-KV block (originally ending at `bool pagedkv_tma;` on
line 125). The new block:

```cpp
// Speculative sparse attention (see plan.md).
// When sparse_block_table != nullptr, the mainloop iterates over the listed
// K/V block indices instead of the dense [n_block_min, n_block_max) range.
// Every entry must index a fully-historical block (strictly left of the
// current M-block's query range); the kernel then runs the no-mask fast
// path. sparse_block_table is mutually exclusive with page_table above —
// callers must set at most one.
int * __restrict__ sparse_block_table;
int sparse_num_blocks;
```

Placement: directly after `bool pagedkv_tma;` (was line 125). No
other lines touched.

### Behavior change

None. Both kernels still read the struct exactly as before; the two
new fields are present but no code references them yet. A caller using
the existing API sees no difference.

### Files modified

| File | Lines added | Lines changed |
|---|---|---|
| `csrc/flash_attn/src/flash.h` | 10 | 0 |
| `hopper/flash.h` | 10 | 0 |

## Stage 3 — FA-2 kernel (non-splitkv sparse path)

**Scope narrowed.** I added a sibling function
`compute_attn_1rowblock_sparse` for the sparse path rather than editing
the four existing N-loops in place. The dense path (both
`compute_attn_1rowblock` and `compute_attn_1rowblock_splitkv`) is
byte-for-byte unchanged except for an 11-line dispatch at the top of
`compute_attn_1rowblock`.

Why this instead of the in-place rewrite originally sketched:
- The two dense N-loops (masking + interior) interleave their pipeline
  carefully with the next-iteration K prefetch (`n_block - 1`) and
  interact with mask-step accounting. Shoehorning a sparse-index path
  into the same loops doubles the branching in the hottest inner loop
  and makes every interaction a review liability.
- The splitkv variant (the other two loops) additionally carries
  Append_KV, rotary, and paged-KV pointer arithmetic that are
  irrelevant to the sparse missed-block pass. Touching those loops
  would be pure risk.
- A standalone sparse function has exactly the branches it needs:
  one linear index loop over `sparse_block_table`, no causal/local
  masking, no boundary `n_masking_steps`, no paged-KV.
- Stage 5 orchestration launches two separate FA calls (speculative
  dense + sparse) and combines their outputs. Both calls go through
  the non-splitkv entry point, so splitkv does not need the sparse
  path.

### File: `csrc/flash_attn/src/flash_fwd_kernel.h`

#### Change A — new function `compute_attn_1rowblock_sparse`

Inserted immediately before `compute_attn_1rowblock` (new body spans
lines 51–325 in the edited file). Signature:

```cpp
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K,
         bool Is_softcap, typename Params>
inline __device__ void compute_attn_1rowblock_sparse(
    const Params &params, const int bidb, const int bidh, const int m_block);
```

Template parameters intentionally drop `Is_causal`, `Is_local`,
`Has_alibi`, `Is_dropout`, `Return_softmax` — none apply to the sparse
missed-block contract (all entries are fully-historical, inference-
time path).

Structure (per sparse row-block):

1. **Setup** — `binfo`, constants, read `num_blocks =
   params.sparse_num_blocks`. Early exit when
   `m_block * kBlockM >= binfo.actual_seqlen_q` matches the dense path.
2. **Empty-partial short-circuit** (`num_blocks == 0`) — writes
   `O = 0` and `LSE = -INFINITY` to gmem and returns. Mirrors the
   dense early-exit block but uses `-INFINITY` (not `+INFINITY`) so
   the combine kernel treats this split as contributing nothing.
3. **Tensor construction** — `mQ/gQ/mK/gK/mV/gV`, smem tensors
   (`sQ/sK/sV/sVt/sVtNoSwizzle`), thread partitioning (`tQgQ/tKgK/
   tVgV`, `tQsQ/tKsK/tVsV`), `TiledMma`, copy atoms, identity
   predicates. All copied verbatim from the dense prologue; these are
   the same partitions the dense path uses.
4. **Q prologue** — `cp_async` Q→smem, conditional
   `cp_async_fence`/`wait<0>`/sync under `Is_Q_in_regs` and
   `Share_Q_K_smem` exactly as the dense path does.
5. **Initial K prefetch** — `cp_async` K for
   `sparse_block_table[0]`, unconditional `Is_even_MN=true` (no
   seqlen-k masking; historical by contract), then
   `cp_async_fence`.
6. **`Is_Q_in_regs && !Share_Q_K_smem` wait/copy** — `cp_async_wait<1>`
   + sync + Q smem→reg, same as dense.
7. **`clear(acc_o)`**, construct `Softmax`.
8. **Main sparse loop** `for (i in [0, num_blocks))`:
   - `cp_async_wait<0>` + sync, `cp_async` V for
     `sparse_block_table[i]`, `cp_async_fence`.
   - `gemm(acc_s = Q @ K)`, conditional `apply_softcap`.
   - `cp_async_wait<0>` + sync.
   - If `i + 1 < num_blocks`: `cp_async` K for
     `sparse_block_table[i + 1]`, `cp_async_fence` (lookahead
     prefetch, replaces the dense `tKgK(_,_,_, n_block - 1)` pattern).
   - `softmax_rescale_o<Is_first=(i==0), Check_inf=false>`.
   - `convert_type` → `gemm_rs(acc_o += P @ V)`.
9. **Epilogue** — `normalize_softmax_lse`, write O via smem stage
   and gmem copy, write LSE. Identical in structure to the dense
   epilogue.

Total: 269 lines added.

#### Change B — dispatch in `compute_attn_1rowblock`

Directly after the template declaration of `compute_attn_1rowblock`
(line 329 in the edited file), added at the very top of the function
body (before any `using` / `extern __shared__`):

```cpp
// Speculative sparse attention dispatch (plan.md Phase 2). When set,
// iterate only over params.sparse_block_table; the dense path below is
// unchanged. Sparse is handled in a sibling function that does not
// depend on the Is_causal / Is_local / Has_alibi / Is_dropout /
// Return_softmax template arguments, so we pass only what it needs.
if (params.sparse_block_table != nullptr) {
    compute_attn_1rowblock_sparse<Kernel_traits, Is_even_MN, Is_even_K, Is_softcap, Params>(
        params, bidb, bidh, m_block
    );
    return;
}
```

11 lines added. No other line in `compute_attn_1rowblock` changed.

#### What was NOT touched

- `compute_attn_1rowblock_splitkv` — completely unchanged. Splitkv
  callers get dense behavior regardless of `sparse_block_table`.
  (Orchestration in stage 5 will route through the non-splitkv path.)
- All kernel-launch templates (`flash_fwd_launch_template.h`) and the
  Python binding (`flash_api.cpp`) — unchanged. Without the binding
  exposing `sparse_block_table`, `params.sparse_block_table` stays
  `nullptr` on every existing call, and the dispatch falls through to
  the dense path. Zero behavior change for existing users.

### Contract reminders encoded in this path

- Every entry in `sparse_block_table` must be a fully-historical
  block: `last_k_pos_of_block < first_q_pos_of_m_block`. The sparse
  loop does no masking and uses `Is_even_MN=true` on every K/V load,
  so an out-of-range entry would read garbage K/V.
- `sparse_num_blocks == 0` is legal and writes `O = 0`,
  `LSE = -INFINITY` (empty partial for combine).
- `sparse_block_table` and paged-KV `block_table` are mutually
  exclusive. The sparse path reads K/V at the natural stride; a
  paged-KV layout is not supported here.

### Files modified

| File | Lines added | Lines changed |
|---|---|---|
| `csrc/flash_attn/src/flash_fwd_kernel.h` | 280 | 0 |

## Stage 4 — FA-3 mainloop sparse sibling methods (Option B)

**Scope.** FA-3 uses a warp-specialized producer/consumer mainloop with
TMA + WGMMA pipelining that is substantially more intricate than FA-2's
cp.async loop: the producer `load` method drives `MainloopPipelineK`/
`MainloopPipelineV`(`/MainloopPipelineVt`) via `PipelineState` cursors,
and the consumer `mma` method runs `IntraWGOverlap` (V one stage behind
K) with per-iteration `consumer_wait`/`consumer_release` and a
`softmax_rescale_O` cadence that differs from the FA-2 rescale pattern.

Rather than splice a sparse index path into those two already-loaded
functions, I added **sibling methods** `load_sparse` and `mma_sparse`
on `CollectiveMainloopFwdSm90` (Option B). The dense methods are
byte-for-byte unchanged. This mirrors the stage-3 choice for FA-2.

The sparse path is deliberately restricted to a single common variant
set — enforced by `static_assert` inside the new methods and by a
`constexpr` gate at the kernel-entry dispatch:

- `!AppendKV`, `!PagedKVNonTMA`, `!HasQv`, `!Transpose_V`,
  `!LargeHeadDimV`, `IntraWGOverlap = true`.

This covers the inference-time configurations that speculative sparse
attention will be used from. For any other variant, the compiled
kernel has no sparse path, and the host-side check (stage 5) rejects
the combination.

### File: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`

#### Change A — new `Arguments` / `Params` fields

Added two fields to both `Arguments` and `Params` structs on
`CollectiveMainloopFwdSm90`, placed adjacent to the paged-KV fields:

```cpp
// Speculative sparse attention (plan.md). When ptr_sparse_block_table
// is non-null the sparse sibling path is taken; sparse_num_blocks gives
// the length of the block-index array. Mutually exclusive with paged
// KV (see host-side check in flash_api.cpp).
int const* ptr_sparse_block_table;
int sparse_num_blocks;
```

Threaded through `to_underlying_arguments()` so args → params just
forwards the two fields.

#### Change B — expose `IntraWGOverlap` as a `static constexpr`

Added `static constexpr bool IsIntraWGOverlap = IntraWGOverlap;` on
the collective. The kernel-entry dispatch needs to gate on this
compile-time flag, and a bare template parameter is not visible
through `CollectiveMainloop::` without this alias. Same pattern as
the existing `PagedKVNonTMA` exposure.

#### Change C — `load_sparse` method

Added after `load_tail` (~200 lines). Shape mirrors `load` but:

- `static_assert`s the restricted variant set so a mis-instantiation
  is caught at compile time instead of silently running the wrong
  path.
- Early returns when `sparse_num_blocks <= 0`. Producer contributes
  nothing; the consumer's `mma_sparse` short-circuit writes the empty
  partial.
- Does not use `paged_kv_manager` — K/V TMA descriptors index the
  natural-stride KV tensors.
- Resolves each iteration's block index via
  `int n_block = params.ptr_sparse_block_table[i];` and loads via
  `tKgK_TMA(_, n_block, bidb_kv)` / `tVgV_TMA(_, n_block, bidb_kv)`.
- Keeps the `IntraWGOverlap` cadence: first K load before `barrier_O`
  wait, then for `i in [1, num_blocks)` load K[i] and V[i-1], and a
  trailing V[last] after the loop. No `Transpose_V` branch; no
  `load_tail` changes — the existing `load_tail` in the dense path
  still drains the pipeline when `mma_sparse` is the consumer.
- Same `PipelineState` advance pattern as `load`: one
  `producer_acquire` → TMA issue → `producer_commit` per K or V
  issue, in lockstep with the consumer.

#### Change D — `mma_sparse` method

Added after `mma` (~200 lines). Returns `bool` (matches `mma` so the
kernel-entry code can treat the two uniformly). Structure:

- `static_assert`s the same restricted variant set.
- Returns `false` immediately when `sparse_num_blocks <= 0` so the
  caller skips the epilogue write for this tile (combine treats the
  absent partial as `LSE = -INFINITY`, which is the empty-partial
  contract from stage 1).
- Waits on `barrier_Q`, runs the prologue Q@K for block[0] with
  `softmax.rescale_O<Is_first=true, Check_inf=false>`, writes P.
- `fwd_step` lambda mirrors the dense `IntraWGOverlap` body with the
  mask hook **omitted** — the sparse contract guarantees fully-
  historical blocks, so there is no masked iteration.
- Main loop `for (i = 1; i < sparse_num_blocks; ++i)` issues Q@K on
  block[i], the running P@V on V[i-1], softmax rescale, P write.
- `QueryEmpty` named-barrier arrive after the last Q use, matching
  the dense path.
- Epilogue: P@V on V[last], `softmax.finalize`, return `true`.
- No `mma_pv` counterpart — sparse path is `!LargeHeadDimV` by
  construction.

### File: `hopper/flash_fwd_launch_template.h`

Added `params.sparse_block_table, params.sparse_num_blocks` to the
`mainloop_args` initializer so the two new fields are wired from
`Flash_fwd_params` (stage 2) into the collective args.

### File: `hopper/flash_fwd_kernel_sm90.h`

Two dispatch points — one at the producer `load` call, one at the
consumer `mma` call. Both guarded by the same constexpr:

```cpp
constexpr bool SparsePathSupported =
    !AppendKV && !CollectiveMainloop::PagedKVNonTMA && !HasQv &&
    !Transpose_V && !LargeHeadDimV && CollectiveMainloop::IsIntraWGOverlap;
```

- For unsupported template variants, `SparsePathSupported` is `false`
  → `use_sparse_load`/`use_sparse_mma` is a compile-time `false` → the
  dense path is selected and the sparse call is not instantiated.
- For supported variants, the runtime check
  `params.mainloop.ptr_sparse_block_table != nullptr` selects at
  launch time between dense and sparse.

Producer dispatch replaces the single `mainloop.load(...)` call with:

```cpp
bool use_sparse_load = false;
if constexpr (SparsePathSupported) {
    use_sparse_load = params.mainloop.ptr_sparse_block_table != nullptr;
}
if (use_sparse_load) {
    if constexpr (SparsePathSupported) {
        mainloop.load_sparse(...);
    }
} else {
    mainloop.load(...);
}
```

Consumer dispatch wraps the `!LargeHeadDimV` branch of the existing
`mma` / `mma_pv` switch; the `LargeHeadDimV` branch is untouched
because the sparse path excludes `LargeHeadDimV`.

### What was NOT touched

- `mma_pv` — the `LargeHeadDimV` split-epilogue consumer remains for
  the dense path only.
- Any backward-pass file.
- `flash_api.cpp` pybind / Python interface — stage 5.
- Paged-KV code paths (`paged_kv.h`, `PagedKVNonTMA` branch in the
  dense mainloop). Sparse and paged-KV remain mutually exclusive;
  stage 5 adds the host-side assert.

### Behavior change

None for existing callers. `params.mainloop.ptr_sparse_block_table`
is `nullptr` on every current call, every dispatch falls through to
the dense path, and the sparse sibling methods are only instantiated
in the compiled kernel (not executed).

### Files modified

| File | Lines added | Lines changed |
|---|---|---|
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | ~420 | ~4 |
| `hopper/flash_fwd_launch_template.h` | 2 | 0 |
| `hopper/flash_fwd_kernel_sm90.h` | ~45 | ~2 |

## Stage 5 — Host binding + Python two-pass orchestration

**Scope.** Expose `sparse_block_table` through the Python / pybind
surface of both FA-2 and FA-3, enforce the host-side mutual-exclusion
contract, and add a high-level Python wrapper that drives the two-pass
+ combine recipe from plan.md (speculative dense pass → sparse missed-
block pass → online-softmax combine via the existing
`FlashAttnFwdCombine` kernel).

**Design notes that shaped the edits:**

- The new arg is appended at the very end of `mha_fwd` in both APIs
  (rather than inserted next to `page_table`), so no existing Python
  call site has to re-shuffle positional arguments. The one internal
  call inside each `_flash_attn_forward` custom_op just appends a
  trailing `None`.
- The low-level custom_op signatures (`_flash_attn_forward`) are
  unchanged. Exposing `sparse_block_table` through them would require
  re-indexing `setup_context` and adjusting the backward `None`-count,
  which is pure churn for a forward-only inference feature. The new
  Python wrapper instead calls `torch.ops.flash_attn_3.fwd` directly.
- The FA-3 mutex check is structural (both sparse and paged-KV reach
  the kernel via the same param struct, so we reject the combination
  at the host). The FA-2 non-splitkv `mha_fwd` has no paged-KV path of
  its own, but the sparse dispatch lives only in
  `compute_attn_1rowblock` (stage 3). So on FA-2 we force
  `num_splits = 1` whenever `sparse_block_table` is passed, which
  routes through the non-splitkv path.

### File: `hopper/flash_api.cpp` (FA-3)

1. Appended a new arg to `mha_fwd`:
   ```cpp
   std::optional<at::Tensor> sparse_block_table_  // (sparse_num_blocks,) int32
   ```
2. After the paged-KV parse block, added:
   ```cpp
   at::Tensor sparse_block_table;
   const bool use_sparse = sparse_block_table_.has_value();
   if (use_sparse) {
       TORCH_CHECK(!paged_KV,
                   "sparse_block_table is mutually exclusive with page_table (paged KV).");
       // … CHECK_DEVICE / CHECK_CONTIGUOUS / dtype == int32 / dim == 1 …
   }
   ```
3. In the params-wiring section (next to the paged-KV wiring), added:
   ```cpp
   if (use_sparse) {
       params.sparse_block_table = sparse_block_table.data_ptr<int>();
       params.sparse_num_blocks  = static_cast<int>(sparse_block_table.numel());
   }
   ```
4. Updated the `TORCH_LIBRARY` `fwd` schema — appended
   `"Tensor? sparse_block_table = None"` before the return type, so
   the dispatcher knows about the new kwarg.

### File: `csrc/flash_attn/flash_api.cpp` (FA-2)

1. Appended `std::optional<at::Tensor> sparse_block_table_` to
   `mha_fwd` (the non-varlen, non-kvcache entry).
2. Validation block (CUDA, int32, 1D, contiguous) placed immediately
   before `set_params_fprop`. Also asserts
   `p_dropout == 0 && !alibi_slopes.has_value()` — neither applies to
   the speculative-decoding use case and both would require extra
   codegen that the stage-3 sparse kernel does not have.
3. Params wiring after `set_params_fprop`:
   ```cpp
   if (use_sparse) {
       params.sparse_block_table = sparse_block_table.data_ptr<int>();
       params.sparse_num_blocks  = static_cast<int>(sparse_block_table.numel());
   }
   ```
4. **Route around splitkv:** the stage-3 sparse dispatch only lives in
   `compute_attn_1rowblock` (non-splitkv). The `set_params_splitkv`
   call now forces `num_splits = 1` whenever `use_sparse` is true.
5. FA-2 uses `PYBIND11_MODULE` (direct binding), so no separate
   schema string — the new arg is visible to Python through the
   updated C++ signature.

### File: `flash_attn/flash_attn_interface.py` (FA-2)

Appended one trailing `None` to the `flash_attn_gpu.fwd(...)` call
inside `_flash_attn_forward` for the new `sparse_block_table` arg.
Custom_op signature is unchanged.

### File: `hopper/flash_attn_interface.py` (FA-3)

1. Appended one trailing `None` to the `flash_attn_3_gpu.fwd(...)`
   call inside `_flash_attn_forward`. Custom_op signature is
   unchanged.
2. Added two new Python entry points (next to `flash_attn_combine`):

   **`flash_attn_with_sparse_block_table(q, k, v, sparse_block_table,
   softmax_scale=None, sm_margin=0)`** — phase-2 only. Calls
   `torch.ops.flash_attn_3.fwd` directly with `sparse_block_table`
   set, `is_causal=False` (the sparse contract guarantees fully-
   historical blocks), `num_splits=1` (non-splitkv). Returns
   `(out, softmax_lse)`.

   **`flash_attn_speculative_sparse(q, k_spec, v_spec, k, v,
   sparse_block_table, softmax_scale=None, causal_spec=True,
   sm_margin=0)`** — full orchestration:

   - Pass 1: `_flash_attn_forward(q, k_spec, v_spec,
     causal=causal_spec)` — dense speculative pass on caller-gathered
     speculative K/V (contains the diagonal block so normal causal
     masking still applies when `causal_spec=True`).
   - Pass 2: `flash_attn_with_sparse_block_table(q, k, v,
     sparse_block_table)` — missed-block pass through the full K/V.
   - Combine: stack the two passes into the `(num_splits, b, s, h,
     ·)` layout the existing combine kernel expects, transposing
     `softmax_lse` from `(b, h, s)` to `(b, s, h)` for stacking, and
     call `flash_attn_combine`. Returns `(out, softmax_lse)` with
     `softmax_lse` transposed back to `(b, h, s)` to match every
     other FA-3 entry point.

### Backward compatibility

- Existing callers of `flash_attn_gpu.fwd` / `torch.ops.flash_attn_3.fwd`
  at the Python level (including `_flash_attn_forward` in both APIs,
  `FlashAttnFunc`, `FlashAttnVarlenFunc`, `flash_attn_with_kvcache`,
  etc.) pass `None` for `sparse_block_table`, so `params.sparse_block_table`
  stays `nullptr` and every existing kernel dispatch falls through to
  the dense path. Zero behavior change.

### Files modified

| File | Lines added | Lines changed |
|---|---|---|
| `hopper/flash_api.cpp` | 22 | 2 |
| `csrc/flash_attn/flash_api.cpp` | 26 | 2 |
| `hopper/flash_attn_interface.py` | 121 | 1 |
| `flash_attn/flash_attn_interface.py` | 1 | 0 |

### Out of scope for this stage

- Per-batch / per-head `sparse_block_table` — v1 is shared across
  batch/head. The block indices are absolute positions along the K/V
  `seqlen` axis, so callers with heterogeneous miss sets will need to
  launch per-batch (or a future extension).
- Exposure on `flash_attn_varlen_func` and FA-2 `mha_varlen_fwd` /
  `mha_fwd_kvcache`. The stage-3 FA-2 sparse dispatch is non-splitkv
  only, and speculative decoding in practice drives the fixed-shape
  `mha_fwd` path; varlen + sparse is a future extension.
- Host-side assertion that every `sparse_block_table` entry actually
  points to a fully-historical block. This is a documentation
  contract today; a debug-mode scan could be added later.

## Stage 5 — Two-pass + combine orchestration  *(pending)*

Python-side scaffolding in `hopper/flash_attn_interface.py` that runs
the speculative pass and missed-block pass into split 0 / split 1 of an
`oaccum` / `lseaccum` pair, then invokes the existing combine kernel.
