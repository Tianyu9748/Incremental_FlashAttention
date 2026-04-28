# Plan: Fix FA-3 Build Errors from Sparse-Kernel Integration

Source: `hopper/compilation_err.txt` — 52 hard errors across the failing
`instantiations/*.cu` compiles. Two distinct error classes, both rooted in
the speculative-sparse-attention integration.

## Current state of the source (verified before writing Try-2)

Confirmed by inspection right before this round:
- `mainloop_fwd_sm90_tma_gmma_ws.hpp:1036` — still `int n_block = params.sparse_block_table[0];`
- `mainloop_fwd_sm90_tma_gmma_ws.hpp:1067` — still `n_block = params.sparse_block_table[i];`
- `mainloop_fwd_sm80.hpp` — `Arguments` (ends at `seqlens_rotary`, line 215),
  `Params` (ends at `seqlens_rotary`, line 261), and `to_underlying_arguments()`
  return list (ends at `args.seqlens_rotary}`, line 304) all lack the trailing
  sparse fields.
- `flash_fwd_launch_template.h:133` — still passes
  `params.sparse_block_table, params.sparse_num_blocks` as the trailing two
  values to `CollectiveMainloop::Arguments{…}`.
- `flash.h:134-135` — `Flash_fwd_params` declares `sparse_block_table` /
  `sparse_num_blocks` correctly. No change needed in the user-facing struct.

So none of Try-1's edits were applied to the tree. The 52 errors are
identical to last round's log.

---

## Try-1 (previous round, NOT applied)

Diagnosis (same as Try-2 below — the diagnosis was correct, it just never
made it to disk):

- **Error 1 (14×)** — typo at `mainloop_fwd_sm90_tma_gmma_ws.hpp:1036, 1067`:
  `params.sparse_block_table` should be `params.ptr_sparse_block_table` to
  match the field declared at lines 406/466 of the same file.
- **Error 2 (38×)** — at `flash_fwd_launch_template.h:133`, the SM80 branch
  resolves `CollectiveMainloop::Arguments` to `CollectiveMainloopFwdSm80::Arguments`,
  whose layout ends at `seqlens_rotary`, so the trailing
  `params.sparse_block_table, params.sparse_num_blocks` are two extras. Fix
  by adding the two trailing fields to the SM80 `Arguments`/`Params` and
  forwarding them in `to_underlying_arguments()`.

Two open questions documented but not resolved in Try-1:
1. Whether to add a runtime `TORCH_CHECK(dprops->major >= 9, …)` for the
   sparse path in `flash_api.cpp::mha_fwd` so SM80/SM89 calls fail loudly
   instead of silently running dense attention.
2. Whether to fix Error 2 by **structural symmetry** (add unused trailing
   fields to SM80, simpler) or by **arch-guarded init** (`if constexpr (Arch >= 90)`
   around the two extra values at `flash_fwd_launch_template.h:133`,
   stricter separation).

Outcome: plan written to this file, code untouched, build retried, same
52 errors.

---

## Try-2 (this round — to be applied)

Same root causes as Try-1. The diagnosis below is the actionable version,
with the open questions resolved so it can be executed without further input.

### Fix 1 — SM90 mainloop typo (14 errors)

In `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`:

| Line | Current | After |
|---|---|---|
| 1036 | `int n_block = params.sparse_block_table[0];` | `int n_block = params.ptr_sparse_block_table[0];` |
| 1067 | `n_block = params.sparse_block_table[i];` | `n_block = params.ptr_sparse_block_table[i];` |

After the edit, run
`grep -n '\bparams\.sparse_block_table\b' hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
and confirm zero remaining hits. Comments using the bare name (e.g. line
1034's "`sparse_block_table[0] is loaded AFTER…`") are descriptive prose,
not code, and can stay.

### Fix 2 — SM80 mainloop `Arguments`/`Params` shape (38 errors)

**Decision (resolves Try-1 Q2):** structural symmetry. Mirror SM90's two
trailing fields into SM80's `Arguments`/`Params`. Justification:
- Matches the pattern already used in this file for other optional fields
  (`kv_batch_idx`, `leftpad_k`, `seqlens_rotary`) that SM80 carries even
  when not exercised.
- Touches a single file (`mainloop_fwd_sm80.hpp`); leaves the shared
  `flash_fwd_launch_template.h` clean of arch conditionals.
- The fields are unread on SM80: the sparse dispatch is gated by
  `IsIntraWGOverlap` in `flash_fwd_kernel_sm90.h`, which is false on SM80.
  Default values (`nullptr`, `0`) flow through unused.

In `hopper/mainloop_fwd_sm80.hpp` make three additions:

1. **`struct Arguments`** — append after the existing `seqlens_rotary` line
   (currently line 215):
   ```cpp
   int const* const ptr_sparse_block_table = nullptr;
   int const sparse_num_blocks = 0;
   ```
2. **`struct Params`** — append after the existing `seqlens_rotary` line
   (currently line 261), the same two lines.
3. **`to_underlying_arguments()`** — the return brace list currently ends
   `…, args.leftpad_k, args.seqlens_rotary}` at line 304. Replace the
   trailing `}` with `, args.ptr_sparse_block_table, args.sparse_num_blocks}`.

No kernel changes for SM80; the fields sit unread.

### Deferred (resolves Try-1 Q1, not part of the build fix)

Runtime arch-check for sparse calls on non-Hopper GPUs is a behavioral
concern, not a compile blocker. Defer to a follow-up: silently running
dense over the full K/V on SM80 is the worst case, and that's a behavior
bug, not a crash. Re-evaluate after the build is green and tests pass on
SM90.

### Verification after applying Try-2

1. `grep -n 'sparse_block_table\|sparse_num_blocks' hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp hopper/mainloop_fwd_sm80.hpp hopper/flash_fwd_launch_template.h hopper/flash.h`
   - SM90 mainloop: only `ptr_sparse_block_table` / `sparse_num_blocks` at
     declarations + reads; no bare `params.sparse_block_table`.
   - SM80 mainloop: two declarations in `Arguments`, two in `Params`, two
     forwarded in `to_underlying_arguments()`.
   - `flash_fwd_launch_template.h:133`: unchanged — passes the user-facing
     `Flash_fwd_params` field names, which is correct.
   - `flash.h:134-135`: unchanged.
2. Backward path stays untouched:
   `grep -n 'sparse_block_table' hopper/mainloop_bwd_*.hpp hopper/flash_bwd_launch_template.h`
   should return nothing.
3. Rebuild:
   ```bash
   cd hopper && pip install . --no-build-isolation 2>&1 | tee compilation_err.txt
   ```
   Expected: all 52 prior errors clear; the wheel build proceeds past the
   `instantiations/*_sm80.cu` and `instantiations/*_sm90.cu` compile phase.

### What to do if Try-2 still fails

Re-read `compilation_err.txt`. Likely follow-on issues if any surface:

- **More "no member" errors** on `Flash_fwd_params` — would mean a stray
  field reference somewhere in `flash_api.cpp` or another `.hpp` that
  doesn't match `flash.h:134-135`. Grep the offending name and align.
- **"Too many initializer values" on backward** — would mean the bwd
  collective `Arguments` was also widened somewhere; not the case in the
  current tree, but if it appears, mirror the same SM80-side widening in
  `mainloop_bwd_sm80.hpp`.
- **Linker errors after compile succeeds** — out of scope of Try-2; report
  back with the new log.

---

## Try-3 (this round — applied)

### Result of Try-2

All 52 prior compile errors are gone. The new `compilation_err.txt`
contains zero `error:` matches in the nvcc output and zero `warning:`
lines. Every `instantiations/*.cu` compiled, the linker produced
`/tmp/.../sparse_flash_attn_3/_C.abi3.so` successfully. Try-2 was
correct.

### New (and only) failure

```
creating /tmp/tmpjzz_z8yk.build-lib/sparse_flash_attn_3
x86_64-linux-gnu-g++ ... -o /tmp/tmpjzz_z8yk.build-lib/sparse_flash_attn_3/_C.abi3.so
copying /tmp/tmpjzz_z8yk.build-lib/sparse_flash_attn_3/_C.abi3.so -> sparse_flash_attn_3
error: could not create 'sparse_flash_attn_3/_C.abi3.so': No such file or directory
```

This is at the **post-link copy** step of an *editable* install
(`Building editable for sparse_flash_attn_3 (pyproject.toml)` per the
log header). setuptools tried to copy the freshly linked `.so` from
the temp build dir to `hopper/sparse_flash_attn_3/_C.abi3.so`, but
that destination directory does not exist.

### Why the previous `flash_attn_3` build worked but `sparse_flash_attn_3` editable doesn't

- The hopper source tree is **flat**: `__init__.py`, `flash_attn_interface.py`,
  `flash_attn_config.py` all sit directly under `hopper/`. There is no
  `flash_attn_3/` source subdirectory and never was — confirmed via the
  preserved `hopper/flash_attn_3.egg-info/SOURCES.txt`, which lists
  only files at hopper-root and under `instantiations/`.
- The previous successful build (`flash_attn_3`) was a **non-editable**
  install. For non-editable, setuptools auto-creates
  `build/lib.linux-x86_64-cpython-312/flash_attn_3/` and drops the `.so`
  there — no source-tree directory needed. The leftover
  `hopper/build/lib.linux-x86_64-cpython-312/flash_attn_3/_C.abi3.so`
  on disk is from that run.
- This time the user invoked an **editable** install. Editable install
  expects `_C.abi3.so` to land *in the source tree* (so in-place
  imports resolve), and setuptools refuses to create the target
  directory implicitly.

The extension is named `sparse_flash_attn_3._C` (`hopper/setup.py:626`,
unchanged from upstream's `flash_attn_3._C`). With a dot in the name,
setuptools insists the prefix (`sparse_flash_attn_3`) be a real package
directory at the source root.

### Fix — create the package directory

Single change:

```bash
mkdir -p hopper/sparse_flash_attn_3
: > hopper/sparse_flash_attn_3/__init__.py
```

Empty `__init__.py` on purpose:
- `flash_attn_interface.py:14` and `:24` already do
  `import sparse_flash_attn_3._C` explicitly, so the package's own
  `__init__.py` doesn't need to re-export anything.
- An empty file makes the directory a regular package without coupling
  package-import success to the C-extension being built. If `_C.so`
  is missing for any reason, `import sparse_flash_attn_3` still
  succeeds; only the explicit `import sparse_flash_attn_3._C` raises.
- `find_packages()` at `hopper/setup.py:709` will now discover
  `sparse_flash_attn_3` as a real package, which aligns with the
  dotted extension name and removes the layout mismatch.

Applied:
```
hopper/sparse_flash_attn_3/__init__.py   (empty, 0 bytes)
```

### Verification

- `ls hopper/sparse_flash_attn_3/__init__.py` exists, size 0.
- No source code changed; this is a layout-only fix.
- Re-run from `hopper/`:
  ```bash
  pip install -e . --no-build-isolation 2>&1 | tee compilation_err.txt
  ```
  Expected: link succeeds (already did under Try-2), copy succeeds
  this time, editable install completes, `python -c "import sparse_flash_attn_3._C"` works.

### Alternative the user can pick instead

If the user prefers to keep the source tree pristine (no new directory),
switch the install command from editable to non-editable:
```bash
cd hopper && pip install . --no-build-isolation
```
That bypasses the missing-directory issue entirely (setuptools creates
it under `build/lib`) but loses the editable workflow. Try-3 keeps
`-e` working; this is the trade-off.

---

## Things I still cannot confirm without input

1. **SM80 runtime safety.** Same as Try-1 Q1, deferred above. If you'd
   like me to fold the `TORCH_CHECK` into Try-2 instead of deferring, say
   so and I'll add it to `flash_api.cpp::mha_fwd` on the sparse branch.
2. **Whether the sparse path is exercised at all on SM80 in tests.** If
   any test under `hopper/test_*.py` calls
   `flash_attn_with_sparse_block_table` without a `dprops.major >= 9`
   gate, it will silently run dense on Ampere and produce wrong results.
   I haven't audited the tests; flag if you want that done before merging.
