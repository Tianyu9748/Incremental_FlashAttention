# Instructions — Renaming & Building the Sparse-Flash-Attention Fork

This repo is a fork of Dao-AILab/flash-attention carrying the speculative
sparse attention kernels (see `plan.md` and `updating_log.md`). Because
your virtualenv already has the upstream `flash-attn` installed, building
this tree under its original names would:

- collide on the **pip distribution name** (`flash-attn` / `flash_attn_3`);
- collide on the **top-level Python package** (`flash_attn` / files in
  `flash_attn_3._C`);
- **double-register torch ops** (`torch.ops.flash_attn::*` for FA-2 and
  `torch.ops.flash_attn_3::*` for FA-3) — PyTorch raises
  `RuntimeError: Tried to register an operator (...) but an operator with
  that name has already been registered` the second a script imports both.

To keep the upstream install working untouched, we rename this fork's
artifacts to:

| Component | Upstream | This fork |
|---|---|---|
| FA-2 pip distribution | `flash-attn` | `sparse-flash-attn-2` |
| FA-2 Python package dir | `flash_attn/` | `sparse_flash_attn_2/` |
| FA-2 compiled extension | `flash_attn_2_cuda` | `sparse_flash_attn_2_cuda` |
| FA-2 torch.ops namespace | `flash_attn` | `sparse_flash_attn_2` |
| FA-3 pip distribution | `flash_attn_3` | `sparse-flash-attn-3` |
| FA-3 Python package dir | `flash_attn_3/` (`._C` ext) | `sparse_flash_attn_3/` |
| FA-3 compiled extension | `flash_attn_3._C` | `sparse_flash_attn_3._C` |
| FA-3 torch.ops namespace | `flash_attn_3` | `sparse_flash_attn_3` |

The sparse kernel code (`compute_attn_1rowblock_sparse`, `load_sparse`,
`mma_sparse`, the Python `flash_attn_with_sparse_block_table` /
`flash_attn_speculative_sparse` wrappers) is unaffected by the rename —
the edits below are purely packaging.

---

## Part 1 — Rename FA-2 to `sparse-flash-attn-2`

Work in `/home/wangtian/flash-attention/`.

### 1.1 Rename the Python package directory

```bash
git mv flash_attn sparse_flash_attn_2
```

### 1.2 Edit `setup.py`

- Line 54:

  ```python
  PACKAGE_NAME = "sparse_flash_attn_2"
  ```

- Line 254 (`check_if_cuda_home_none` argument — label only):

  ```python
  check_if_cuda_home_none("sparse_flash_attn_2")
  ```

- Both extension definitions (currently line 305 and line 528):

  ```python
  name="sparse_flash_attn_2_cuda",
  ```

- Line 443 (`check_if_rocm_home_none` — label only, ROCm path):

  ```python
  check_if_rocm_home_none("sparse_flash_attn_2")
  ```

- Line 537 (`get_package_version` reads `__init__.py` by hardcoded
  path to extract `__version__`):

  ```python
  with open(Path(this_dir) / "sparse_flash_attn_2" / "__init__.py", "r") as f:
  ```

  Without this edit, `pip install` fails immediately with
  `FileNotFoundError: … /flash_attn/__init__.py` before any compile
  starts, because setup.py is invoked before `find_packages` resolves.

- `find_packages(exclude=...)` already picks up the new directory by
  name (no edit needed) because it excludes everything except the
  top-level package directories.

### 1.3 Edit `sparse_flash_attn_2/flash_attn_interface.py`

- Line 15 and 17 (the import + warning message):

  ```python
  import sparse_flash_attn_2_cuda
  ...
  warnings.warn("sparse_flash_attn_2_cuda (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
  ```

- Line 23:

  ```python
  import sparse_flash_attn_2_cuda as flash_attn_gpu
  ```

  (Keep the local alias `flash_attn_gpu` — it keeps the rest of the
  file unchanged.)

- Replace every `"flash_attn::..."` decorator string (used by the
  `_torch_custom_op_wrapper` / `_torch_register_fake_wrapper` indirections
  defined at the top of the file) with `"sparse_flash_attn_2::..."`, and
  the `torch.ops.flash_attn.*` references (lines 149, 248, 343, 457).
  Sed does this cleanly:

  ```bash
  sed -i \
    -e 's/"flash_attn::/"sparse_flash_attn_2::/g' \
    -e 's/torch\.ops\.flash_attn\./torch.ops.sparse_flash_attn_2./g' \
    sparse_flash_attn_2/flash_attn_interface.py
  ```

  Note: do **not** try to match `custom_op("flash_attn::` — the code uses
  `_torch_custom_op_wrapper(...)` (a local wrapper), so the bare
  `custom_op(` substring doesn't appear and the pattern would silently
  no-op.

### 1.4 Rewrite internal package imports under `sparse_flash_attn_2/`

There are ~47 `.py` files under the renamed directory with internal
`from flash_attn.xxx` / `import flash_attn.xxx` references (models/,
modules/, layers/, ops/, utils/, cute/, etc.). A single sed handles
them all without touching `flash_attn_interface` / `flash_attn_func` /
other `flash_attn_*` identifiers (the `\b` word boundary does not
split on `_`):

```bash
find sparse_flash_attn_2 -name '*.py' -exec sed -i \
  -e 's/\bfrom flash_attn\./from sparse_flash_attn_2./g' \
  -e 's/\bimport flash_attn\./import sparse_flash_attn_2./g' \
  -e 's/\bfrom flash_attn import/from sparse_flash_attn_2 import/g' \
  -e 's/\bimport flash_attn\b/import sparse_flash_attn_2/g' \
  {} +
```

This also fixes `sparse_flash_attn_2/__init__.py` line 8
(`from flash_attn.flash_attn_interface import ...` →
`from sparse_flash_attn_2.flash_attn_interface import ...`). Verify with:

```bash
grep -rn 'from flash_attn\b\|import flash_attn\b\|from flash_attn\.\|import flash_attn\.' sparse_flash_attn_2/
```

Expected: no output.

### 1.5 Edit `MANIFEST.in`

Lines 8–12 recursive-include the package directory by name. After the
`git mv`, they point at a directory that no longer exists, so the
sdist / editable install will drop the C++/CUDA source tree. Rename:

```bash
sed -i 's|recursive-include flash_attn |recursive-include sparse_flash_attn_2 |g' MANIFEST.in
```

### 1.6 Nothing to edit in the C++ side

`csrc/flash_attn/flash_api.cpp` uses `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`,
and `TORCH_EXTENSION_NAME` is defined to the extension name by
setuptools at compile time. Changing `name="sparse_flash_attn_2_cuda"`
in `setup.py` is enough.

---

## Part 2 — Rename FA-3 to `sparse-flash-attn-3`

Work in `/home/wangtian/flash-attention/hopper/`.

### 2.1 Edit `hopper/setup.py`

- Line 36:

  ```python
  PACKAGE_NAME = "sparse_flash_attn_3"
  ```

  Everything else in this file is parameterized on `PACKAGE_NAME`,
  including the extension name `f"{PACKAGE_NAME}._C"` (line 626) and
  the wheel filename.

- The `py_modules=["flash_attn_interface", "flash_attn_config"]` at
  line 720 is fine as-is; those are top-level helper modules and
  their names do not collide with the upstream FA-3 package.

### 2.2 Edit `hopper/flash_attn_interface.py`

Replace the three occurrences that hard-code `flash_attn_3`:

```python
# Line 14
import sparse_flash_attn_3._C
# Line 16 (warning text only — cosmetic)
warnings.warn("sparse_flash_attn_3._C (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
# Line 24
import sparse_flash_attn_3._C # Registers operators with PyTorch
# Line 28
flash_attn_3_gpu = torch.ops.sparse_flash_attn_3
```

Keep the local alias `flash_attn_3_gpu` — every other line in this
file uses it, so no other edits needed.

Also update the `custom_op` / `register_fake` decorator strings and
`torch.ops.flash_attn_3.*` references in this file:

```bash
sed -i \
  -e 's/"flash_attn_3::/"sparse_flash_attn_3::/g' \
  -e 's/torch\.ops\.flash_attn_3\./torch.ops.sparse_flash_attn_3./g' \
  flash_attn_interface.py
```

### 2.3 Edit `hopper/flash_api.cpp`

Two lines define the torch library namespace:

- Line 1697: `TORCH_LIBRARY(sparse_flash_attn_3, m) {`
- Line 1789: `TORCH_LIBRARY_IMPL(sparse_flash_attn_3, CUDA, m) {`

### 2.4 Edit `hopper/flash_api_stable.cpp` (if building the stable-ABI target)

- Line 1892: `STABLE_TORCH_LIBRARY(sparse_flash_attn_3, m) {`
- Line 1983: `STABLE_TORCH_LIBRARY_IMPL(sparse_flash_attn_3, CUDA, m) {`

If you don't build the stable-ABI entry point, leaving this file
alone is fine — it won't be linked in.

### 2.5 Sanity-check the rename

After editing, confirm no stale references remain:

```bash
grep -rn 'flash_attn_3\|flash_attn_2_cuda' hopper/flash_attn_interface.py hopper/flash_api.cpp hopper/setup.py
grep -rn '"flash_attn::' sparse_flash_attn_2/
grep -rn 'flash_attn_2_cuda' sparse_flash_attn_2/ setup.py
```

Every hit should now point to `sparse_flash_attn_2` / `sparse_flash_attn_3`.
Anything left means you missed an occurrence.

---

## Part 3 — Build

Activate the virtualenv first:

```bash
source /HSC/users/wangtian/venv/llm/bin/activate
```

### 3.1 Build FA-2 (`sparse-flash-attn-2`)

From the repo root (`/home/wangtian/flash-attention/`):

```bash
MAX_JOBS=4 pip install -e . --no-build-isolation
```

- `-e` installs editable, so future Python-side edits don't require a
  reinstall (C++ edits still need a rebuild).
- `MAX_JOBS=4` caps the parallel compile processes. FA-2 has ~60
  instantiation `.cu` files; each takes ~1–2 GB of RAM at peak. Raise
  or lower based on the machine. Expect **15–45 min** on a cold
  build.
- `--no-build-isolation` uses your venv's already-installed `torch` /
  `ninja` / `packaging` instead of pip creating a fresh build env
  (which would re-download torch).
- Do **not** set `FLASH_ATTENTION_FORCE_BUILD=FALSE` — by default the
  original upstream `setup.py` tries to download a prebuilt wheel
  matching your torch/CUDA combo. After the rename, no such prebuilt
  wheel exists for `sparse-flash-attn-2`, so the fallback "build from
  source" path is what you want. Force it explicitly if needed:

  ```bash
  FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 pip install -e . --no-build-isolation
  ```

If you only need a subset of head-dims during development, set:

```bash
FLASH_ATTENTION_DISABLE_HDIM32=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
MAX_JOBS=4 pip install -e . --no-build-isolation
```

to cut the build time roughly in half. Keep hdim 64 and 128 for a
usable install. (All `FLASH_ATTENTION_DISABLE_*` flags are defined in
`setup.py`.)

### 3.2 Build FA-3 (`sparse-flash-attn-3`)

From `/home/wangtian/flash-attention/hopper/`:

```bash
cd hopper
MAX_JOBS=4 pip install -e . --no-build-isolation
```

FA-3 needs:

- CUDA ≥ 12.3 with `nvcc` on PATH (FA-3 uses CUTLASS 3.x + Hopper TMA).
- An SM90 GPU to actually run the kernel (compile works without one,
  run requires one). Check with `nvidia-smi`.
- A recent PyTorch built against the same CUDA toolkit major version.

Expect a **20–60 min** cold build; FA-3 has ~80 instantiation files
under `hopper/instantiations/`, and each nvcc invocation uses more
memory than FA-2 does. Drop `MAX_JOBS` to `2` if the machine OOMs.

FA-3 has its own disable flags, mirrored in `hopper/setup.py` and
surfaced as env vars, e.g. `FLASHATTENTION_DISABLE_HDIM64=TRUE`. Same
trick as above for faster dev cycles. Keep hdim 128 — it's the most
commonly used for H100 inference.

### 3.3 Verify the install

```bash
python -c "
import sparse_flash_attn_2
import sparse_flash_attn_2_cuda
print('FA-2 ops:', [o for o in dir(torch.ops.sparse_flash_attn_2) if not o.startswith('_')])
" 2>&1 | head
```

```bash
python -c "
import torch, sparse_flash_attn_3._C
print('FA-3 ops:', [o for o in dir(torch.ops.sparse_flash_attn_3) if not o.startswith('_')])
"
```

Both should print a list ending with `fwd`, `bwd`, etc. If you see
`RuntimeError: operator ... has already been registered`, the rename
of the `TORCH_LIBRARY` namespace / `custom_op` strings (section 1.3 or
2.2–2.4) is incomplete and still colliding with the upstream install.

### 3.4 Run the combine-identity test (sanity check, no GPU needed)

This validates stage 1 of the sparse work:

```bash
pytest tests/test_sparse_combine_identity.py -v
```

Expected: 38 passing cases.

### 3.5 Optional: build just the CUDA extensions without reinstalling

For iterative C++/CUDA development, rebuild in-place:

```bash
# FA-2
python setup.py build_ext --inplace -j 4
# FA-3
cd hopper && python setup.py build_ext --inplace -j 4
```

With `-e` installs, the in-place `.so` is picked up immediately by the
next Python process. No reinstall step required after a C++ edit —
only a rebuild.

---

## Part 4 — Reverting the rename

If you later want to re-sync with upstream, the easiest path is a fresh
branch from the upstream tag, cherry-picking the sparse kernel commits
on top — the rename itself is pure packaging churn and shouldn't be
merged upstream.
