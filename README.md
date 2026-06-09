# Sparse FlashAttention with Incremental Repair

Built on top of [FlashAttention](https://github.com/Dao-AILab/flash-attention) by Tri Dao et al. This fork adds a **speculative + sparse attention** path on FA-3 (Hopper) so a decoding step can attend to a caller-chosen subset of K/V blocks, then "repair" the result over the blocks it missed — all merged exactly via the FlashAttention online-softmax combine, with no recomputation on the overlap.

The vanilla FA-2 and FA-3 forward/backward kernels are unchanged and continue to work. To coexist with an upstream `flash-attn` install in the same virtualenv, the packages here are renamed to `sparse-flash-attn-2` / `sparse-flash-attn-3`.

---

## What's in this fork

- **Sparse block-table forward (FA-3).** A new forward path that attends only to the K/V blocks listed in a `sparse_block_table` (1D `int32`). Each entry covers a `block_size`-token slice of `k`/`v`; `block_size ∈ {16, 32, 64, 128, 256}` at hdim=128. Implemented in `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` via `load_sparse()` + `mma_sparse()`, surfaced as `flash_attn_with_sparse_block_table()`.

- **Two-pass speculative + sparse decoding with online-softmax combine.** Run dense FA over a speculative K/V subset (optionally causal), run sparse FA over the missed blocks, merge with the existing `FlashAttnFwdCombine` kernel — exact, no recomputation on the overlap. Surfaced as `flash_attn_speculative_sparse()`.

- **Per-block partials.** `flash_attn_with_sparse_block_table_partials()` returns one `(o, lse)` per block so callers can choose the subset to combine themselves.

- **Validated combine identity.** `FA(A ∪ B) == Combine(FA(A), FA(B))` is verified numerically in fp64 by `tests/test_sparse_combine_identity.py` (38 passing cases — no GPU required).

- **FA-2 sparse path** (`csrc/flash_attn/`): the same `sparse_block_table` contract is wired through `compute_attn_1rowblock_sparse()` for Ampere/Hopper. The Python surface is the standard FA-2 API; pass `sparse_block_table` through `torch.ops.sparse_flash_attn_2.fwd`.

---

## Requirements

- NVIDIA Hopper GPU (H100 / H800, SM90) to run the FA-3 sparse path. Ampere works for FA-2.
- CUDA ≥ 12.3 with `nvcc` on `PATH`; **CUDA 12.8 recommended**.
- PyTorch built against the same CUDA major version.
- `ninja`, `packaging`, `psutil`.

---

## Installation

Build from source — there are no prebuilt wheels for the renamed packages.

**FA-3 (Hopper, primary):**

```bash
cd hopper
MAX_JOBS=4 pip install -e . --no-build-isolation
```

**FA-2 (optional, Ampere/Hopper fallback):**

```bash
MAX_JOBS=4 pip install -e . --no-build-isolation
```

Cold builds take 20–60 min. Drop `MAX_JOBS` to 2 if `nvcc` OOMs. For faster dev iteration, disable head-dims you don't need (FA-3 example):

```bash
FLASHATTENTION_DISABLE_HDIM64=TRUE FLASHATTENTION_DISABLE_HDIM192=TRUE \
FLASHATTENTION_DISABLE_HDIM256=TRUE MAX_JOBS=4 pip install -e . --no-build-isolation
```

Verify the FA-3 install registered correctly:

```bash
python -c "import torch, sparse_flash_attn_3._C; print([o for o in dir(torch.ops.sparse_flash_attn_3) if not o.startswith('_')])"
```

The printed op list should include `fwd`, `bwd`, etc. If you see `RuntimeError: operator ... has already been registered`, an old `flash-attn` install is still in the venv and is colliding — uninstall it, or import the renamed package in a fresh process.

See `instruction.md` for the full build / rename reference.

---

## Usage

The new functions live in `hopper/flash_attn_interface.py` and are importable as `flash_attn_interface` after `cd hopper`.

### 1. Sparse-only forward — attend to a chosen set of blocks

```python
from flash_attn_interface import flash_attn_with_sparse_block_table

# q: (B, Sq, H, D)
# k, v: (B, Sk, Hk, D)         — Sk must be a multiple of block_size
# sparse_block_table: (N,) int32 CUDA — absolute block indices into k/v
out, lse = flash_attn_with_sparse_block_table(
    q, k, v, sparse_block_table, block_size=128,
)
# out: (B, Sq, H, D),  lse: (B, H, Sq)
```

Each entry of `sparse_block_table` indexes a block of `block_size` consecutive K/V tokens (physical offset = `block_idx * block_size`). Every selected block must be **fully historical** with respect to every Q row — the kernel applies no causal or local mask. Pass an empty table to get `out = 0, lse = -inf` (a neutral partial for combine).

### 2. Speculative + sparse decoding with exact combine

```python
from flash_attn_interface import flash_attn_speculative_sparse

# k_spec, v_spec: speculative dense K/V subset (must include the causal diagonal)
# k, v:           the full K/V that sparse_block_table indexes into
# sparse_block_table: (N,) int32 — the *missed* blocks not covered by k_spec/v_spec
out, lse = flash_attn_speculative_sparse(
    q, k_spec, v_spec, k, v, sparse_block_table, causal_spec=True,
)
# out: (B, Sq, H, D) in q's dtype,  lse: (B, H, Sq) fp32
```

Internally this runs FA over `(q, k_spec, v_spec)` with `causal_spec` masking, runs the sparse pass over the missed blocks (always unmasked), and merges the two `(o, lse)` partials through the standard `FlashAttnFwdCombine` kernel. The result is bit-equivalent (up to fp accumulation order) to attending over the union of both block sets.

### 3. Per-block partials (advanced)

For callers orchestrating their own combine (e.g. selecting different block subsets per query block), `flash_attn_with_sparse_block_table_partials()` returns one `(o_b, lse_b)` per entry in `sparse_block_table`; feed them into `flash_attn_combine` yourself.

---

## Repo layout

| Path | Contents |
|---|---|
| `hopper/` | FA-3 (CUTLASS 3.x, SM90 TMA+WGMMA+WS). **Sparse work lives here.** Python entry: `hopper/flash_attn_interface.py`. |
| `csrc/flash_attn/` | FA-2 kernels (CUTLASS 2.x). Sparse path in `flash_fwd_kernel.h` + `compute_attn_1rowblock_sparse`. |
| `sparse_flash_attn_2/` | FA-2 Python package (renamed from upstream `flash_attn/`). |
| `tests/`, `benchmarks/`, `examples/`, `training/` | Correctness tests, perf benchmarks, example scripts, training reference code. |

---

## Testing

```bash
# Reference combine identity — CPU, no GPU needed
pytest tests/test_sparse_combine_identity.py -v

# FA-3 sparse path — requires an SM90 GPU
pytest hopper/test_flash_attn.py -k sparse

# FA-2 regression
pytest tests/test_flash_attn.py
```

---

## Citations

If you use this work, please also cite the upstream FlashAttention papers it is built on:

```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={NeurIPS},
  year={2022}
}

@article{dao2023flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  year={2023}
}

@article{shah2024flashattention3,
  title={FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision},
  author={Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  year={2024}
}
```

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://tridao.me/publications/flash2/flash2.pdf
- FlashAttention-3 paper: https://tridao.me/publications/flash3/flash3.pdf

## License

BSD-3-Clause, inherited from upstream FlashAttention. See `LICENSE`.
