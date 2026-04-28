# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This repository contains two generations of FlashAttention CUDA kernels. Active work here targets **Hopper (SM90)**:

- **FA-2** (`csrc/flash_attn/`) — CUTLASS 2.x-based, C++/CUDA. Supports SM80 (Ampere) and SM90 (Hopper). Compiled ahead-of-time into per-(hdim, dtype, causal) `.cu` instantiations.
- **FA-3** (`hopper/`) — CUTLASS 3.x-based, uses TMA + WGMMA + warp-specialization. Targets SM90 (Hopper) as the primary architecture.

FA-4 (`flash_attn/cute/`) exists but is not the focus here.

## Build & Install

**FA-2** (from repo root):
```bash
pip install flash-attn --no-build-isolation
# or with parallelism:
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**FA-3** (from `hopper/`):
```bash
cd hopper && pip install . --no-build-isolation
```

Dependencies: `torch`, `ninja`, `packaging`, `psutil`, CUDA toolkit with `nvcc`.

## Running Tests

**FA-2:**
```bash
pytest tests/test_flash_attn.py
pytest tests/test_flash_attn.py -k "test_flash_attn_output" -x
```

**FA-3:**
```bash
pytest hopper/test_flash_attn.py
pytest hopper/test_flash_attn.py -k "test_flash_attn_output" -x
pytest hopper/test_kvcache.py
```

If you get OOM errors, use `nvidia-smi` to find a free GPU and select it with `CUDA_VISIBLE_DEVICES=<id>`.

## Code Architecture

### FA-2 (`csrc/flash_attn/src/`)

| File | Role |
|---|---|
| `flash.h` | Core `Flash_params` struct (pointers, strides, dimensions) |
| `kernel_traits.h` | `Flash_fwd_kernel_traits` / `Flash_bwd_kernel_traits`: tile shape (kBlockM/N), warp count, smem layout |
| `flash_fwd_kernel.h` | Forward kernel: online softmax loop over K/V tiles |
| `flash_bwd_kernel.h` | Backward kernel |
| `flash_fwd_launch_template.h` | Grid/block launch logic for forward |
| `flash_bwd_launch_template.h` | Grid/block launch logic for backward |
| `flash_bwd_preprocess_kernel.h` | Backward preprocess (dO * O rowsum) |
| `softmax.h` | Online softmax: `Softmax` struct with `scale_apply_exp2` |
| `mask.h` | Causal and local/sliding-window masking |
| `block_info.h` | `BlockInfo`: per-block M/N tile range accounting for padding and causal |
| `utils.h` | Warp reductions, predicates, async copy helpers |
| `flash_api.cpp` | Python binding entry point (pybind11) |
| `src/flash_fwd_hdim*_*.cu` | Instantiation files — one per (hdim, dtype, causal) combination |
| `src/flash_bwd_hdim*_*.cu` | Backward instantiation files |
| `src/generate_kernels.py` | Script that regenerates the `.cu` instantiation files |

**Key pattern:** Tile config is baked into `kernel_traits.h` via template parameters. To add a new tile shape or head dim, add entries in `generate_kernels.py` and re-run it, then rebuild.

### FA-3 (`hopper/`)

| File | Role |
|---|---|
| `flash.h` | Core `Flash_params` / `Flash_fwd_params` / `Flash_bwd_params` structs |
| `flash_fwd_kernel_sm90.h` | SM90 forward kernel entry (warp-specialized: producer/consumer warps) |
| `flash_bwd_kernel_sm90.h` | SM90 backward kernel entry |
| `flash_fwd_kernel_sm80.h` | SM80 fallback forward kernel |
| `flash_bwd_kernel_sm80.h` | SM80 fallback backward kernel |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` | **Core FA-3 Hopper forward**: `CollectiveMainloopFwdSm90` — TMA loads (K, V, Q), WGMMA, pipelined K/V stages |
| `mainloop_bwd_sm90_tma_gmma_ws.hpp` | **Core FA-3 Hopper backward**: `CollectiveMainloopBwdSm90` |
| `mainloop_fwd_sm80.hpp` | Ampere fallback forward mainloop |
| `mainloop_bwd_sm80.hpp` | Ampere fallback backward mainloop |
| `epilogue_fwd.hpp` | Forward epilogue: write O and LSE to global memory |
| `epilogue_bwd.hpp` | Backward epilogue: write dQ, dK, dV |
| `softmax.h` | Online softmax with running max/sum |
| `mask.h` | Causal, local window, and custom mask support |
| `seqlen.h` | Variable-length sequence bookkeeping |
| `block.h` | Block/tile offset and range computation |
| `paged_kv.h` | Paged KV cache support |
| `pack_gqa.h` | GQA head packing |
| `tile_scheduler.hpp` | Tile scheduling (static and varlen-aware) |
| `named_barrier.hpp` | Named barrier enums for warp-group synchronization |
| `sm90_pipeline_no_cluster.hpp` | Custom pipeline state for non-cluster TMA |
| `flash_fwd_launch_template.h` | Forward launch: selects SM90 vs SM80 path, sets grid/block |
| `flash_bwd_launch_template.h` | Backward launch template |
| `flash_fwd_combine.cu` | SplitKV partial-result combine kernel |
| `flash_api.cpp` | Python binding entry point |
| `flash_attn_interface.py` | Python API: `flash_attn_func`, `flash_attn_varlen_func`, `flash_attn_with_kvcache` |
| `instantiations/` | Per-(hdim, dtype, causal) `.cu` instantiation files |
| `generate_kernels.py` | Regenerates `instantiations/` |

**Key pattern:** FA-3 uses warp-specialization — producer warps issue TMA loads while consumer warp-groups run WGMMA. The pipeline in `mainloop_fwd_sm90_tma_gmma_ws.hpp` coordinates them via `MainloopPipelineK` / `MainloopPipelineV` (CUTLASS `PipelineTmaAsync`). To change pipeline depth, modify the `Stages` template parameter.

## Key Differences FA-2 vs FA-3

| | FA-2 | FA-3 |
|---|---|---|
| Location | `csrc/flash_attn/` | `hopper/` |
| CUTLASS version | 2.x (`csrc/cutlass/`) | 3.x (system or bundled) |
| SM90 strategy | Standard CUDA threads | TMA + WGMMA + warp-specialization |
| Async copy | `cp.async` | TMA (`tma_load_K/V/Q`) |
| MMA | `mma.sync` (HMMA) | `wgmma.mma_async` |
| Pipeline | Manual smem double-buffering | CUTLASS `PipelineTmaAsync` |
| Instantiation | Per-`.cu` files in `src/` | Per-`.cu` files in `instantiations/` |

## Debugging GPU Kernels

- Use `printf` with thread guards to avoid flooding output:
  ```cpp
  if (threadIdx.x % 32 == 0 && blockIdx.x == 0) { printf(...); }
  ```
- `compute-sanitizer --tool=racecheck` — note: false positives with raw TMA (`cp.async.bulk`)
- `CUTE_DSL_KEEP_PTX=1` / `CUTE_DSL_LINEINFO=1` — for PTX inspection (FA-4, less relevant here)
- For FA-3 pipeline deadlocks: bisect with `printf` at `producer_acquire` / `consumer_wait` / `consumer_release` callsites to isolate which stage hangs
