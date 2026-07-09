# Q3 — Hardware dependency: roofline / utilization evidence

**Question (Reviewer uVhq #2).** *Discuss how the performance gains may change on future
hardware platforms with different compute-memory characteristics.*

`roofline_table.py` re-aggregates the Nsight Compute profiles from the DynamicSparseAttn
repo (`latency/triton/ncu_out_*`; GLM-4-9B, batch 1, Quest + InfLLM-v2, 16K–512K) plus the
paper's Table 6 batch sweep, and prints the idle-compute headroom that PRR exploits along
with a first-order projection of how it changes as compute outpaces bandwidth.

## Reproduce

```bash
# The NCU profiles live in the DynamicSparseAttn repo; point PRR_NCU_ROOT at them:
PRR_NCU_ROOT=/path/to/DynamicSparseAttn/latency/triton python roofline/roofline_table.py
```

The committed `utilization.csv` already contains the aggregated numbers, so the tables can
be reproduced without re-profiling.

## Findings

- **Decode uses ~13% of peak compute; ~87% of the SMs are idle** across all context
  lengths (batch 1). DRAM-bandwidth utilization is ~23–26%, so decode is memory-bound —
  the tensor cores sit idle while selection streams KV from HBM.
- **The SMs stay idle even at batch 16** (paper Table 6): SM 6.5–9.3%, i.e. ~90–93% idle.
- **Projection:** modelling long-context decode as memory-bound (runtime set by
  bytes/bandwidth), if a future GPU grows peak compute by `k×` relative to HBM bandwidth,
  the same attention work occupies `SM%/k` of peak, so idle headroom grows (≈93.5% at
  `k=2`, ≈96.8% at `k=4`). Because the compute:bandwidth gap has been widening — and
  low-precision tensor cores (fp8/fp4) push it further — the idle compute PRR monetizes is
  expected to persist or grow.

## Interpretation for the paper

PRR's gain is a function of the compute-memory gap, not of a specific GPU. On the
mainstream trajectory (compute outpacing bandwidth) the idle-compute headroom PRR exploits
widens, so gains are robust; they would erode only in a bandwidth-dominant regime (e.g.
processing-in-memory) that *also* removes the selection-to-attention serialization.

`utilization.csv` contains the tidy per-stage numbers behind these tables.
