"""Numerical validation of the speculative sparse attention combine identity.

For any query attending to a set of K/V blocks S = A ∪ B (disjoint), the
dense attention output over S must equal the online-softmax merge of two
partial attentions computed over A and B independently:

    FA(S) == Combine( FA(A), FA(B) )

This test verifies the identity in fp64 using a pure-Python reference
implementation. It does not require a GPU and does not depend on any CUDA
kernel; it is the prerequisite sanity check from plan.md before touching
flash-attention kernel code.
"""

import pytest
import torch


def reference_attention(q, k, v, scale):
    """Dense scaled dot-product attention. Returns (O, LSE).

    Shapes:
        q: [B, H, Q, D]
        k: [B, H, K, D]
        v: [B, H, K, D]
    Returns:
        o:   [B, H, Q, D]
        lse: [B, H, Q]   (natural log)
    """
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    m = s.max(dim=-1, keepdim=True).values
    p = (s - m).exp()
    l = p.sum(dim=-1, keepdim=True)
    o = torch.einsum("bhqk,bhkd->bhqd", p / l, v)
    lse = (m + l.log()).squeeze(-1)
    return o, lse


def combine_partials(o1, lse1, o2, lse2):
    """Merge two (O, LSE) partials using the online softmax identity.

    This is the math FlashAttnFwdCombine implements on GPU:

        new_lse = log(exp(lse1) + exp(lse2))
        new_O   = (exp(lse1)*O1 + exp(lse2)*O2) / (exp(lse1) + exp(lse2))

    Computed via the lse_max trick for numerical stability.
    """
    lse_max = torch.maximum(lse1, lse2)
    w1 = (lse1 - lse_max).exp()
    w2 = (lse2 - lse_max).exp()
    denom = w1 + w2
    o = (w1.unsqueeze(-1) * o1 + w2.unsqueeze(-1) * o2) / denom.unsqueeze(-1)
    lse = lse_max + denom.log()
    return o, lse


def gather_blocks(x, block_indices, block_size):
    """Gather K or V blocks in the order given by block_indices."""
    return torch.cat(
        [x[..., i * block_size : (i + 1) * block_size, :] for i in block_indices],
        dim=-2,
    )


@pytest.mark.parametrize("seqlen_q", [1, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize(
    "split",
    [
        ([0, 2, 5, 7], [1, 3, 4, 6]),
        ([0], [1, 2, 3, 4, 5, 6, 7]),
        ([0, 1, 2, 3, 4, 5, 6], [7]),
        ([2, 3], [0, 1, 4, 5, 6, 7]),
    ],
    ids=["interleaved", "spec_heavy", "miss_heavy", "contiguous_miss"],
)
def test_combine_identity_matches_dense(seqlen_q, block_size, split):
    torch.manual_seed(0)
    B, H, D = 2, 4, 64
    n_blocks = 8
    seqlen_k = n_blocks * block_size
    scale = D ** -0.5

    dtype = torch.float64
    q = torch.randn(B, H, seqlen_q, D, dtype=dtype)
    k = torch.randn(B, H, seqlen_k, D, dtype=dtype)
    v = torch.randn(B, H, seqlen_k, D, dtype=dtype)

    a_idx, b_idx = split
    assert set(a_idx).isdisjoint(set(b_idx))

    union_idx = sorted(a_idx + b_idx)
    k_union = gather_blocks(k, union_idx, block_size)
    v_union = gather_blocks(v, union_idx, block_size)
    o_ref, lse_ref = reference_attention(q, k_union, v_union, scale)

    k_a = gather_blocks(k, a_idx, block_size)
    v_a = gather_blocks(v, a_idx, block_size)
    o_a, lse_a = reference_attention(q, k_a, v_a, scale)

    k_b = gather_blocks(k, b_idx, block_size)
    v_b = gather_blocks(v, b_idx, block_size)
    o_b, lse_b = reference_attention(q, k_b, v_b, scale)

    o_combined, lse_combined = combine_partials(o_a, lse_a, o_b, lse_b)

    torch.testing.assert_close(o_combined, o_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(lse_combined, lse_ref, atol=1e-10, rtol=1e-10)


def test_combine_is_order_invariant():
    """Combine(A, B) must equal Combine(B, A) — the merge is symmetric."""
    torch.manual_seed(1)
    shape = (2, 4, 4)
    o_a = torch.randn(*shape, 64, dtype=torch.float64)
    o_b = torch.randn(*shape, 64, dtype=torch.float64)
    lse_a = torch.randn(*shape, dtype=torch.float64)
    lse_b = torch.randn(*shape, dtype=torch.float64)

    o_ab, lse_ab = combine_partials(o_a, lse_a, o_b, lse_b)
    o_ba, lse_ba = combine_partials(o_b, lse_b, o_a, lse_a)

    torch.testing.assert_close(o_ab, o_ba)
    torch.testing.assert_close(lse_ab, lse_ba)


def test_combine_with_empty_partial_is_identity():
    """Combine(X, empty_partial) must equal X.

    An empty partial has lse = -inf (no contribution). The combine kernel
    must treat it as a pass-through.
    """
    torch.manual_seed(2)
    shape = (2, 4, 4)
    o_x = torch.randn(*shape, 64, dtype=torch.float64)
    lse_x = torch.randn(*shape, dtype=torch.float64)

    o_empty = torch.zeros_like(o_x)
    lse_empty = torch.full_like(lse_x, float("-inf"))

    o_out, lse_out = combine_partials(o_x, lse_x, o_empty, lse_empty)
    torch.testing.assert_close(o_out, o_x)
    torch.testing.assert_close(lse_out, lse_x)
