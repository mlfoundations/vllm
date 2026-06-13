# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 repro harness: DCP GQA + FlashAttention LSE-combine parity.

Reproduces the decode-side Decode-Context-Parallel (DCP) cross-rank combine
defect on SYNTHETIC per-rank GQA partial ``(attn_out, lse)`` tensors, calling
the REAL fork combine functions (``cp_lse_ag_out_rs`` / ``_cp_lse_common`` /
``correct_attn_out`` in ``vllm/v1/attention/ops/common.py``) — WITHOUT a full
model. The collectives are emulated single-process by ``FakeCPGroup``.

Contract under test (the ``dcp=N`` vs ``dcp=1`` invariant): the DCP-combined
attention output over a KV history sharded across ``N`` ranks must equal the
full unsharded attention over the same KV. This harness EXPECTS the GQA case to
DIVERGE at Stage 0 (that's the repro); the MHA case is a control that should
match (confirms the harness + reference are correct).

Run on GPU (Triton ``_correct_attn_cp_out_kernel`` needs CUDA):
    VLLM_USE_FLASHINFER_SAMPLER=0 \
      python -m pytest tests/dcp_gqa_repro/test_dcp_combine_parity.py -v -s

Or as a standalone script (prints the divergence table):
    VLLM_USE_FLASHINFER_SAMPLER=0 python tests/dcp_gqa_repro/test_dcp_combine_parity.py
"""

from __future__ import annotations

import itertools

import pytest
import torch

from tests.dcp_gqa_repro._dcp_ref import (
    FakeCPGroup,
    ref_attention,
    shard_attention,
)

# Real fork combine code — a fix to these flips FAIL -> PASS without touching
# this harness.
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs


def _dcp_combine(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    scale: float,
    q_per_kv: int,
    H: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate ``_forward_with_dcp`` context-combine for the whole KV history
    sharded across ``N`` DCP ranks, returning rank-0's local-head output
    ``[B, H, D]`` after the real ``cp_lse_ag_out_rs`` combine.
    """
    B, _, D = q.shape
    S = k_full.shape[1]
    assert S % N == 0, "KV history must split evenly across ranks for the repro"

    # --- query all_gather(dim=1): rank-major, [B, H*N, D] ---
    # Every rank holds the same local query heads; the gathered workspace tiles
    # them rank-major (rank0 heads, rank1 heads, ...).
    q_gathered = q.repeat(1, N, 1).contiguous()  # [B, H*N, D]

    # --- shard the KV history along the token dim ---
    k_shards = list(torch.chunk(k_full, N, dim=1))
    v_shards = list(torch.chunk(v_full, N, dim=1))

    # --- per-rank partial FA over its shard (FA returns lse in [Hg, B]) ---
    # NOTE: pass ctx=None into every correct_attn_out call so each launch goes
    # through the full kernel[grid](*args, **constexpr) warmup. A *shared*
    # CPTritonContext caches inner_kernel and replays it WITHOUT the constexpr
    # args, which mismatches when HEAD_DIM/N_ROUNDED change across cases (Triton
    # "takes 27 arguments (24 given)").
    per_rank_out: list[torch.Tensor] = []
    per_rank_lse_hb: list[torch.Tensor] = []
    Hg = H * N
    for r in range(N):
        out_r, lse_r = shard_attention(
            q_gathered, k_shards[r], v_shards[r], scale, q_per_kv
        )
        per_rank_out.append(out_r.to(dtype))  # [B, Hg, D]
        per_rank_lse_hb.append(lse_r.to(dtype))  # [Hg, B]

    # The combine wants LSE as [B, Hg] (the `context_lse.transpose(0,1)` handoff
    # at flash_attn.py:957). Build the per-rank [B, Hg] LSE list the LSE
    # all_gather inside _cp_lse_common will stitch.
    per_rank_lse_bh = [lse.transpose(0, 1).contiguous() for lse in per_rank_lse_hb]

    # --- run the REAL combine end-to-end for rank 0 ---
    # cp_lse_ag_out_rs(out, lse, group) does: _cp_lse_common (LSE all_gather +
    # correct_attn_out on rank-0's out) then reduce_scatter(out, dim=1). The
    # reduce_scatter must see EVERY rank's corrected out, so we pre-correct each
    # rank's out (each shares the same gathered LSE set) via the real
    # _cp_lse_common and register them; the FakeCPGroup's reduce_scatter then
    # sums + slices. This calls the genuine fork code throughout.
    from vllm.v1.attention.ops.common import _cp_lse_common

    corrected_per_rank: list[torch.Tensor] = []
    for r in range(N):
        g_r = FakeCPGroup(world_size=N, rank_in_group=r)
        g_r.register_all_gather(0, per_rank_lse_bh)
        out_cor_r, _lse_cor_r = _cp_lse_common(
            per_rank_out[r].clone(), per_rank_lse_bh[r], g_r, ctx=None
        )
        corrected_per_rank.append(out_cor_r)  # [B, Hg, D]

    group = FakeCPGroup(world_size=N, rank_in_group=0)
    group.register_all_gather(0, per_rank_lse_bh)
    group.register_reduce_scatter(0, corrected_per_rank)
    # Rank-0 driven through the REAL cp_lse_ag_out_rs (its _cp_lse_common
    # re-corrects rank-0's out identically; its reduce_scatter uses our
    # registered per-rank corrected outs).
    out0, _lse0 = cp_lse_ag_out_rs(
        per_rank_out[0].clone(), per_rank_lse_bh[0], group, ctx=None, return_lse=True
    )
    return out0.float()  # [B, H, D]


CONFIGS = list(
    itertools.product(
        [(12, 2), (16, 2), (12, 12)],  # (H, H_kv); last is MHA control
        [2, 4],  # N = dcp_world_size
        [16, 256],  # S = KV history length (short, long)
    )
)


def _run_case(H, H_kv, N, S, device, seed=0):
    torch.manual_seed(seed)
    D = 128
    B = 4
    q_per_kv = H // H_kv
    scale = 1.0 / (D**0.5)
    dtype = torch.bfloat16

    q = torch.randn(B, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H_kv, D, device=device, dtype=dtype)

    ref_out, _ = ref_attention(q, k, v, scale)  # [B,H,D] fp32
    dcp_out = _dcp_combine(q, k, v, scale, q_per_kv, H, N, device, dtype)

    delta = (dcp_out - ref_out).abs()
    max_d = delta.max().item()
    # first divergence (token, head) above a generous bf16 tol
    tol = 5e-2
    idx = (delta.amax(dim=-1) > tol).nonzero()  # [num, 2] -> (b, h)
    first = tuple(idx[0].tolist()) if idx.numel() else None
    # greedy-argmax proxy: flatten head*D as "logits", check argmax match
    ref_arg = ref_out.reshape(B, -1).argmax(dim=-1)
    dcp_arg = dcp_out.reshape(B, -1).argmax(dim=-1)
    argmax_flips = int((ref_arg != dcp_arg).sum().item())
    return max_d, first, argmax_flips


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")
@pytest.mark.parametrize("HH,N,S", CONFIGS)
def test_dcp_combine_parity(HH, N, S):
    H, H_kv = HH
    device = torch.device("cuda")
    max_d, first, flips = _run_case(H, H_kv, N, S, device)
    is_mha = H == H_kv
    print(
        f"\n[H={H} H_kv={H_kv} N={N} S={S}] max|Δ|={max_d:.4e} "
        f"first_div(b,h)={first} argmax_flips={flips}"
    )
    if is_mha:
        # MHA control: combine should match the reference within bf16 tol.
        assert max_d < 5e-2, (
            f"MHA control DIVERGED (max|Δ|={max_d:.4e}); harness/LSE math is "
            f"suspect, not GQA-specific."
        )
    else:
        # GQA: Stage 0 EXPECTS divergence (this assert is the repro). If it
        # passes, the synthetic combine did NOT reproduce the defect.
        assert max_d < 5e-2, (
            f"GQA REPRO: DCP combine diverges from reference "
            f"(max|Δ|={max_d:.4e}, first_div={first}, argmax_flips={flips})"
        )


def _main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"{'H':>3} {'Hkv':>3} {'N':>2} {'S':>4} "
          f"{'max|Δ|':>12} {'first(b,h)':>12} {'argmax_flips':>12} {'verdict':>8}")
    for (H, H_kv), N, S in CONFIGS:
        try:
            max_d, first, flips = _run_case(H, H_kv, N, S, device)
        except Exception as e:  # Triton kernel unavailable on CPU
            print(f"{H:>3} {H_kv:>3} {N:>2} {S:>4}  ERROR: {e}")
            continue
        verdict = "MATCH" if max_d < 5e-2 else "DIVERGE"
        print(f"{H:>3} {H_kv:>3} {N:>2} {S:>4} {max_d:>12.4e} "
              f"{str(first):>12} {flips:>12} {verdict:>8}")


if __name__ == "__main__":
    _main()
