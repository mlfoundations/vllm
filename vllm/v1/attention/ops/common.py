# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os as _os

import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.triton_utils import tl, triton

# Stage-1 DCP root-cause instrumentation (env-gated, default OFF). When
# SKYRL_DCP_DEBUG=1, _cp_lse_common dumps the EXACT `lses` tensor handed to
# correct_attn_out (shape/strides) + a pure-torch replication of the kernel's
# per-cell rescale factor so we can read off whether the LSE stride walk lands
# on the right (rank, B, H) cells. Inert unless the env var is set.
_SKYRL_DCP_DEBUG = _os.environ.get("SKYRL_DCP_DEBUG", "0") == "1"
_skyrl_common_calls = 0


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    # correct output
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N, "IS_BASE_E": is_lse_base_on_e}
    global _skyrl_common_calls
    _dbg_corr = _SKYRL_DCP_DEBUG and _skyrl_common_calls < 60
    if _dbg_corr:
        _out_before = out.detach().float().clone()
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    if _dbg_corr:
        import logging

        _lg = logging.getLogger("vllm.v1.attention.ops.common")
        # ACTUAL factor the kernel applied = out_after / out_before (per cell).
        _after = out.detach().float()
        _ratio = torch.where(
            _out_before.abs() > 1e-6, _after / _out_before, torch.ones_like(_after)
        )
        # Expected factor from the SAME lses (correct online-softmax weight).
        lf = lses.float()
        safe = torch.where(torch.isfinite(lf), lf, torch.full_like(lf, -1e30))
        lse_g_t = torch.logsumexp(safe, dim=0)  # [B,H]
        exp_factor = torch.exp(safe[cp_rank] - lse_g_t)  # [B,H]
        fin = torch.isfinite(exp_factor) & (exp_factor < 0.999)
        if fin.any():
            ef = exp_factor[fin]  # cells where a non-trivial weight is expected
            # applied ratio at those (b,h), broadcast over D -> take [...,0]
            af = _ratio[..., 0][fin]
            _lg.info(
                "[SKYRL_DCP_DEBUG] (corr) cp_rank=%d cells-with-expected-weight<1: "
                "n=%d | EXPECTED factor mean=%.4f min=%.4f | KERNEL-APPLIED ratio "
                "mean=%.4f min=%.4f max=%.4f (if APPLIED~1 while EXPECTED<1 ⇒ the "
                "rescale is NOT landing on the output)",
                cp_rank,
                int(fin.sum().item()),
                ef.mean().item(),
                ef.min().item(),
                af.mean().item(),
                af.min().item(),
                af.max().item(),
            )
    return out, lse


def _cp_lse_common(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    global _skyrl_common_calls
    if _SKYRL_DCP_DEBUG and _skyrl_common_calls < 60:
        _skyrl_common_calls += 1
        import logging

        _lg = logging.getLogger("vllm.v1.attention.ops.common")
        r = cp_group.rank_in_group
        # Raw per-(n) LSE at b=0,h=r*cp_h (this rank's first own head) to read
        # whether the gathered cross-rank LSE rows are REAL or -inf/garbage.
        try:
            cp_h_dbg = lses.shape[2] // cp_group.world_size
            Hg_dbg = lses.shape[2]
            # FULL lses[:,0,:] table: row n = rank n's LSE for EVERY head-slot.
            # The per-head-slot global = logsumexp over n; the correct weight for
            # head-slot h on rank r is exp(lses[r,0,h]-global_h). Reading the
            # table shows whether head-slots OUTSIDE rank r's own block get a
            # weight!=1 (they must, for the cross-rank sum to be a softmax).
            tbl = [
                [round(lses[nn, 0, hh].item(), 3) for hh in range(Hg_dbg)]
                for nn in range(cp_group.world_size)
            ]
            _lg.info(
                "[SKYRL_DCP_DEBUG] (common-LSEtab) rank=%d cp_h=%d Hg=%d "
                "lses[n,0,:] per n = %s (own block = [%d:%d])",
                r,
                cp_h_dbg,
                Hg_dbg,
                tbl,
                r * cp_h_dbg,
                (r + 1) * cp_h_dbg,
            )
        except Exception as e:
            _lg.info("[SKYRL_DCP_DEBUG] (common-LSE) err %s", e)
        # Replicate the kernel's per-cell factor in pure torch using the SAME
        # `lses` tensor (so any stride/orientation defect shows identically):
        #   lse_global = logsumexp_n(lses[n,b,h]);  factor = exp(lses[r,b,h]-lse_g)
        lf = lses.float()  # [N,B,H]
        safe = torch.where(torch.isfinite(lf), lf, torch.full_like(lf, -1e30))
        lse_g = torch.logsumexp(safe, dim=0)  # [B,H]
        factor = torch.exp(safe[r] - lse_g)  # [B,H]
        # Where the gathered cp_attn_lse rows actually came from: compare lses[r]
        # against THIS rank's own cp_attn_lse (must be equal if reshape is right).
        own = cp_attn_lse.float()
        own_vs_slice = (lses[r].float() - own).abs().max().item()
        _lg.info(
            "[SKYRL_DCP_DEBUG] (common) rank=%d N=%d lses.shape=%s "
            "lses.stride=%s cp_attn_lse.shape=%s | lses[r]==own? max|Δ|=%.3e | "
            "factor(this rank) min=%.4f max=%.4f mean=%.4f | factor==1 frac=%.3f",
            r,
            cp_group.world_size,
            tuple(lses.shape),
            tuple(lses.stride()),
            tuple(cp_attn_lse.shape),
            own_vs_slice,
            factor[torch.isfinite(factor)].min().item()
            if torch.isfinite(factor).any()
            else float("nan"),
            factor[torch.isfinite(factor)].max().item()
            if torch.isfinite(factor).any()
            else float("nan"),
            factor[torch.isfinite(factor)].mean().item()
            if torch.isfinite(factor).any()
            else float("nan"),
            ((factor > 0.999).float().mean().item()),
        )
    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
    out_fp32: bool = False,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]

    out_fp32: if True, return the combined output in fp32 instead of downcasting
        back to ``cp_attn_out``'s dtype. The cross-rank online-softmax recombine is
        always accumulated in fp32 (see below); ``out_fp32=True`` additionally skips
        the final downcast so a caller that immediately blends the result with
        another partial (e.g. the FlashAttention DCP context+self merge) can keep the
        whole finish in fp32 and quantize only once, at the final attention output.
    """
    # Stage-2 UNIFIED single-call probe (SKYRL_DCP_DEBUG2=1, default OFF). Builds
    # the FULLY-CORRECT reference (gather every rank's RAW cp_attn_out + cp_attn_lse,
    # per-slot online-softmax combine, then take this rank's reduce_scatter block)
    # and diffs it against the ACTUAL post-reduce_scatter kernel output — all from
    # the SAME tensors on the SAME rank in ONE call (eliminates the cross-run
    # ambiguity of the Stage-1 dumps). Inert unless the env var is set.
    if (
        _os.environ.get("SKYRL_DCP_DEBUG2", "0") == "1"
        and cp_group.world_size > 1
        and getattr(cp_lse_ag_out_rs, "_dbg2_n", 0) < 24
    ):
        import logging as _logging

        _lg2 = _logging.getLogger("vllm.v1.attention.ops.common")
        try:
            w = cp_group.world_size
            r = cp_group.rank_in_group
            raw = cp_attn_out.contiguous()  # [B,Hg,D] this rank's RAW FA out
            lse_in = cp_attn_lse.contiguous()  # [B,Hg] this rank's RAW FA lse
            raw_all = cp_group.all_gather(raw, dim=0)  # [w*B,Hg,D]
            lse_all = cp_group.all_gather(lse_in, dim=0)  # [w*B,Hg]
            B = raw.shape[0]
            raw_s = torch.stack(
                [raw_all[i * B : (i + 1) * B].float() for i in range(w)], 0
            )  # [w,B,Hg,D]
            lse_s = torch.stack(
                [lse_all[i * B : (i + 1) * B].float() for i in range(w)], 0
            )  # [w,B,Hg]
            safe = torch.where(
                torch.isfinite(lse_s), lse_s, torch.full_like(lse_s, -1e30)
            )
            g = torch.logsumexp(safe, dim=0)  # [B,Hg]
            wts = torch.exp(safe - g.unsqueeze(0)).unsqueeze(-1)  # [w,B,Hg,1]
            ref_full = (raw_s * wts).sum(0)  # [B,Hg,D] correct combined, all slots
            cp_h = raw.shape[1] // w
            ref_block = ref_full[:, r * cp_h : (r + 1) * cp_h, :]  # this rank's rs block
            _ref_block = ref_block.detach().clone()
            cp_lse_ag_out_rs._dbg2_n = getattr(cp_lse_ag_out_rs, "_dbg2_n", 0) + 1
            cp_lse_ag_out_rs._dbg2_ref = _ref_block
        except Exception as e:  # pragma: no cover - debug only
            _lg2.info("[SKYRL_DCP_DEBUG2] (pre) err %s", e)
            cp_lse_ag_out_rs._dbg2_ref = None
    # GQA-LSE DCP combine fix: accumulate the cross-rank online-softmax recombine
    # in fp32, not in the bf16/fp16 attention dtype, AND return the combined context
    # in fp32 so the downstream merge_attn_states (context + new-token self term) can
    # also run without a bf16 round-trip. Rationale:
    #   * correct_attn_out rescales each rank's partial by exp(lse_r - lse_global) and
    #     reduce_scatter SUMS those rescaled partials across ranks. Doing that
    #     rescale+sum in bf16 (the previous behavior, since cp_attn_out is the model
    #     dtype) loses precision; FlashAttention itself accumulates the softmax in
    #     fp32, and the A2A sibling combine (_dcp_a2a_unpack_combine_kernel) already
    #     accumulates in an fp32 register, so AG+RS was the only DCP combine doing a
    #     bf16 cross-rank sum.
    #   * Downcasting the *combined* context back to bf16 before the merge re-injects a
    #     ~4e-2 quantization (measured) at exactly the value the merge then blends with
    #     the self term — enough, under a 128-expert top-8 MoE router, to flip top-k
    #     and then greedy tokens vs dcp=1. Keeping the combined context fp32 lets the
    #     caller merge in fp32 and downcast only once, at the final attention output —
    #     matching the single fp32-accumulated FA call of the dcp=1 path.
    # The returned LSE is already fp32 (FA emits fp32 LSE).
    _orig_dtype = cp_attn_out.dtype
    if _orig_dtype != torch.float32:
        cp_attn_out = cp_attn_out.float()
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    global _skyrl_common_calls
    _dbg_rs = _SKYRL_DCP_DEBUG and _skyrl_common_calls <= 60
    if _dbg_rs:
        _out_pre_rs = out.detach().float().clone()  # [B,Hg,D] this rank's weighted out
    out = cp_group.reduce_scatter(out, dim=1)
    # Downcast to the input dtype unless the caller wants to keep fp32 for a
    # subsequent fp32 merge (out_fp32=True). Default preserves the prior contract
    # for all other DCP callers (FlashInfer, MLA).
    if not out_fp32 and out.dtype != _orig_dtype:
        out = out.to(_orig_dtype)
    if (
        _os.environ.get("SKYRL_DCP_DEBUG2", "0") == "1"
        and getattr(cp_lse_ag_out_rs, "_dbg2_ref", None) is not None
    ):
        import logging as _logging

        _lg2 = _logging.getLogger("vllm.v1.attention.ops.common")
        try:
            _ref = cp_lse_ag_out_rs._dbg2_ref
            cp_lse_ag_out_rs._dbg2_ref = None
            _k = out.detach().float()
            fin = torch.isfinite(_ref).all(-1) & torch.isfinite(_k).all(-1)
            d = (_ref[fin] - _k[fin]).abs().max().item() if fin.any() else float("nan")
            # also report the unweighted-sum baseline: if kernel == raw-sum, d_unw ~ 0
            _lg2.info(
                "[SKYRL_DCP_DEBUG2] (post) rank=%d UNIFIED max|kernel - "
                "correct-online-softmax| = %.4e (SMALL ⇒ combine CORRECT; "
                "LARGE ⇒ combine WRONG). kernel[0,0,0]=%.6f ref[0,0,0]=%.6f",
                cp_group.rank_in_group,
                d,
                _k.reshape(-1)[0].item() if _k.numel() else float("nan"),
                _ref.reshape(-1)[0].item() if _ref.numel() else float("nan"),
            )
        except Exception as e:  # pragma: no cover - debug only
            _lg2.info("[SKYRL_DCP_DEBUG2] (post) err %s", e)
    if _dbg_rs:
        import logging

        _lg = logging.getLogger("vllm.v1.attention.ops.common")
        r = cp_group.rank_in_group
        w = cp_group.world_size
        cp_h = _out_pre_rs.shape[1] // w
        # FULL pre-rs weighted output across ALL Hg heads at b=0,d=0, plus the
        # post-reduce_scatter value for this rank's own head 0. reduce_scatter
        # output head h on rank r = sum_s in_s[r*cp_h + h]; so post[0] should be
        # sum over ranks of in_s[r*cp_h]. We log each rank's full row so the two
        # ranks' logs can be cross-summed by hand.
        pre_row = [_out_pre_rs[0, h, 0].item() for h in range(_out_pre_rs.shape[1])]
        post = out.detach().float()[0, 0, 0].item()  # after rs, this rank head 0
        _lg.info(
            "[SKYRL_DCP_DEBUG] (rs) rank=%d cp_h=%d pre-rs WEIGHTED out[0,:,0]=%s "
            "| post-rs out[0,0,0]=%.6f (post[r] = sum_s pre_s[0, r*cp_h, 0]; "
            "cross-ref the other rank's pre-rs row at index r*cp_h=%d)",
            r,
            cp_h,
            ["%.5f" % x for x in pre_row],
            post,
            r * cp_h,
        )
    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out


def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    # Same fp32-accumulation fix as cp_lse_ag_out_rs: the per-rank rescale +
    # cross-rank all_reduce sum of the online-softmax recombine must accumulate in
    # fp32 (matching FlashAttention's internal accumulation and the A2A combine),
    # not in the bf16/fp16 model dtype, to avoid a router-flipping combine error.
    _orig_dtype = cp_attn_out.dtype
    if _orig_dtype != torch.float32:
        cp_attn_out = cp_attn_out.float()
    out, lse = _cp_lse_common(
        cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, is_lse_base_on_e=is_lse_base_on_e
    )
    out = cp_group.all_reduce(out)
    if out.dtype != _orig_dtype:
        out = out.to(_orig_dtype)

    if return_lse:
        return out, lse
    return out


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Compute start index and sequence length from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax

    # compute input row indices for valid (b, t)
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # Pointers
    # x_ptr: row-major [N, D]
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

    # out_ptr: row-major [B, Lmax, D]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # Initialize with PAD. PAD_IS_UINT8 selects the pad tensor's dtype so
    # integer-typed outputs (e.g. MXFP4 packed nibbles, ue8m0 scale bytes)
    # get an exact-byte pad rather than going through an fp32→uint8 cast
    # that's implementation-defined outside of value 0.
    d_mask = off_d[None, :] < D
    if PAD_IS_UINT8:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
    else:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Pack sequences of different lengths into a batched tensor.

    Supports float dtypes (any, via fp32 pad) and ``torch.uint8`` (exact-byte
    pad — e.g. MXFP4 packed nibbles or ue8m0 scale bytes). For uint8 inputs
    ``pad_value`` must be an integer in ``[0, 255]``.

    Args:
        x: [N, ...] — input tensor where N is total number of tokens.
        lengths: [B] — sequence lengths for each batch.
        pad_value: value to use for padding. Defaults to ``-inf`` which is
            only sensible for float dtypes; pass ``0`` (or any byte) for
            uint8 inputs.
        block_t: block size for time dimension.
        block_d: block size for feature dimension.

    Returns:
        packed: [B, Lmax, ...] — packed tensor.
    """
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255, (
            f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        )
        pad_constexpr: int | float = int(pad_value)
    else:
        pad_constexpr = float(pad_value)

    # Handle multi-dimensional input by reshaping to (N, -1)
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=is_uint8,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out


@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # bounds: compute start from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    # compute output row indices for valid (b, t)
    out_row = in_start + off_t

    # Pointers
    # packed_ptr: row-major [B, Lmax, D]
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # out_ptr: row-major [N, D]
    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Efficient Triton implementation.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch
        block_t: block size for time dimension
        block_d: block size for feature dimension

    Returns:
        unpacked_tensor: [N, ...] where N = sum(lengths)
    """

    # Handle multi-dimensional input by reshaping to (B, Lmax, -1)
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # Calculate total number of elements
    N = int(lengths.sum().item())

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        lengths.int(),
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out
