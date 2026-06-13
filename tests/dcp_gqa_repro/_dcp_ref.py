# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference helpers for the Stage-0 DCP GQA LSE-combine repro harness.

This module is TEST-ONLY. It provides:
  * ``FakeCPGroup`` — a single-process stand-in for ``GroupCoordinator`` that
    emulates ``all_gather`` / ``reduce_scatter`` over a python list of per-rank
    tensors, matching vLLM's concat-style collective semantics
    (``vllm/distributed/device_communicators/base_device_communicator.py``).
  * ``ref_attention`` — a numerically-exact torch softmax attention that also
    returns the natural-log LSE, used as the ``dcp=1`` ground truth.
  * ``shard_attention`` — runs ``ref_attention`` over a single KV shard to
    produce the per-rank partial ``(attn_out [B,H,D], lse [H,B])`` exactly as
    ``flash_attn_varlen_func`` would (LSE returned in ``[H,B]`` orientation).

No vLLM *source* is modified; we only IMPORT the real combine functions in the
harness so a later fix flips the test from FAIL -> PASS.
"""

from __future__ import annotations

import torch


class FakeCPGroup:
    """Single-process emulation of a DCP ``GroupCoordinator``.

    Each "rank" calls collectives with its *own* local tensor. To emulate this
    in one process we register the per-rank tensors once and replay them.

    Two usage patterns mirror the real code:

    * ``all_gather(x, dim)``: in the real code every rank holds the SAME
      already-gathered query workspace heads, but produces a DIFFERENT local
      LSE/out. We model the collective as concat over the supplied per-rank
      list along ``dim`` (rank-major), matching the concat-style all_gather.

    * ``reduce_scatter(x, dim)``: sum the per-rank tensors then take this rank's
      chunk along ``dim``.

    The combine functions (``cp_lse_ag_out_rs`` / ``_cp_lse_common``) call
    ``all_gather`` on the LSE and ``reduce_scatter`` on the corrected output.
    Because every rank's call must see all ranks' data, we stash the per-rank
    payloads on the group and the collective stitches them.
    """

    def __init__(self, world_size: int, rank_in_group: int):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        # per-collective-call registries, keyed by an integer call-id so that
        # successive collectives in one combine don't collide.
        self._ag_payloads: dict[int, list[torch.Tensor]] = {}
        self._rs_payloads: dict[int, list[torch.Tensor]] = {}

    # -- registration helpers (test driver fills these in) -----------------
    def register_all_gather(self, call_id: int, per_rank: list[torch.Tensor]):
        self._ag_payloads[call_id] = per_rank

    def register_reduce_scatter(self, call_id: int, per_rank: list[torch.Tensor]):
        self._rs_payloads[call_id] = per_rank

    # -- collective API (matches GroupCoordinator signature) ---------------
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        # The driver registers the per-rank LSE list under call_id 0 (there is
        # exactly one LSE all_gather in the combine path).
        per_rank = self._ag_payloads.get(0)
        if per_rank is None:
            # Fallback: replicate this rank's input across ranks (used only if
            # the driver does not register; concat-style).
            per_rank = [input_] * self.world_size
        if dim < 0:
            dim += input_.dim()
        return torch.cat(per_rank, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        per_rank = self._rs_payloads.get(0)
        if per_rank is None:
            per_rank = [input_] * self.world_size
        if dim < 0:
            dim += input_.dim()
        summed = torch.stack(per_rank, dim=0).sum(dim=0)
        chunks = torch.chunk(summed, self.world_size, dim=dim)
        return chunks[self.rank_in_group].contiguous()


def ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact (fp32-accumulated) GQA softmax attention.

    Args:
        q: ``[B, H, D]`` (one decode query token per batch entry).
        k: ``[B, S, H_kv, D]`` KV history keys.
        v: ``[B, S, H_kv, D]`` KV history values.
        scale: softmax scale (1/sqrt(D)).

    Returns:
        out:  ``[B, H, D]``
        lse:  ``[H, B]`` natural-log log-sum-exp (FA orientation), where
              ``lse = log(sum_s exp(scale * q.k))`` over the S keys.
    """
    B, H, D = q.shape
    S = k.shape[1]
    H_kv = k.shape[2]
    q_per_kv = H // H_kv

    qf = q.float()
    kf = k.float()
    vf = v.float()

    out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
    lse = torch.empty((H, B), dtype=torch.float32, device=q.device)
    for h in range(H):
        kvh = h // q_per_kv
        # [B, S]
        scores = torch.einsum("bd,bsd->bs", qf[:, h, :], kf[:, :, kvh, :]) * scale
        m = scores.max(dim=-1, keepdim=True).values  # [B,1]
        ex = torch.exp(scores - m)  # [B,S]
        denom = ex.sum(dim=-1, keepdim=True)  # [B,1]
        # weighted value
        out[:, h, :] = torch.einsum("bs,bsd->bd", ex / denom, vf[:, :, kvh, :])
        lse[h, :] = (m.squeeze(-1) + torch.log(denom.squeeze(-1)))
    return out, lse


def shard_attention(
    q_gathered: torch.Tensor,
    k_shard: torch.Tensor,
    v_shard: torch.Tensor,
    scale: float,
    q_per_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-rank partial attention over one KV shard.

    Emulates ``flash_attn_varlen_func`` in ``_forward_with_dcp``: the query has
    already been all-gathered across DCP ranks on the head dim, so
    ``q_gathered`` is ``[B, H * dcp_world_size, D]`` (rank-major). The shard's
    KV has ``H_kv`` heads; the gathered query has
    ``H_kv * q_per_kv * dcp_world_size`` heads. GQA grouping maps query head
    ``h`` to kv head ``(h // q_per_kv) % H_kv``.

    Returns:
        out:  ``[B, H*dcp_world_size, D]``
        lse:  ``[H*dcp_world_size, B]`` natural-log LSE (FA orientation).
    """
    B, Hg, D = q_gathered.shape
    H_kv = k_shard.shape[2]

    qf = q_gathered.float()
    kf = k_shard.float()
    vf = v_shard.float()

    out = torch.empty((B, Hg, D), dtype=torch.float32, device=q_gathered.device)
    lse = torch.empty((Hg, B), dtype=torch.float32, device=q_gathered.device)
    for h in range(Hg):
        kvh = (h // q_per_kv) % H_kv
        scores = torch.einsum("bd,bsd->bs", qf[:, h, :], kf[:, :, kvh, :]) * scale
        m = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - m)
        denom = ex.sum(dim=-1, keepdim=True)
        out[:, h, :] = torch.einsum("bs,bsd->bd", ex / denom, vf[:, :, kvh, :])
        lse[h, :] = (m.squeeze(-1) + torch.log(denom.squeeze(-1)))
    return out, lse
