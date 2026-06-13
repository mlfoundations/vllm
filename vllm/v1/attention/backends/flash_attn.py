# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Stage-1 DCP-GQA root-cause instrumentation (env-gated, default OFF).
#
# Set SKYRL_DCP_DEBUG=1 to dump the construction-step tensors inside
# `_forward_with_dcp` so the dcp=2 real path can be diffed against a dcp=1
# reference. This is INERT (G1: no behavior change) unless the env var is set.
# Keep it committed (gated off) so Stage 2/3 can reuse it. See
# notes/vllm/stage1_root_cause_scope.md.
#
# Controls:
#   SKYRL_DCP_DEBUG=1            enable dumps
#   SKYRL_DCP_DEBUG_MAXCALLS=N   only dump the first N _forward_with_dcp calls
#                                (per process; default 4) to avoid log flooding
#   SKYRL_DCP_DEBUG_REF=1        also recompute, in fp32, a reference combine of
#                                the all-gathered per-rank (out, lse) and diff it
#                                against the kernel's context_attn_out_cor — this
#                                isolates whether the context-combine itself is
#                                wrong on the REAL (NCCL-gathered, real-FA-LSE)
#                                inputs vs. the KV-shard contents / final merge.
# ---------------------------------------------------------------------------
import os as _os

_SKYRL_DCP_DEBUG = _os.environ.get("SKYRL_DCP_DEBUG", "0") == "1"
_SKYRL_DCP_DEBUG_REF = _os.environ.get("SKYRL_DCP_DEBUG_REF", "0") == "1"
# Stage-2 re-localization probe (SKYRL_DCP_DEBUG3=1, default OFF, G1-inert).
# Single-call, same-rank, fully-correct reference for the FINAL merge of the
# corrected CONTEXT term with this rank's NEW-TOKEN self term. Mirrors the
# trustworthy SKYRL_DCP_DEBUG2 pattern (no cross-run compares): inside ONE
# `_forward_with_dcp` call it rebuilds the true full attention output by an fp32
# online-softmax of (gathered per-rank raw context partials) + (this rank's raw
# self partial), and diffs it against the kernel's merged `output`. Also diffs
# the merge in isolation (correct fp32 2-way merge of the kernel's OWN
# context_attn_out_cor/context_lse_cor + query_attn_out/query_lse vs kernel
# output) so a wrong merge is separable from a wrong context/self TERM.
_SKYRL_DCP_DEBUG3 = _os.environ.get("SKYRL_DCP_DEBUG3", "0") == "1"
try:
    _SKYRL_DCP_DEBUG3_MAXCALLS = int(
        _os.environ.get("SKYRL_DCP_DEBUG3_MAXCALLS", "200")
    )
except ValueError:
    _SKYRL_DCP_DEBUG3_MAXCALLS = 200
_skyrl_dcp_debug3_calls = 0
try:
    _SKYRL_DCP_DEBUG_MAXCALLS = int(_os.environ.get("SKYRL_DCP_DEBUG_MAXCALLS", "4"))
except ValueError:
    _SKYRL_DCP_DEBUG_MAXCALLS = 4
_skyrl_dcp_debug_calls = 0


def _skyrl_t(name, t):
    """One-line tensor summary for the Stage-1 DCP dump (None-safe)."""
    if t is None:
        return f"{name}=None"
    try:
        flat = t.detach().float().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        stats = (
            f"min={finite.min().item():.4e} max={finite.max().item():.4e} "
            f"mean={finite.mean().item():.4e}"
            if finite.numel() > 0
            else "all-nonfinite"
        )
    except Exception as e:  # pragma: no cover - debug only
        stats = f"<stat-error {e}>"
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"stride={tuple(t.stride())} contig={t.is_contiguous()} {stats}"
    )


class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model.

    Only inspects FlashAttentionImpl layers. Other backends (e.g.
    TurboQuant, MLA) use their own metadata builders and are skipped.
    """
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        if not isinstance(layer.impl, FlashAttentionImpl):
            continue
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if is_quantized_kv_cache(cache_dtype):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_kv_lens = seq_lens - query_lens
            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            self._dcp_context_kv_lens[num_reqs:] = 0
            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

        self._dcp_dtype: torch.dtype | None = None
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires ≥16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttention): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # Scatter write into the KV cache using slot_mapping indices.
        # No TMA kernel is invoked here, so stride canonicalization is not needed.
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _skyrl_dcp_ref_combine_check(
        self,
        context_attn_out: torch.Tensor,
        context_lse: torch.Tensor,
        context_attn_out_cor: torch.Tensor,
        context_lse_cor: torch.Tensor,
        dcp_grp,
    ) -> None:
        """Stage-1 reference combine check (SKYRL_DCP_DEBUG_REF=1).

        Each DCP rank produced a partial ``(context_attn_out [B,Hg,D],
        context_lse [Hg,B])`` over its OWN KV shard (Hg = num_heads*dcp_world).
        The CORRECT cross-rank combine for the gathered query is an
        online-softmax reduction over the per-rank partials. We all-gather every
        rank's partials and recompute that reduction in fp32, then diff this
        rank's own-head slice against the kernel's ``context_attn_out_cor`` /
        ``context_lse_cor``. If the kernel matches this reference, the
        context-combine math is correct on the REAL inputs (so the e2e drift is
        in candidate (a) KV-shard contents or (e) the final merge); if it
        diverges, the locus is the combine input-construction itself (candidate
        1/d head accounting).
        """
        try:
            r = dcp_grp.rank_in_group
            w = dcp_grp.world_size
            B, Hg, D = context_attn_out.shape  # Hg = num_heads * dcp_world
            cp_h = Hg // w
            # Gather every rank's partial out [B,Hg,D] and lse [Hg,B] and split
            # back into the per-rank list (torch.chunk on the gathered dim is
            # robust to whichever rank-major layout the collective produced).
            out_all = dcp_grp.all_gather(context_attn_out.contiguous(), dim=0)
            lse_all = dcp_grp.all_gather(
                context_lse.transpose(0, 1).contiguous(), dim=0
            )  # gather [B,Hg] -> [w*B, Hg]
            out_list = [c.float() for c in torch.chunk(out_all, w, dim=0)]
            lse_list = [c.float() for c in torch.chunk(lse_all, w, dim=0)]
            out_stk = torch.stack(out_list, dim=0)  # [W,B,Hg,D]
            lse_stk = torch.stack(lse_list, dim=0)  # [W,B,Hg]
            # CORRECT cross-rank combine: per head-cell, global LSE over ranks,
            # then sum_r out_r * exp(lse_r - lse_global). This is exactly what
            # correct_attn_out (per-rank rescale) + reduce_scatter (sum over
            # ranks, scatter heads) compute together — base-e.
            lse_g = torch.logsumexp(lse_stk, dim=0)  # [B,Hg]
            wts = torch.exp(lse_stk - lse_g.unsqueeze(0)).unsqueeze(-1)  # [W,B,Hg,1]
            ref_out_full = (out_stk * wts).sum(0)  # [B,Hg,D]
            # reduce_scatter keeps THIS rank's head block.
            ref_out = ref_out_full[:, r * cp_h : (r + 1) * cp_h, :]
            ref_lse = lse_g[:, r * cp_h : (r + 1) * cp_h]  # [B,cp_h]
            # context_lse_cor arrives as [B,cp_h] (already transposed back).
            clc = context_lse_cor.float()
            if clc.shape != ref_lse.shape and clc.transpose(0, 1).shape == ref_lse.shape:
                clc = clc.transpose(0, 1)
            # Compare only finite cells (zero-context ranks emit -inf LSE / 0 out).
            fin = torch.isfinite(ref_out).all(dim=-1) & torch.isfinite(
                context_attn_out_cor.float()
            ).all(dim=-1)
            if fin.any():
                d_out = (
                    (context_attn_out_cor.float()[fin] - ref_out[fin]).abs().max().item()
                )
            else:
                d_out = float("nan")
            finl = torch.isfinite(ref_lse) & torch.isfinite(clc)
            d_lse = (clc[finl] - ref_lse[finl]).abs().max().item() if finl.any() else float("nan")
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (REF) kernel-vs-fp32-online-softmax "
                "combine over finite cells: max|Δout|=%.4e max|Δlse|=%.4e "
                "Hg=%d cp_h=%d B=%d (SMALL ⇒ combine math correct on real "
                "inputs ⇒ locus is (a) KV-shard contents or (e) merge; LARGE "
                "⇒ combine input-construction/head-accounting is the defect)",
                r,
                d_out,
                d_lse,
                Hg,
                cp_h,
                B,
            )
            # One-shot ELEMENT dump on the first LARGE-Δout finite cell: show the
            # per-rank raw outputs, the per-rank weights, the REF sum, and the
            # kernel result for head 0 of this rank's block — to read off whether
            # the kernel summed across ranks (reduce_scatter) or kept only its
            # own weighted output (a missing cross-rank sum / wrong head block).
            if (
                not getattr(self, "_skyrl_dcp_ref_elem_done", False)
                and d_out == d_out
                and d_out > 0.05
            ):
                bi = 0
                hi = r * cp_h  # this rank's first own head (global head index)
                # per-rank raw FA out + weight at [bi, hi, 0]
                raw = [out_stk[s, bi, hi, 0].item() for s in range(w)]
                wt = [wts[s, bi, hi, 0].item() for s in range(w)]
                lse_per = [lse_stk[s, bi, hi].item() for s in range(w)]
                ref_v = ref_out_full[bi, hi, 0].item()
                ker_v = context_attn_out_cor.float()[bi, 0, 0].item()
                logger.info(
                    "[SKYRL_DCP_DEBUG] rank%d (REF-ELEM) b=0 head_global=%d d=0 | "
                    "per-rank raw_out=%s | per-rank lse=%s | per-rank weight="
                    "exp(lse_r-lse_g)=%s | REF sum_r(raw*wt)=%.6f | KERNEL "
                    "context_attn_out_cor[0,0,0]=%.6f",
                    r,
                    hi,
                    ["%.5f" % x for x in raw],
                    ["%.4f" % x for x in lse_per],
                    ["%.5f" % x for x in wt],
                    ref_v,
                    ker_v,
                )
                self._skyrl_dcp_ref_elem_done = True
        except Exception as e:  # pragma: no cover - debug only
            logger.info("[SKYRL_DCP_DEBUG] (REF) check err %s", e)

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        global _skyrl_dcp_debug_calls
        _dbg = _SKYRL_DCP_DEBUG and (
            _skyrl_dcp_debug_calls < _SKYRL_DCP_DEBUG_MAXCALLS
        )
        _dcp_grp = get_dcp_group()
        if _dbg:
            _skyrl_dcp_debug_calls += 1
            _r = _dcp_grp.rank_in_group
            _w = _dcp_grp.world_size
            logger.info(
                "[SKYRL_DCP_DEBUG] === _forward_with_dcp call #%d | "
                "dcp_rank=%d dcp_world=%d num_heads(per-tp)=%d num_kv_heads=%d "
                "q_per_kv=%d head_size=%d scale=%s interleave=%s causal=%s "
                "fa_version=%s kv_dtype=%s ===",
                _skyrl_dcp_debug_calls,
                _r,
                _w,
                self.num_heads,
                self.num_kv_heads,
                self.num_queries_per_kv,
                self.head_size,
                self.scale,
                getattr(self, "cp_kv_cache_interleave_size", "?"),
                attn_metadata.causal,
                self.vllm_flash_attn_version,
                self.kv_cache_dtype,
            )
            # Candidate (a): varlen KV sharding boundaries this rank reads.
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (a) %s | max_dcp_context_kv_len=%s "
                "max_seq_len=%s | %s | %s",
                _r,
                _skyrl_t("dcp_context_kv_lens", attn_metadata.dcp_context_kv_lens),
                attn_metadata.max_dcp_context_kv_len,
                attn_metadata.max_seq_len,
                _skyrl_t("seq_lens", attn_metadata.seq_lens),
                _skyrl_t("query_start_loc", attn_metadata.query_start_loc),
            )
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (a) %s | %s",
                _r,
                _skyrl_t("query(pre-gather)", query),
                _skyrl_t("block_table", attn_metadata.block_table),
            )
            # Candidate (c): the descales actually fed to FA.
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (c) %s | %s | %s",
                _r,
                _skyrl_t("q_descale", q_descale),
                _skyrl_t("k_descale", k_descale),
                _skyrl_t("v_descale", v_descale),
            )

        query = query.contiguous()
        query_across_dcp = _dcp_grp.all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        n = query_across_dcp.shape[0]
        if _dbg:
            # Candidate (d): the real NCCL all_gather head ordering. The gathered
            # query carries num_heads*dcp_world heads on dim=1. The FA kernel will
            # derive its GQA grouping from (gathered_q_heads / kv_heads). Confirm
            # whether rank r's OWN heads sit at [r*num_heads : (r+1)*num_heads]
            # (rank-major replication, what the per-rank LSE slice assumes) AND
            # what GQA group FA will assign each gathered head.
            _Hg = query_across_dcp.shape[1]
            _fa_qpkv = _Hg // self.num_kv_heads if self.num_kv_heads else -1
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (d) %s | gathered_q_heads=%d "
                "=> FA-effective q_per_kv=%d (kv_heads=%d). own-slice "
                "[%d:%d]. per-head->FA-kv-group = head//%d ; per-head->true-kv "
                "= (head%%%d)//%d",
                _r,
                _skyrl_t("query_across_dcp", query_across_dcp),
                _Hg,
                _fa_qpkv,
                self.num_kv_heads,
                _r * self.num_heads,
                (_r + 1) * self.num_heads,
                _fa_qpkv,
                self.num_heads,
                self.num_queries_per_kv,
            )
            # Per-rank head-identity probe: are the gathered heads literally
            # rank-major REPLICAS of the same query (queries are NOT sharded
            # across DCP, only KV is)? Compare rank r's own slice vs rank 0's.
            try:
                _own = query_across_dcp[
                    :, _r * self.num_heads : (_r + 1) * self.num_heads, :
                ]
                _slice0 = query_across_dcp[:, 0 : self.num_heads, :]
                _rep_dmax = (_own.float() - _slice0.float()).abs().max().item()
                logger.info(
                    "[SKYRL_DCP_DEBUG] rank%d (d) own-slice vs rank0-slice "
                    "max|Δ|=%.4e (==0 ⇒ gathered heads are rank-major REPLICAS "
                    "of identical query, so the dcp_world copies differ ONLY by "
                    "which KV shard they attend — the combine assumption)",
                    _r,
                    _rep_dmax,
                )
            except Exception as e:  # pragma: no cover - debug only
                logger.info("[SKYRL_DCP_DEBUG] rank%d (d) replica-probe err %s", _r, e)
        (dcp_context_out,) = current_workspace_manager().get_simultaneous(
            (
                (n, self.num_heads * self.dcp_world_size, self.head_size),
                self._dcp_dtype,
            ),
        )
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=dcp_context_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        if _dbg:
            # Candidate (b): the LSE base FA actually emits. Standard FA emits
            # natural-log (base-e) LSE; cp_lse_ag_out_rs defaults is_lse_base_on_e
            # =True. FlashInfer's DCP path forces False (backend-specific!). Dump
            # raw context_lse magnitude so a base-2 (=log2) vs base-e mismatch is
            # visible (base-2 values are ~1.4427x the base-e values for the same
            # softmax denominator).
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (b/e) %s | %s",
                _dcp_grp.rank_in_group,
                _skyrl_t("context_attn_out(FA,[B,Hg,D])", context_attn_out),
                _skyrl_t("context_lse(FA,[Hg,B])", context_lse),
            )
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        _context_lse_in = context_lse.transpose(0, 1)
        if _dbg:
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (1) %s -> dcp_combine",
                _dcp_grp.rank_in_group,
                _skyrl_t("context_lse.transpose(0,1)[B,Hg]", _context_lse_in),
            )
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            _context_lse_in,
            _dcp_grp,
            return_lse=True,
        )
        if _dbg:
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (2) post-dcp_combine %s | %s "
                "(reduce_scatter'd to this rank's num_heads; LSE sliced "
                "[cp_num_heads*rank:...])",
                _dcp_grp.rank_in_group,
                _skyrl_t("context_attn_out_cor", context_attn_out_cor),
                _skyrl_t("context_lse_cor(pre-T)", context_lse_cor),
            )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()
        if _dbg and _SKYRL_DCP_DEBUG_REF:
            self._skyrl_dcp_ref_combine_check(
                context_attn_out, context_lse, context_attn_out_cor,
                context_lse_cor, _dcp_grp,
            )

        (dcp_query_out,) = current_workspace_manager().get_simultaneous(
            ((query.shape[0], self.num_heads, self.head_size), self._dcp_dtype),
        )
        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=dcp_query_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        if _dbg:
            # Candidate (e): the merge of the gathered/corrected CONTEXT term with
            # this rank's OWN new-token self-attention term. The Stage-0 harness
            # SKIPPED this merge (folded all KV into the sharded context) and
            # still passed — so this finish is a strong suspect. Dump both inputs.
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (e) merge inputs: %s | %s | %s | %s",
                _dcp_grp.rank_in_group,
                _skyrl_t("context_attn_out_cor", context_attn_out_cor),
                _skyrl_t("context_lse_cor[B,H]", context_lse_cor),
                _skyrl_t("query_attn_out(self)", query_attn_out),
                _skyrl_t("query_lse(self,[H,B])", query_lse),
            )
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )
        if _dbg:
            logger.info(
                "[SKYRL_DCP_DEBUG] rank%d (e) post-merge %s",
                _dcp_grp.rank_in_group,
                _skyrl_t("output(final)", output[:n]),
            )
        # ------------------------------------------------------------------
        # Stage-2 re-localization probe (SKYRL_DCP_DEBUG3): single-call,
        # same-rank, fully-correct references. Two diffs, both fp32, both built
        # from THIS call's own tensors (no cross-run ambiguity):
        #   (M) MERGE-only: correct 2-way online softmax of the kernel's OWN
        #       (context_attn_out_cor, context_lse_cor) + (query_attn_out,
        #       query_lse) vs the kernel's `output`. Isolates the merge math.
        #   (F) FULL: gather every rank's RAW context partial (out,lse) +
        #       this rank's RAW self partial (out,lse), do the complete
        #       (N context shards + 1 self) online softmax, vs `output`.
        #       Isolates whether a TERM fed to the merge is itself wrong
        #       (context-shard contents / self) even when combine+merge are OK.
        # Per-decode-step max|Δ| growth localizes the 0.103 drift.
        global _skyrl_dcp_debug3_calls
        if (
            _SKYRL_DCP_DEBUG3
            and _skyrl_dcp_debug3_calls < _SKYRL_DCP_DEBUG3_MAXCALLS
        ):
            _skyrl_dcp_debug3_calls += 1
            try:
                r = _dcp_grp.rank_in_group
                w = _dcp_grp.world_size
                # --- (M) merge-only correct reference (base-e online softmax) ---
                # context_attn_out_cor/query_attn_out: [B,H,D];
                # context_lse_cor/query_lse: [H,B] -> transpose to [B,H].
                co = context_attn_out_cor.float()
                qo = query_attn_out.float()
                cl = context_lse_cor.float().transpose(0, 1)  # [B,H]
                ql = query_lse.float().transpose(0, 1)  # [B,H]
                ls = torch.stack([cl, ql], dim=0)  # [2,B,H]
                ls_safe = torch.where(
                    torch.isfinite(ls), ls, torch.full_like(ls, -1e30)
                )
                g_m = torch.logsumexp(ls_safe, dim=0)  # [B,H]
                w_c = torch.exp(ls_safe[0] - g_m).unsqueeze(-1)  # [B,H,1]
                w_q = torch.exp(ls_safe[1] - g_m).unsqueeze(-1)
                ref_merge = co * w_c + qo * w_q  # [B,H,D]
                k_out = output[:n].float()
                finm = (
                    torch.isfinite(ref_merge).all(-1)
                    & torch.isfinite(k_out).all(-1)
                )
                d_merge = (
                    (ref_merge[finm] - k_out[finm]).abs().max().item()
                    if finm.any()
                    else float("nan")
                )
                # --- (F) full from-scratch reference (N context shards + self) ---
                # Gather raw context partials across ranks (Hg = H*w replicated
                # query heads), take this rank's own H-head block from each shard.
                raw_c = context_attn_out.contiguous()  # [B,Hg,D]
                raw_cl = context_lse.transpose(0, 1).contiguous()  # [B,Hg]
                B = raw_c.shape[0]
                gc_out = _dcp_grp.all_gather(raw_c, dim=0)  # [w*B,Hg,D]
                gc_lse = _dcp_grp.all_gather(raw_cl, dim=0)  # [w*B,Hg]
                Hg = raw_c.shape[1]
                Hloc = Hg // w
                # shard s's partial for THIS rank's own query heads = the
                # [r*Hloc:(r+1)*Hloc] head-block of shard s's gathered tensor.
                ctx_out = torch.stack(
                    [
                        gc_out[s * B : (s + 1) * B, r * Hloc : (r + 1) * Hloc, :].float()
                        for s in range(w)
                    ],
                    0,
                )  # [w,B,H,D]
                ctx_lse = torch.stack(
                    [
                        gc_lse[s * B : (s + 1) * B, r * Hloc : (r + 1) * Hloc].float()
                        for s in range(w)
                    ],
                    0,
                )  # [w,B,H]
                self_out = qo.unsqueeze(0)  # [1,B,H,D]
                self_lse = ql.unsqueeze(0)  # [1,B,H]
                all_out = torch.cat([ctx_out, self_out], 0)  # [w+1,B,H,D]
                all_lse = torch.cat([ctx_lse, self_lse], 0)  # [w+1,B,H]
                all_safe = torch.where(
                    torch.isfinite(all_lse), all_lse, torch.full_like(all_lse, -1e30)
                )
                g_f = torch.logsumexp(all_safe, dim=0)  # [B,H]
                wts_f = torch.exp(all_safe - g_f.unsqueeze(0)).unsqueeze(-1)
                ref_full = (all_out * wts_f).sum(0)  # [B,H,D]
                finf = (
                    torch.isfinite(ref_full).all(-1) & torch.isfinite(k_out).all(-1)
                )
                d_full = (
                    (ref_full[finf] - k_out[finf]).abs().max().item()
                    if finf.any()
                    else float("nan")
                )
                # --- (C) CONTEXT-ONLY: does (F)'s reconstructed context term
                # (online-softmax of the gathered RAW context partials) equal
                # the kernel's corrected context (context_attn_out_cor)? If (C)~0
                # then (F)'s context build is faithful (so a large (F) means the
                # SELF term / decode self-attn is the defect); if (C) large then
                # (F)'s head-indexing is itself off (false-positive guard vs the
                # Stage-1 trap). Cross-checks against the trustworthy DEBUG2.
                g_c = torch.logsumexp(ctx_lse, dim=0)  # [B,H] context-only global
                wts_c = torch.exp(ctx_lse - g_c.unsqueeze(0)).unsqueeze(-1)
                ref_ctx = (ctx_out * wts_c).sum(0)  # [B,H,D]
                kc = context_attn_out_cor.float()
                finc = torch.isfinite(ref_ctx).all(-1) & torch.isfinite(kc).all(-1)
                d_ctx = (
                    (ref_ctx[finc] - kc[finc]).abs().max().item()
                    if finc.any()
                    else float("nan")
                )
                # (S) SELF-only: kernel's query_attn_out vs a recompute is not
                # available here, but compare the kernel's context LSE the merge
                # used (context_lse_cor) against (F)'s context-only global LSE —
                # a mismatch means the LSE handed to the merge disagrees with the
                # true context LSE (the decode-growing weight defect).
                clc = context_lse_cor.float().transpose(0, 1)  # [B,H]
                finl = torch.isfinite(clc) & torch.isfinite(g_c)
                d_clse = (
                    (clc[finl] - g_c[finl]).abs().max().item()
                    if finl.any()
                    else float("nan")
                )
                # --- (S2) SELF-term recompute (TRUSTWORTHY, same-call, no
                # cross-rank gather): redo the new-token self-attention in fp32
                # directly from the SAME raw (query,key,value) the self FA used,
                # per request (causal), and diff against query_attn_out /
                # query_lse. This is the ONLY merge input with zero trustworthy
                # coverage so far (combine=DEBUG2 bit-exact, ctxLSE=Lc~0,
                # merge=M~noise). A nonzero (S2) localizes the defect to the
                # decode self term; (S2)~0 means all merge inputs are correct
                # and the defect is downstream of attention (output routing).
                d_self_o = float("nan")
                d_self_l = float("nan")
                try:
                    qf = query.float()  # [T, H, D]
                    kf = key.float()  # [T, Hkv, D]
                    vf = value.float()
                    qsl = attn_metadata.query_start_loc
                    sc = self.scale
                    H = qf.shape[1]
                    Hkv = kf.shape[1]
                    rep = H // Hkv
                    nreq = qsl.shape[0] - 1
                    so_ref = torch.empty_like(qf)
                    sl_ref = torch.empty(
                        (H, qf.shape[0]), dtype=torch.float32, device=qf.device
                    )
                    for rq in range(nreq):
                        a = int(qsl[rq].item())
                        b = int(qsl[rq + 1].item())
                        if b <= a:
                            continue
                        qx = qf[a:b]  # [t,H,D]
                        kx = kf[a:b]  # [t,Hkv,D]
                        vx = vf[a:b]
                        if rep > 1:
                            kx = kx.repeat_interleave(rep, dim=1)
                            vx = vx.repeat_interleave(rep, dim=1)
                        # scores [H,t,t]
                        sco = torch.einsum(" qhd,khd->hqk", qx, kx) * sc
                        t = b - a
                        cm = torch.tril(
                            torch.ones(t, t, device=qf.device, dtype=torch.bool)
                        )
                        sco = sco.masked_fill(~cm.unsqueeze(0), float("-inf"))
                        lse_r = torch.logsumexp(sco, dim=-1)  # [H,t]
                        prob = torch.softmax(sco, dim=-1)
                        out_r = torch.einsum("hqk,khd->qhd", prob, vx)  # [t,H,D]
                        so_ref[a:b] = out_r
                        sl_ref[:, a:b] = lse_r
                    qo_k = query_attn_out.float()
                    ql_k = query_lse.float()  # [H,T]
                    fso = torch.isfinite(so_ref).all(-1) & torch.isfinite(qo_k).all(-1)
                    d_self_o = (
                        (so_ref[fso] - qo_k[fso]).abs().max().item()
                        if fso.any()
                        else float("nan")
                    )
                    fsl = torch.isfinite(sl_ref) & torch.isfinite(ql_k)
                    d_self_l = (
                        (sl_ref[fsl] - ql_k[fsl]).abs().max().item()
                        if fsl.any()
                        else float("nan")
                    )
                except Exception as _es:  # pragma: no cover - debug only
                    d_self_o = -1.0
                    logger.info("[SKYRL_DCP_DEBUG3] rank%d (S2) err %s", r, _es)
                # Per-token (decode-position) max|Δ| for the FULL ref, so the
                # GROWTH with decode length is visible directly.
                per_tok = (
                    (ref_full - k_out).abs().amax(dim=(1, 2)).tolist()
                    if k_out.numel()
                    else []
                )
                logger.info(
                    "[SKYRL_DCP_DEBUG3] rank%d call#%d n_tok=%d "
                    "(M)merge-only max|Δ|=%.4e  (F)full-from-raw max|Δ|=%.4e  "
                    "(C)ctx-recon-vs-kernel max|Δ|=%.4e  (Lc)ctxLSE-vs-true "
                    "max|Δ|=%.4e  (S2)self-out max|Δ|=%.4e self-lse max|Δ|=%.4e  "
                    "per-tok|Δ|=%s  (S2~0 ⇒ self term correct, all merge inputs "
                    "correct ⇒ defect downstream of attn; S2 large ⇒ decode "
                    "self-attn defect)",
                    r,
                    _skyrl_dcp_debug3_calls,
                    int(n),
                    d_merge,
                    d_full,
                    d_ctx,
                    d_clse,
                    d_self_o,
                    d_self_l,
                    ["%.3e" % x for x in per_tok[:8]],
                )
            except Exception as _e3:  # pragma: no cover - debug only
                logger.info("[SKYRL_DCP_DEBUG3] rank%d probe err %s",
                            _dcp_grp.rank_in_group, _e3)

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
