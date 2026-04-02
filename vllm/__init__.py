# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401

MODULE_ATTRS = {
    "bc_linter_skip": "._bc_linter:bc_linter_skip",
    "bc_linter_include": "._bc_linter:bc_linter_include",
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster

    from ._bc_linter import bc_linter_include, bc_linter_skip
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "bc_linter_skip",
    "bc_linter_include",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]

# FP8 per-tensor no-transpose patch for weight sync compatibility
import os as _os
if _os.environ.get("SKYRL_FUSE_WEIGHTS") == "1":
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod as _Fp8LM
        _orig_process = _Fp8LM.process_weights_after_loading

        def _patched_process(self, layer, *args, **kwargs):
            # Save param attributes before FP8 processing
            _saved = {}
            for _pn, _p in layer.named_parameters():
                _a = {'subclass_type': type(_p)}
                for _attr in ('weight_loader', 'output_dim', 'input_dim',
                              '_output_dim', '_input_dim', 'packed_dim',
                              'packed_factor', 'tp_rank', 'tp_size'):
                    if hasattr(_p, _attr):
                        _a[_attr] = getattr(_p, _attr)
                _saved[_pn] = _a

            # Run original (quantize + transpose)
            _result = _orig_process(self, layer, *args, **kwargs)

            # Un-transpose the weight back to [out, in] for weight sync compat
            # The apply() method will transpose on-the-fly
            if hasattr(layer, 'weight') and layer.weight.data.dim() == 2:
                import torch
                layer.weight = torch.nn.Parameter(
                    layer.weight.data.t().contiguous(), requires_grad=False)

            # Restore attributes
            for _pn, _p in layer.named_parameters():
                if _pn in _saved:
                    for _attr, _val in _saved[_pn].items():
                        try:
                            setattr(_p, _attr, _val)
                        except Exception:
                            pass
            return _result

        _Fp8LM.process_weights_after_loading = _patched_process
    except Exception as _e:
        import warnings as _w
        _w.warn(f"FP8 no-transpose patch failed: {_e}")

# FlashRL FP8 patch - auto-activate when FLASHRL_CONFIG is set
if _os.environ.get("FLASHRL_CONFIG"):
    try:
        from vllm.model_executor.layers.patch import apply_patch as _apply_flashrl
        _apply_flashrl()
    except Exception as _e:
        import warnings as _w
        _w.warn(f"FlashRL patch failed: {_e}")

