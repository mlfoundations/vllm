#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stage-1 DCP GQA LSE-combine ROOT-CAUSE instrumentation — Jupiter reformo
# reservation, 4 GPU (tp=4, dcp=2). Runs the REAL DCP path (Qwen2.5-1.5B) with
# the env-gated SKYRL_DCP_DEBUG instrumentation in vllm/v1/attention/backends/
# flash_attn.py so the dcp=2 construction-step tensors can be diffed against the
# (correct) combine math. Uses the existing SkyRL Stage-3 parity driver, which
# already builds dcp=1 then dcp=2 on identical prompts/seed.
#
#   sbatch -D /e/scratch/jureap59/feuer1/vllm tests/dcp_gqa_repro/run_stage1_reformo.sbatch
#
#SBATCH --job-name=dcp_stage1_rootcause
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --reservation=reformo
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=dcp_stage1_rootcause_%j.out

set -euo pipefail

SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif
REPO="${SLURM_SUBMIT_DIR:-/e/scratch/jureap59/feuer1/vllm}"
# Standalone in-fork driver (no MarinSkyRL dependency); deploys with the branch.
DRIVER=${DRIVER:-tests/dcp_gqa_repro/stage1_dcp_driver.py}
export HF_HOME=/e/scratch/jureap59/feuer1/hf_cache
cd "$REPO"

echo "=== Stage-1 DCP root-cause: host=$(hostname) job=${SLURM_JOB_ID:-NA} repo=$REPO ==="
echo "=== git: $(git rev-parse --short HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
echo "=== driver: $DRIVER ==="

apptainer exec --nv "$SIF" bash -lc '
  set -euo pipefail
  cd "'"$REPO"'"
  export PYTHONPATH="'"$REPO"'":${PYTHONPATH:-}
  export HF_HOME=/e/scratch/jureap59/feuer1/hf_cache
  export HF_HUB_OFFLINE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  # Triton JIT links its launcher .so against -l:libcuda.so.1; the runtime stub
  # mounted by apptainer --nv lives in /.singularity.d/libs but is NOT on gcc'"'"'s
  # LIBRARY_PATH (only the cuda stubs dir is) -> ld "cannot find -l:libcuda.so.1".
  # Add the --nv driver dir so the launcher links. (GH200/aarch64 gcc gotcha.)
  export LIBRARY_PATH="/.singularity.d/libs:${LIBRARY_PATH:-}"
  # ---- Stage-1 instrumentation switches (env-gated; default OFF in source) ----
  export SKYRL_DCP_DEBUG=1
  export SKYRL_DCP_DEBUG_REF=1
  # Capture enough decode steps to see the drift GROW with KV history. Per
  # decode step every attn layer fires _forward_with_dcp once; cap so the log is
  # bounded but spans several decode positions.
  export SKYRL_DCP_DEBUG_MAXCALLS=120
  # Longer generation so the sharded KV history grows and the (suspected)
  # mis-weighted cross-rank term accumulates into the observed ~0.1 drift.
  export DCP_PARITY_MAX_TOKENS=64
  export DCP_PARITY_TP=4
  export DCP_PARITY_DCP=2

  python - <<PY
import torch, vllm
import vllm.v1.attention.backends.flash_attn as fa
print("torch", torch.__version__, "vllm", vllm.__version__, "cuda", torch.cuda.is_available())
print("flash_attn ->", fa.__file__)
assert fa.__file__.startswith("'"$REPO"'"), f"flash_attn NOT from fork tree: {fa.__file__}"
assert hasattr(fa, "_skyrl_t"), "Stage-1 instrumentation helper missing from fork flash_attn.py"
assert torch.cuda.is_available(), "no CUDA device visible"
print("instrumentation present:", fa._SKYRL_DCP_DEBUG, fa._SKYRL_DCP_DEBUG_REF, fa._SKYRL_DCP_DEBUG_MAXCALLS)
PY

  echo "=== Stage-1 real-path run with SKYRL_DCP_DEBUG dumps ==="
  python "'"$DRIVER"'" || true
'
echo "=== done ==="
