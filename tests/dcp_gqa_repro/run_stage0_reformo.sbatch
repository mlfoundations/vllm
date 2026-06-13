#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stage-0 DCP GQA LSE-combine repro harness — Jupiter reformo, 1 GPU, debug QOS.
# Runs the synthetic combine parity harness inside the 0.20.2rc0 SIF (torch 2.11
# / vllm 0.20.2rc0). The Triton _correct_attn_cp_out_kernel needs a GPU.
#
#   sbatch -D /e/scratch/jureap59/feuer1/vllm tests/dcp_gqa_repro/run_stage0_reformo.sbatch
#
#SBATCH --job-name=dcp_stage0_repro
#SBATCH --partition=reformo
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=dcp_stage0_repro_%j.out

set -euo pipefail

SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif
# Repo root: prefer the submit dir (sbatch -D), else the standard reformo clone.
REPO="${SLURM_SUBMIT_DIR:-/e/scratch/jureap59/feuer1/vllm}"
cd "$REPO"

echo "=== Stage-0 DCP repro: host=$(hostname) job=${SLURM_JOB_ID:-NA} repo=$REPO ==="
echo "=== git: $(git rev-parse --short HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="

export VLLM_USE_FLASHINFER_SAMPLER=0

apptainer exec --nv "$SIF" bash -lc '
  set -euo pipefail
  cd "'"$REPO"'"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export PYTHONPATH="'"$REPO"'":${PYTHONPATH:-}
  # Version asserts (env-map §2d): torch 2.11 / vllm 0.20.2rc0.
  python - <<PY
import torch, vllm
print("torch", torch.__version__, "vllm", vllm.__version__, "cuda", torch.cuda.is_available())
assert torch.__version__.startswith("2.11"), f"expected torch 2.11, got {torch.__version__}"
assert vllm.__version__.startswith("0.20.2rc0"), f"expected vllm 0.20.2rc0, got {vllm.__version__}"
assert torch.cuda.is_available(), "no CUDA device visible"
PY
  echo "=== standalone harness (divergence table) ==="
  python tests/dcp_gqa_repro/test_dcp_combine_parity.py
  echo "=== pytest (MHA control should PASS; GQA asserts are the repro) ==="
  python -m pytest tests/dcp_gqa_repro/test_dcp_combine_parity.py -v -s || true
'
echo "=== done ==="
