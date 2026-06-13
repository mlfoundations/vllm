#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stage-2 fp32 DISCRIMINATOR — reformo 4 GPU (tp=4, dcp=2). All trustworthy
# same-call probes proved every merge input + merge math correct in fp32
# (combine=DEBUG2 bit-exact, ctxLSE Lc~0, self S2~0, merge M~bf16-noise ~3e-2).
# So the per-step dcp=2 attention output equals the correct full attention
# within bf16 rounding. This run re-checks the e2e greedy divergence with
# dtype=float32: if first_div VANISHES -> the 0.103 drift is bf16
# non-associativity of the sharded recombine vs monolithic attention (NOT an
# algorithmic bug); if it PERSISTS -> a real state/algorithmic defect remains.
#
#   sbatch -D /e/scratch/jureap59/feuer1/vllm tests/dcp_gqa_repro/run_stage2_fp32_disc_reformo.sbatch
#
#SBATCH --job-name=dcp_s2_fp32
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --reservation=reformo
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=dcp_stage2_fp32_%j.out

set -euo pipefail

SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif
REPO="${SLURM_SUBMIT_DIR:-/e/scratch/jureap59/feuer1/vllm}"
DRIVER=${DRIVER:-tests/dcp_gqa_repro/stage1_dcp_driver.py}
export HF_HOME=/e/scratch/jureap59/feuer1/hf_cache
cd "$REPO"

echo "=== Stage-2 fp32 discriminator: host=$(hostname) job=${SLURM_JOB_ID:-NA} repo=$REPO ==="
echo "=== git: $(git rev-parse --short HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="

apptainer exec --nv "$SIF" bash -lc '
  set -euo pipefail
  cd "'"$REPO"'"
  export PYTHONPATH="'"$REPO"'":${PYTHONPATH:-}
  export HF_HOME=/e/scratch/jureap59/feuer1/hf_cache
  export HF_HUB_OFFLINE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  export LIBRARY_PATH="/.singularity.d/libs:${LIBRARY_PATH:-}"
  export SKYRL_DCP_DEBUG3=0
  export SKYRL_DCP_DEBUG2=0
  export SKYRL_DCP_DEBUG=0
  export DCP_PARITY_DTYPE=float32
  export DCP_PARITY_MAX_TOKENS=64
  export DCP_PARITY_TP=4
  export DCP_PARITY_DCP=2

  echo "=== Stage-2 fp32 discriminator run ==="
  python "'"$DRIVER"'" || true
'
echo "=== done ==="
