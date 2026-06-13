#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stage-2 RE-LOCALIZATION probe — Jupiter reformo, 4 GPU (tp=4, dcp=2).
# Runs the REAL DCP path (Qwen2.5-1.5B) with SKYRL_DCP_DEBUG3=1 so
# _forward_with_dcp prints, per decode call on each rank, two single-call
# fp32 references vs the kernel's merged `output`:
#   (M) merge-only correct 2-way online softmax of the kernel's own
#       (context_attn_out_cor,context_lse_cor)+(query_attn_out,query_lse).
#   (F) full from-scratch (N gathered raw context shards + this rank's raw
#       self term) online softmax.
# M~0 & F~0  => merge+terms correct at that step.
# M~0 & F large => a context/self TERM fed to merge is itself wrong (suspect a).
# M large => merge math/inputs wrong (suspect e).
# Per-token |Δ| shows the decode-position growth that matches the 0.103 drift.
#
#   sbatch -D /e/scratch/jureap59/feuer1/vllm tests/dcp_gqa_repro/run_stage2_relocalize_reformo.sbatch
#
#SBATCH --job-name=dcp_s2_reloc
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --reservation=reformo
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=dcp_stage2_reloc_%j.out

set -euo pipefail

SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif
REPO="${SLURM_SUBMIT_DIR:-/e/scratch/jureap59/feuer1/vllm}"
DRIVER=${DRIVER:-tests/dcp_gqa_repro/stage1_dcp_driver.py}
export HF_HOME=/e/scratch/jureap59/feuer1/hf_cache
cd "$REPO"

echo "=== Stage-2 re-localization: host=$(hostname) job=${SLURM_JOB_ID:-NA} repo=$REPO ==="
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
  # Stage-2 re-localization probe ON; Stage-1 dumps + DEBUG2 OFF (small log).
  export SKYRL_DCP_DEBUG3=1
  export SKYRL_DCP_DEBUG2=0
  export SKYRL_DCP_DEBUG=0
  export DCP_PARITY_MAX_TOKENS=64
  export DCP_PARITY_TP=4
  export DCP_PARITY_DCP=2

  echo "=== Stage-2 re-localization run ==="
  python "'"$DRIVER"'" || true
'
echo "=== done ==="
