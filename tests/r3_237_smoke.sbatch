#!/bin/bash
#SBATCH --job-name=r3_237_opcount
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --reservation=reformo
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/e/scratch/jureap59/feuer1/r3_237_smoke/logs/%x_%j.out
#SBATCH --error=/e/scratch/jureap59/feuer1/r3_237_smoke/logs/%x_%j.err

set -x
BASE=/e/scratch/jureap59/feuer1/r3_237_smoke
SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif

# Stage-overridable env (exported by the launcher before sbatch, with defaults).
R3_FLAG="${R3_FLAG:-on}"
R3_REPRO_MODE="${R3_REPRO_MODE:-opcount}"
R3_OPCOUNT="${R3_OPCOUNT:-1}"
RUN_TAG="${RUN_TAG:-opcount}"
OUT="${BASE}/out/${RUN_TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Which capturer to load: "patched" binds the edited local file over the SIF's
# /opt/vllm_build copy (validation-only single-file bind, NOT a deploy); "stock"
# uses the SIF's baked capturer (the pre-patch baseline for Stage-3 parity).
CAPTURER_VARIANT="${CAPTURER_VARIANT:-patched}"
BIND_ARGS=""
if [ "$CAPTURER_VARIANT" = "patched" ]; then
  BIND_ARGS="--bind ${BASE}/bind/routed_experts_capturer.py:/opt/vllm_build/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"
fi

export HF_HUB_CACHE=/e/scratch/jureap59/feuer1/hf_cache/hub
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_R3_OPCOUNT="${R3_OPCOUNT}"
export R3_OUT="$OUT"
export R3_FLAG="$R3_FLAG"
export R3_REPRO_MODE="$R3_REPRO_MODE"
export R3_TP="${R3_TP:-2}"
export R3_STEPS="${R3_STEPS:-64}"

echo "=== r3_237 smoke: variant=$CAPTURER_VARIANT flag=$R3_FLAG mode=$R3_REPRO_MODE opcount=$R3_OPCOUNT out=$OUT ==="

apptainer exec --nv $BIND_ARGS \
  --env LIBRARY_PATH=/.singularity.d/libs \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_R3_OPCOUNT="${R3_OPCOUNT}" \
  --env VLLM_R3_OPCOUNT_DIR="$OUT" \
  --env R3_OUT="$OUT" \
  --env R3_FLAG="$R3_FLAG" \
  --env R3_REPRO_MODE="$R3_REPRO_MODE" \
  --env R3_TP="${R3_TP:-2}" \
  --env R3_STEPS="${R3_STEPS:-64}" \
  --env HF_HUB_CACHE=/e/scratch/jureap59/feuer1/hf_cache/hub \
  --env HF_HUB_OFFLINE=1 \
  "$SIF" python ${BASE}/harness/r3_rank_symmetry_repro.py

echo "=== EXIT $? ==="
echo "=== op-log files ==="
ls -la "$OUT" || true
for f in "$OUT"/oplog_rank*.json; do echo "--- $f ---"; cat "$f" 2>/dev/null; done
