#!/usr/bin/env bash
# Submit from the repository root with: sbatch experiments/eth_ucy/run_mid_benchmark.sbatch
#
# Discover Faster-specific names on its login node before adding either line:
#   sinfo -o "%P %G %c %m %l"
#   sacctmgr show assoc user="$USER" format=Account,Partition
#SBATCH --job-name=mid-eth-ucy
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --partition=REPLACE_WITH_FASTER_GPU_PARTITION
##SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
# Uncomment to train one held-out scene per GPU job. Each task writes separate results.
##SBATCH --array=0-4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/experiments/eth_ucy/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/eth_ucy/outputs/mid_faster}"
CONDA_ENV="${CONDA_ENV:-sicnav_mid_env}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

# If Faster exposes Conda through a module, load it here before this fallback.
if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
fi
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
JOB_SUFFIX="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_task${SLURM_ARRAY_TASK_ID}}"
RUN_LOG="${LOG_DIR}/mid-${JOB_SUFFIX}.log"
mkdir -p "${MPLCONFIGDIR}" "${LOG_DIR}"

# Stream all shell/Python output to Slurm and to a durable per-job log in real time.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Live application log: ${RUN_LOG}"
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.get_device_name(0))"

SCENES=(eth hotel univ zara1 zara2)
if [[ -n "${SCENE:-}" ]]; then
    TARGET_SCENES=("${SCENE}")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TARGET_SCENES=("${SCENES[${SLURM_ARRAY_TASK_ID}]}")
else
    TARGET_SCENES=("${SCENES[@]}")
fi

cd "${PROJECT_ROOT}"
srun --unbuffered python experiments/eth_ucy/run_mid_benchmark.py \
    --device cuda \
    --scenes "${TARGET_SCENES[@]}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name mid_faster \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}"
