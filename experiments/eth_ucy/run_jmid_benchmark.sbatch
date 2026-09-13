#!/usr/bin/env bash
# Submit from the repository root with:
#   sbatch experiments/eth_ucy/run_jmid_benchmark.sbatch
# Use SCENE=<name> for one fold, or uncomment the array directive below.
#SBATCH --job-name=jmid-eth-ucy
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --partition=REPLACE_WITH_FASTER_GPU_PARTITION
##SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
##SBATCH --array=0-4

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -d "${PROJECT_ROOT}/trajectory_prediction/safe-interactive-crowdnav/sicnav_diffusion/JMID/MID" ]]; then
    echo "PROJECT_ROOT does not contain the SICNav JMID submodule: ${PROJECT_ROOT}" >&2
    echo "Submit from the repository root or set PROJECT_ROOT explicitly." >&2
    exit 1
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/experiments/eth_ucy/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/eth_ucy/outputs/jmid_faster}"
CONDA_ENV="${CONDA_ENV:-sicnav_jmid_env}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
DEVICE="${DEVICE:-cuda}"

module load Miniconda3/23.10.0-1
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found after loading Miniconda3/23.10.0-1." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Fail before requesting a device if activation selected an incomplete environment.
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
python -c "import numpy, torch; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__)" || {
    echo "The active environment is missing JMID dependencies." >&2
    echo "Create it on Faster with:" >&2
    echo "  conda env create -n ${CONDA_ENV} -f trajectory_prediction/safe-interactive-crowdnav/sicnav_diffusion/JMID/MID/environment.yml" >&2
    echo "Or update it with: conda env update -n ${CONDA_ENV} -f trajectory_prediction/safe-interactive-crowdnav/sicnav_diffusion/JMID/MID/environment.yml --prune" >&2
    exit 1
}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
JOB_SUFFIX="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_task${SLURM_ARRAY_TASK_ID}}"
RUN_LOG="${LOG_DIR}/jmid-${JOB_SUFFIX}.log"
mkdir -p "${MPLCONFIGDIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Live application log: ${RUN_LOG}"

if [[ "${DEVICE}" == cuda* ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || echo "WARNING: nvidia-smi failed; checking PyTorch CUDA directly."
    else
        echo "WARNING: nvidia-smi is unavailable; checking PyTorch CUDA directly."
    fi
    if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.get_device_name(0))"
    else
        echo "WARNING: CUDA was requested but is unavailable. Falling back to CPU."
        DEVICE="cpu"
    fi
else
    echo "Using explicitly requested device: ${DEVICE}"
fi

SCENES=(eth hotel univ zara1 zara2)
if [[ -n "${SCENE:-}" ]]; then
    TARGET_SCENES=("${SCENE}")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TARGET_SCENES=("${SCENES[${SLURM_ARRAY_TASK_ID}]}")
else
    TARGET_SCENES=("${SCENES[@]}")
fi

cd "${PROJECT_ROOT}"
srun --unbuffered python experiments/eth_ucy/run_jmid_benchmark.py \
    --device "${DEVICE}" \
    --scenes "${TARGET_SCENES[@]}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name jmid_faster \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}"
