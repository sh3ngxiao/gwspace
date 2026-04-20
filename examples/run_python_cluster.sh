#!/bin/bash
# Generic Slurm launcher for GWSpace Python jobs.
# Default target: examples/frequency.py
#
# Typical use:
#   sbatch examples/run_python_cluster.sh
#   sbatch --cpus-per-task=2 --mem=6G --time=01:00:00 examples/run_python_cluster.sh
#   PYTHON_SCRIPT=examples/fisher.py sbatch examples/run_python_cluster.sh
#   PYTHON_SCRIPT=examples/frequency.py TIMEOUT_SECONDS=1800 sbatch examples/run_python_cluster.sh

#SBATCH --job-name=gwspace_py
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --partition=compute

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

: "${PYTHON_BIN:=python}"
: "${PYTHON_SCRIPT:=examples/frequency.py}"
: "${CONDA_ENV_NAME:=gwspace312}"
: "${WORK_DIR:=${SLURM_SUBMIT_DIR:-${REPO_DIR}}}"
: "${CACHE_ROOT:=${SLURM_TMPDIR:-${REPO_DIR}/.cache}}"
: "${CPU_LIMIT:=${SLURM_CPUS_PER_TASK:-4}}"
: "${MEM_GB_LIMIT:=8}"
: "${STRICT_MEM_LIMIT:=0}"
: "${TIMEOUT_SECONDS:=}"
: "${MPLBACKEND:=Agg}"
: "${MPLCONFIGDIR:=${CACHE_ROOT}/matplotlib}"

mkdir -p "${CACHE_ROOT}" "${MPLCONFIGDIR}"

export PYTHONUNBUFFERED=1
export MPLBACKEND
export MPLCONFIGDIR
export OMP_NUM_THREADS="${CPU_LIMIT}"
export OPENBLAS_NUM_THREADS="${CPU_LIMIT}"
export MKL_NUM_THREADS="${CPU_LIMIT}"
export NUMEXPR_NUM_THREADS="${CPU_LIMIT}"
export VECLIB_MAXIMUM_THREADS="${CPU_LIMIT}"
export BLIS_NUM_THREADS="${CPU_LIMIT}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_DIR}"
fi

if [[ "${PYTHON_SCRIPT}" = /* ]]; then
  SCRIPT_PATH="${PYTHON_SCRIPT}"
else
  SCRIPT_PATH="${REPO_DIR}/${PYTHON_SCRIPT}"
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Python script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_DIR}/python_job_${RUN_ID}.log"

echo "Start time: $(date)" | tee -a "${RUN_LOG}"

if type module >/dev/null 2>&1; then
  module load apps/anaconda202309 || echo "[WARN] Failed to load module apps/anaconda202309" | tee -a "${RUN_LOG}"
  module load gsl2.7.1 || echo "[WARN] Failed to load module gsl2.7.1" | tee -a "${RUN_LOG}"
else
  echo "[WARN] module command is unavailable, skipping module load." | tee -a "${RUN_LOG}"
fi

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
  else
    echo "[WARN] conda is unavailable, using current Python environment." | tee -a "${RUN_LOG}"
  fi
fi

cd "${WORK_DIR}"

if [[ "${STRICT_MEM_LIMIT}" == "1" ]]; then
  MEM_KB=$((MEM_GB_LIMIT * 1024 * 1024))
  if ulimit -Sv "${MEM_KB}" 2>/dev/null; then
    echo "Applied ulimit soft virtual memory cap: ${MEM_GB_LIMIT} GiB" | tee -a "${RUN_LOG}"
  else
    echo "[WARN] Failed to apply ulimit -Sv. Slurm memory limit still applies." | tee -a "${RUN_LOG}"
  fi
fi

echo "REPO_DIR=${REPO_DIR}" | tee -a "${RUN_LOG}"
echo "WORK_DIR=${WORK_DIR}" | tee -a "${RUN_LOG}"
echo "SCRIPT_PATH=${SCRIPT_PATH}" | tee -a "${RUN_LOG}"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${RUN_LOG}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}" | tee -a "${RUN_LOG}"
echo "CACHE_ROOT=${CACHE_ROOT}" | tee -a "${RUN_LOG}"
echo "MPLCONFIGDIR=${MPLCONFIGDIR}" | tee -a "${RUN_LOG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}" | tee -a "${RUN_LOG}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}" | tee -a "${RUN_LOG}"
echo "CPU_LIMIT=${CPU_LIMIT}" | tee -a "${RUN_LOG}"
echo "MEM_GB_LIMIT=${MEM_GB_LIMIT}" | tee -a "${RUN_LOG}"
echo "STRICT_MEM_LIMIT=${STRICT_MEM_LIMIT}" | tee -a "${RUN_LOG}"
echo "TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-none}" | tee -a "${RUN_LOG}"
echo "MPLBACKEND=${MPLBACKEND}" | tee -a "${RUN_LOG}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}" | tee -a "${RUN_LOG}"

RUN_CMD=()
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  RUN_CMD+=(timeout --signal=TERM "${TIMEOUT_SECONDS}")
fi
if [[ -x /usr/bin/time ]]; then
  RUN_CMD+=(/usr/bin/time -v)
fi
RUN_CMD+=("${PYTHON_BIN}" "${SCRIPT_PATH}" "$@")

"${RUN_CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "End time: $(date)" | tee -a "${RUN_LOG}"
echo "Run log: ${RUN_LOG}" | tee -a "${RUN_LOG}"
