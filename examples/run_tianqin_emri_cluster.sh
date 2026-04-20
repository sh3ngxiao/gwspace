#!/bin/bash
# Slurm launcher for TianQin DC EMRI catalogue export.
#
# Typical use:
#   sbatch examples/run_tianqin_emri_cluster.sh
#   CONFIG_PATH=configs/tianqin_dc/emri_catalog_batch.json sbatch examples/run_tianqin_emri_cluster.sh
#   CONFIG_PATH=configs/tianqin_dc/emri_catalog_simple.json OUTPUT_PATH=outputs/emri_test.h5 sbatch examples/run_tianqin_emri_cluster.sh
#   DRY_RUN=1 bash examples/run_tianqin_emri_cluster.sh

#SBATCH --job-name=tq_emri_xyz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --partition=compute

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

: "${CONFIG_PATH:=configs/tianqin_dc/emri_catalog_simple.json}"
: "${OUTPUT_PATH:=}"
: "${PYTHON_BIN:=python}"
: "${WORK_DIR:=${SLURM_SUBMIT_DIR:-${REPO_DIR}}}"
: "${CPU_LIMIT:=${SLURM_CPUS_PER_TASK:-1}}"
: "${MEM_GB_LIMIT:=16}"
: "${STRICT_MEM_LIMIT:=0}"
: "${TIMEOUT_SECONDS:=}"
: "${DRY_RUN:=0}"
: "${CACHE_ROOT:=${SLURM_TMPDIR:-${REPO_DIR}/.cache}}"
: "${MPLBACKEND:=Agg}"
: "${MPLCONFIGDIR:=${CACHE_ROOT}/matplotlib}"

if [[ -z "${CONDA_ENV_NAME+x}" ]]; then
  CONDA_ENV_NAME="gwspace312"
fi

mkdir -p "${CACHE_ROOT}" "${MPLCONFIGDIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND
export MPLCONFIGDIR
export OMP_NUM_THREADS="${CPU_LIMIT}"
export OPENBLAS_NUM_THREADS="${CPU_LIMIT}"
export MKL_NUM_THREADS="${CPU_LIMIT}"
export NUMEXPR_NUM_THREADS="${CPU_LIMIT}"
export VECLIB_MAXIMUM_THREADS="${CPU_LIMIT}"
export BLIS_NUM_THREADS="${CPU_LIMIT}"

if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_ABS="${CONFIG_PATH}"
else
  CONFIG_ABS="${REPO_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_ABS}" >&2
  exit 1
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
  if [[ "${OUTPUT_PATH}" = /* ]]; then
    OUTPUT_ABS="${OUTPUT_PATH}"
  else
    OUTPUT_ABS="${REPO_DIR}/${OUTPUT_PATH}"
  fi
  mkdir -p "$(dirname "${OUTPUT_ABS}")"
fi

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_DIR}/tianqin_emri_${RUN_ID}.log"

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
echo "CONFIG_ABS=${CONFIG_ABS}" | tee -a "${RUN_LOG}"
echo "OUTPUT_ABS=${OUTPUT_ABS:-from-config}" | tee -a "${RUN_LOG}"
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
echo "DRY_RUN=${DRY_RUN}" | tee -a "${RUN_LOG}"

python - "${CONFIG_ABS}" <<'PY' | tee -a "${RUN_LOG}"
import glob
import json
import math
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    raw = json.load(handle)

obs = raw.get("observation", {})
catalog = raw.get("catalog", {})
duration_s = float(obs.get("duration_s", 0.0))
sample_rate_hz = float(obs.get("sample_rate_hz", 0.0))
num_samples = int(round(duration_s * sample_rate_hz)) if duration_s > 0.0 and sample_rate_hz > 0.0 else 0
sample_spacing_s = (1.0 / sample_rate_hz) if sample_rate_hz > 0.0 else math.nan

source_hint = None
if isinstance(catalog.get("max_sources"), int):
    source_hint = int(catalog["max_sources"])
elif isinstance(catalog.get("row_numbers"), list) and catalog["row_numbers"]:
    source_hint = len(catalog["row_numbers"])
elif isinstance(catalog.get("rows_per_file"), int):
    rows_per_file = int(catalog["rows_per_file"])
    if isinstance(catalog.get("paths"), list):
        source_hint = len(catalog["paths"]) * rows_per_file
    elif isinstance(catalog.get("path"), str):
        source_hint = rows_per_file
    elif isinstance(catalog.get("path_glob"), str):
        source_hint = len(glob.glob(catalog["path_glob"])) * rows_per_file

single_array_gib = num_samples * 8 / 1024**3
single_source_peak_gib = single_array_gib * 75

print("Config summary:")
print(f"  duration_s = {duration_s}")
print(f"  sample_rate_hz = {sample_rate_hz}")
print(f"  sample_spacing_s = {sample_spacing_s}")
print(f"  num_samples = {num_samples}")
if source_hint is not None:
    print(f"  selected_sources_hint = {source_hint}")
print(f"  single_float64_array_gib = {single_array_gib:.3f}")
print(f"  rough_single_source_peak_memory_gib = {single_source_peak_gib:.2f}  # heuristic only")
if num_samples >= 10_000_000:
    print("  WARNING: sample count is very large; TD XYZ generation may require several GiB even for one source.")
PY

CMD=("${PYTHON_BIN}" "-m" "tianqin_dc.emri_cli" "--config" "${CONFIG_ABS}")
if [[ -n "${OUTPUT_PATH}" ]]; then
  CMD+=("--output" "${OUTPUT_ABS}")
fi

printf 'Command:' | tee -a "${RUN_LOG}"
printf ' %q' "${CMD[@]}" | tee -a "${RUN_LOG}"
printf '\n' | tee -a "${RUN_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed." | tee -a "${RUN_LOG}"
  exit 0
fi

RUN_CMD=()
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  RUN_CMD+=(timeout --signal=TERM "${TIMEOUT_SECONDS}")
fi
if [[ -x /usr/bin/time ]]; then
  RUN_CMD+=(/usr/bin/time -v)
fi
RUN_CMD+=("${CMD[@]}")

"${RUN_CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "End time: $(date)" | tee -a "${RUN_LOG}"
echo "Run log: ${RUN_LOG}" | tee -a "${RUN_LOG}"
