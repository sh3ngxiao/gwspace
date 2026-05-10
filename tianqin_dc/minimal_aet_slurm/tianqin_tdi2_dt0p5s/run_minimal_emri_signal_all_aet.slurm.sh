#!/bin/bash
#SBATCH -J tq2_min_emri
#SBATCH -p gpu_part
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=55
#SBATCH --mem=1T
#SBATCH --time=72:00:00
#SBATCH -o /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.out
#SBATCH -e /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/zhuangzhenye/jobs/GWSpace}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/tianqin_dc/tianqin_tdi2_dt0p5s/minimal_emri_signal_all_aet.json}"
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
LOG_DIR="$RUN_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"
TQ_NUMERICAL_ORBIT_PATH="${TQ_NUMERICAL_ORBIT_PATH:-/public/home/zhuangzhenye/jobs/tianqin_orbit/satellite_positions_ssb_tq_1s_from_oem.npy}"
TQ_NUMERICAL_ORBIT_OUT_OF_RANGE="${TQ_NUMERICAL_ORBIT_OUT_OF_RANGE:-raise}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

if type module >/dev/null 2>&1; then
  module load apps/anaconda202309 || echo "[WARN] Failed to load module apps/anaconda202309"
  module load gsl2.7.1 || module load gsl || echo "[WARN] Failed to load a GSL module"
fi
if [[ -n "${GSL_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$GSL_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
export TQ_NUMERICAL_ORBIT_PATH
export TQ_NUMERICAL_ORBIT_OUT_OF_RANGE

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"
echo "TQ numerical orbit: $TQ_NUMERICAL_ORBIT_PATH"
echo "TQ numerical orbit out-of-range: $TQ_NUMERICAL_ORBIT_OUT_OF_RANGE"
echo "Workers: $WORKERS"
echo "Threads per worker: $THREADS_PER_WORKER"
echo "Output override: ${OUTPUT:-<config output.path>}"
echo "Source metadata override: ${SOURCE_METADATA_OUTPUT:-<config source_metadata_output.path>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

CMD=("$PYTHON_BIN" -u -m tianqin_dc.minimal_catalog_aet_tq_numorbit --config "$CONFIG")
CMD+=(--workers "$WORKERS")
if [[ -n "${OUTPUT:-}" ]]; then
  CMD+=(--output "$OUTPUT")
fi
if [[ -n "${SOURCE_METADATA_OUTPUT:-}" ]]; then
  CMD+=(--source-metadata-output "$SOURCE_METADATA_OUTPUT")
fi
if [[ "${NO_SOURCE_METADATA_OUTPUT:-0}" == "1" ]]; then
  CMD+=(--no-source-metadata-output)
fi
if [[ -n "${MAX_SOURCES:-}" ]]; then
  CMD+=(--max-sources "$MAX_SOURCES")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "End time: $(date)"
