#!/bin/bash
#SBATCH -J tq2_min_sgwb
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
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/tianqin_dc/tianqin_tdi2_dt0p5s/minimal_sgwb_signal_all_aet.json}"
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
LOG_DIR="$RUN_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"
echo "Output override: ${OUTPUT:-<config output.path>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

CMD=("$PYTHON_BIN" -u -m tianqin_dc.minimal_sgwb_aet --config "$CONFIG")
if [[ -n "${OUTPUT:-}" ]]; then
  CMD+=(--output "$OUTPUT")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "End time: $(date)"
