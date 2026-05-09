#!/bin/bash
#SBATCH -J tq_smbhb_md
#SBATCH -p cpu_part
#SBATCH -w comput11
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.out
#SBATCH -e /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/zhuangzhenye/jobs/GWSpace}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/tianqin_dc/smbhb_multi_detector_aet.json}"
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$RUN_ROOT/logs"
OUT_DIR="$RUN_ROOT/outputs"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_DIR/smbhb_multi_detector_aet_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$OUTPUT_DIR"
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
echo "Output dir: $OUTPUT_DIR"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

CMD=("$PYTHON_BIN" -u -m tianqin_dc.multi_detector_aet --config "$CONFIG" --output-dir "$OUTPUT_DIR")
if [[ -n "${PREFIX:-}" ]]; then
  CMD+=(--prefix "$PREFIX")
fi
if [[ -n "${DURATION_S:-}" ]]; then
  CMD+=(--duration-s "$DURATION_S")
fi
if [[ -n "${SAMPLE_RATE_HZ:-}" ]]; then
  CMD+=(--sample-rate-hz "$SAMPLE_RATE_HZ")
fi
if [[ -n "${MAX_PLOT_POINTS:-}" ]]; then
  CMD+=(--max-plot-points "$MAX_PLOT_POINTS")
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "End time: $(date)"
