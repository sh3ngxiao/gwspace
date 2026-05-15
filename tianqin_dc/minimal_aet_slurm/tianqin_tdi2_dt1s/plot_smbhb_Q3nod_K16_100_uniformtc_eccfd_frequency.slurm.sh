#!/bin/bash
#SBATCH -J tq_freq_q3n_u100
#SBATCH -p gpu_part
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.out
#SBATCH -e /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/zhuangzhenye/jobs/GWSpace}"
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
LOG_DIR="$RUN_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"

INPUT="${INPUT:-$RUN_ROOT/minimal_aet_tianqin_tdi2_dt1s/smbhb_Q3nod_K16_100_uniformtc_eccfd_aet.h5}"
OUTPUT="${OUTPUT:-${INPUT%.h5}_frequency.png}"
CHANNELS="${CHANNELS:-A,E,T}"
QUANTITY="${QUANTITY:-asd}"
WINDOW="${WINDOW:-hann}"
DETREND="${DETREND:-mean}"
MAX_PLOT_POINTS="${MAX_PLOT_POINTS:-50000}"
BIN_STAT="${BIN_STAT:-max}"
DPI="${DPI:-180}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

mkdir -p "$LOG_DIR"
mkdir -p "${MPLCONFIGDIR:-$RUN_ROOT/matplotlib_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib_cache}"

cd "$PROJECT_ROOT"

if type module >/dev/null 2>&1 && { command -v modulecmd >/dev/null 2>&1 || [[ -x /usr/bin/modulecmd ]]; }; then
  module load apps/anaconda202309 || echo "[WARN] Failed to load module apps/anaconda202309"
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Channels: $CHANNELS"
echo "Quantity: $QUANTITY"
echo "Window: $WINDOW"
echo "Detrend: $DETREND"
echo "Max plot points: $MAX_PLOT_POINTS"
echo "Bin stat: $BIN_STAT"
echo "Frequency min: ${F_MIN:-<auto>}"
echo "Frequency max: ${F_MAX:-<auto>}"
echo "Threads: $THREADS"

CMD=(
  "$PYTHON_BIN" -u -m tianqin_dc.plot_minimal_aet_frequency
  --input "$INPUT"
  --output "$OUTPUT"
  --channels "$CHANNELS"
  --quantity "$QUANTITY"
  --window "$WINDOW"
  --detrend "$DETREND"
  --max-plot-points "$MAX_PLOT_POINTS"
  --bin-stat "$BIN_STAT"
  --dpi "$DPI"
)
if [[ -n "${F_MIN:-}" ]]; then
  CMD+=(--f-min "$F_MIN")
fi
if [[ -n "${F_MAX:-}" ]]; then
  CMD+=(--f-max "$F_MAX")
fi
if [[ -n "${SAMPLE_SPACING_S:-}" ]]; then
  CMD+=(--sample-spacing-s "$SAMPLE_SPACING_S")
fi
if [[ -n "${TITLE:-}" ]]; then
  CMD+=(--title "$TITLE")
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"

echo "End time: $(date)"
