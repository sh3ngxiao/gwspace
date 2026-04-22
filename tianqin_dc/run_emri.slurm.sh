#!/bin/bash
#SBATCH -J tq_emri
#SBATCH -p cpu_part
#SBATCH -w comput11
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.out
#SBATCH -e /public/home/zhuangzhenye/jobs/gwspace_runs/logs/%x-%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/zhuangzhenye/jobs/GWSpace}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/tianqin_dc/emri_catalog_simple.json}"
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
LOG_DIR="$RUN_ROOT/logs"
OUT_DIR="$RUN_ROOT/outputs"
OUTPUT="${OUTPUT:-$OUT_DIR/emri_${SLURM_JOB_ID}.h5}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

srun "$PYTHON_BIN" -u -m tianqin_dc.emri_cli --config "$CONFIG" --output "$OUTPUT"

echo "End time: $(date)"

