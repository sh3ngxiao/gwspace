#!/bin/bash
# Kerr EMRI frequency/strain plot launcher for the cluster.
#
# Typical use:
#   sbatch examples/run_frequency_kerr_cluster.sh
#   GWSPACE_EMRI_MODEL=FastKerrEccentricEquatorialFlux sbatch examples/run_frequency_kerr_cluster.sh

#SBATCH -J tq_emri_freq
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
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
LOG_DIR="$RUN_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_ROOT/examples/frequency.py}"

export GWSPACE_EMRI_MODEL="${GWSPACE_EMRI_MODEL:-FastKerrEccentricEquatorialFlux}"
export FEW_FILE_STORAGE_PATH="${FEW_FILE_STORAGE_PATH:-$RUN_ROOT/few_data}"
export FEW_FILE_DOWNLOAD_PATH="${FEW_FILE_DOWNLOAD_PATH:-$RUN_ROOT/few_data/download}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/.mplconfig}"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR" "$FEW_FILE_DOWNLOAD_PATH"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "RUN_ROOT: $RUN_ROOT"
echo "Python: $PYTHON_BIN"
echo "Python script: $PYTHON_SCRIPT"
echo "GWSPACE_EMRI_MODEL: $GWSPACE_EMRI_MODEL"
echo "FEW_FILE_STORAGE_PATH: $FEW_FILE_STORAGE_PATH"
echo "FEW_FILE_DOWNLOAD_PATH: $FEW_FILE_DOWNLOAD_PATH"
echo "MPLBACKEND: $MPLBACKEND"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"

srun "$PYTHON_BIN" -u "$PYTHON_SCRIPT" "$@"

echo "End time: $(date)"
