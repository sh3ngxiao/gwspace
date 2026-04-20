#!/bin/bash
#SBATCH -J tq_emri
#SBATCH -p cpu_part
#SBATCH -w comput11
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/tianqin_dc/emri_catalog_simple.json}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/emri_${SLURM_JOB_ID}.h5}"
ENV_NAME="${ENV_NAME:-gwspace312}"

cd "$PROJECT_ROOT"
mkdir -p logs outputs

module purge
module load compiler/gcc-compiler/11.4.0
module load mathlib/gsl/gnu/2.6
module load mathlib/lapack/gnu/3.9.1

source ~/.bashrc
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(which python)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_NNODES: ${SLURM_NNODES:-N/A}"
echo "SLURM_TASKS_PER_NODE: ${SLURM_TASKS_PER_NODE:-N/A}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-N/A}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-N/A}"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

srun python -u -m tianqin_dc.emri_cli --config "$CONFIG" --output "$OUTPUT"

echo "End time: $(date)"
