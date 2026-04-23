#!/bin/bash
# Kerr EMRI Fisher launcher for the cluster.
#
# Typical use:
#   sbatch examples/run_fisher_kerr_cluster.sh
#   FISHER_STEP_MODE=relative FISHER_REL_STEPS=1e-7,3e-7,1e-6 sbatch examples/run_fisher_kerr_cluster.sh
#   FISHER_TOBS_YR=0.1 FISHER_CORNER_TAG=T0p100yr_M_mu_a_p0_kerr_scaled sbatch examples/run_fisher_kerr_cluster.sh
#   FISHER_STEP_MODE=absolute FISHER_PARAM_STEPS=M=1.0,mu=4e-5,a=1e-5,p0=3e-5 FISHER_STEP_SCALES=0.3,1.0,3.0 sbatch examples/run_fisher_kerr_cluster.sh

#SBATCH -J tq_emri_fisher
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
RUN_ROOT="${RUN_ROOT:-/public/home/zhuangzhenye/jobs/gwspace_runs}"
LOG_DIR="$RUN_ROOT/logs"
FISHER_OUT_DIR="${FISHER_OUT_DIR:-$RUN_ROOT/fisher_results}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/zhuangzhenye/.conda/envs/gwspace312/bin/python}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_ROOT/examples/fisher.py}"

export GWSPACE_EMRI_MODEL="${GWSPACE_EMRI_MODEL:-FastKerrEccentricEquatorialFlux}"
export FISHER_TOBS_YR="${FISHER_TOBS_YR:-1.0}"
export FISHER_DT="${FISHER_DT:-20.0}"
export FISHER_PARAMS="${FISHER_PARAMS:-M,mu,a,p0}"
export FISHER_STEP_MODE="${FISHER_STEP_MODE:-relative}"
export FISHER_MAX_BINS="${FISHER_MAX_BINS:-40000}"
export FISHER_ENABLE_SCALED="${FISHER_ENABLE_SCALED:-1}"
export FISHER_CORNER_DIR="${FISHER_CORNER_DIR:-$FISHER_OUT_DIR}"
export FISHER_CORNER_TAG="${FISHER_CORNER_TAG:-T1p000yr_M_mu_a_p0_kerr_scaled}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/.mplconfig}"

case "$FISHER_STEP_MODE" in
  relative)
    export FISHER_REL_STEPS="${FISHER_REL_STEPS:-3e-7}"
    export FISHER_PARAM_STEPS=""
    export FISHER_STEP_SCALES=""
    ;;
  absolute)
    export FISHER_REL_STEPS="${FISHER_REL_STEPS:-1e-6}"
    export FISHER_PARAM_STEPS="${FISHER_PARAM_STEPS:-M=1.0,mu=4e-5,a=1e-5,p0=3e-5}"
    export FISHER_STEP_SCALES="${FISHER_STEP_SCALES:-0.3,1.0,3.0}"
    ;;
  *)
    echo "Unsupported FISHER_STEP_MODE=$FISHER_STEP_MODE (expected: relative or absolute)." >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR" "$FISHER_OUT_DIR" "$MPLCONFIGDIR"
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
echo "FISHER_TOBS_YR: $FISHER_TOBS_YR"
echo "FISHER_DT: $FISHER_DT"
echo "FISHER_PARAMS: $FISHER_PARAMS"
echo "FISHER_STEP_MODE: $FISHER_STEP_MODE"
echo "FISHER_REL_STEPS: $FISHER_REL_STEPS"
echo "FISHER_PARAM_STEPS: $FISHER_PARAM_STEPS"
echo "FISHER_STEP_SCALES: $FISHER_STEP_SCALES"
echo "FISHER_MAX_BINS: $FISHER_MAX_BINS"
echo "FISHER_ENABLE_SCALED: $FISHER_ENABLE_SCALED"
echo "FISHER_CORNER_DIR: $FISHER_CORNER_DIR"
echo "FISHER_CORNER_TAG: $FISHER_CORNER_TAG"

srun "$PYTHON_BIN" -u "$PYTHON_SCRIPT" "$@"

echo "End time: $(date)"
