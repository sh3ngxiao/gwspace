#!/bin/bash
# Kerr EMRI Fisher reparameterization launcher for the cluster.
#
# Typical use:
#   sbatch examples/run_fisher_kerr_reparam_cluster.sh
#   FISHER_TOBS_YR=5.0 sbatch examples/run_fisher_kerr_reparam_cluster.sh
#   FISHER_STEP_MODE=relative FISHER_REL_STEPS=3e-8,1e-7,3e-7 FISHER_CORNER_TAG=T5p000yr_kerr_obs sbatch examples/run_fisher_kerr_reparam_cluster.sh
#   FISHER_STEP_MODE=absolute FISHER_PARAM_STEPS=M=1.0,mu=4e-5,a=1e-5,p0=3e-5 FISHER_STEP_SCALES=0.3,1.0,3.0 sbatch examples/run_fisher_kerr_reparam_cluster.sh

#SBATCH -J tq_emri_reparam
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
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_ROOT/examples/fisher_reparam.py}"

export GWSPACE_EMRI_MODEL="${GWSPACE_EMRI_MODEL:-FastKerrEccentricEquatorialFlux}"
export FISHER_TOBS_YR="${FISHER_TOBS_YR:-5.0}"
export FISHER_DT="${FISHER_DT:-20.0}"
export FISHER_PARAMS="${FISHER_PARAMS:-M,mu,a,p0}"
export FISHER_STEP_MODE="${FISHER_STEP_MODE:-relative}"
export FISHER_MAX_BINS="${FISHER_MAX_BINS:-400000}"
export FISHER_ENABLE_SCALED="${FISHER_ENABLE_SCALED:-1}"
export FISHER_CORNER_DIR="${FISHER_CORNER_DIR:-$FISHER_OUT_DIR/reparam}"
export FISHER_CORNER_TAG="${FISHER_CORNER_TAG:-T5p000yr_M_mu_a_p0_kerr_obs}"

export FISHER_EXP_RAW_PARAMS="${FISHER_EXP_RAW_PARAMS:-M,mu,a,p0}"
export FISHER_EXP_BASIS="${FISHER_EXP_BASIS:-kerr_circ_observables}"
export FISHER_EXP_COMPARE_PHYSICAL="${FISHER_EXP_COMPARE_PHYSICAL:-1}"
export FISHER_EXP_PLOT_CORNER="${FISHER_EXP_PLOT_CORNER:-1}"
export FISHER_EXP_CORNER_DIR="${FISHER_EXP_CORNER_DIR:-$FISHER_CORNER_DIR}"
export FISHER_EXP_CORNER_TAG="${FISHER_EXP_CORNER_TAG:-$FISHER_CORNER_TAG}"
export FISHER_EXP_BASIS_REL_STEP="${FISHER_EXP_BASIS_REL_STEP:-1e-6}"
export FISHER_EXP_FDOT_DT_SEC="${FISHER_EXP_FDOT_DT_SEC:-86400.0}"
export FISHER_EXP_FDOT_STEPS="${FISHER_EXP_FDOT_STEPS:-16}"

export FEW_FILE_STORAGE_PATH="${FEW_FILE_STORAGE_PATH:-$RUN_ROOT/few_data}"
export FEW_FILE_DOWNLOAD_PATH="${FEW_FILE_DOWNLOAD_PATH:-$RUN_ROOT/few_data/download}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/.mplconfig}"

case "$FISHER_STEP_MODE" in
  relative)
    export FISHER_REL_STEPS="${FISHER_REL_STEPS:-3e-8,1e-7,3e-7}"
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

mkdir -p "$LOG_DIR" "$FISHER_OUT_DIR" "$FISHER_CORNER_DIR" "$MPLCONFIGDIR" "$FEW_FILE_DOWNLOAD_PATH"
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
echo "FISHER_EXP_RAW_PARAMS: $FISHER_EXP_RAW_PARAMS"
echo "FISHER_EXP_BASIS: $FISHER_EXP_BASIS"
echo "FISHER_EXP_COMPARE_PHYSICAL: $FISHER_EXP_COMPARE_PHYSICAL"
echo "FISHER_EXP_PLOT_CORNER: $FISHER_EXP_PLOT_CORNER"
echo "FISHER_EXP_CORNER_DIR: $FISHER_EXP_CORNER_DIR"
echo "FISHER_EXP_CORNER_TAG: $FISHER_EXP_CORNER_TAG"
echo "FISHER_EXP_BASIS_REL_STEP: $FISHER_EXP_BASIS_REL_STEP"
echo "FISHER_EXP_FDOT_DT_SEC: $FISHER_EXP_FDOT_DT_SEC"
echo "FISHER_EXP_FDOT_STEPS: $FISHER_EXP_FDOT_STEPS"
echo "FEW_FILE_STORAGE_PATH: $FEW_FILE_STORAGE_PATH"
echo "FEW_FILE_DOWNLOAD_PATH: $FEW_FILE_DOWNLOAD_PATH"

srun "$PYTHON_BIN" -u "$PYTHON_SCRIPT" "$@"

echo "End time: $(date)"
