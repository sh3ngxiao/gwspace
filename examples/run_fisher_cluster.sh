#!/bin/bash
# ================= 资源申请部分 =================
#SBATCH --job-name=tq_emri_fim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=gw_output_%j.log
#SBATCH --error=gw_error_%j.log
##SBATCH --partition=compute

# ================= 环境加载与运行部分 =================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

: "${PYTHON_BIN:=python}"
: "${CONDA_ENV_NAME:=gwspace312}"
: "${PYTHON_SCRIPT:=examples/fisher.py}"
: "${FISHER_TOBS_YR:=5.0}"
: "${FISHER_DT:=20.0}"
: "${FISHER_MAX_BINS:=120000}"

export FISHER_TOBS_YR
export FISHER_DT
export FISHER_MAX_BINS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ "${PYTHON_SCRIPT}" = /* ]]; then
  SCRIPT_PATH="${PYTHON_SCRIPT}"
else
  SCRIPT_PATH="${REPO_DIR}/${PYTHON_SCRIPT}"
fi

WORK_DIR="${SLURM_SUBMIT_DIR:-${REPO_DIR}}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_DIR}/fisher_run_${RUN_ID}.log"

echo "任务开始运行时间: $(date)" | tee -a "${RUN_LOG}"

# 1. 加载必须的底层模块（若集群支持module）
if type module >/dev/null 2>&1; then
  module load apps/anaconda202309
  module load gsl2.7.1
else
  echo "[WARN] 当前环境没有module命令，跳过module load." | tee -a "${RUN_LOG}"
fi

# 2. 刷新配置并激活conda环境
if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "[WARN] 未找到conda命令，将直接使用当前python环境." | tee -a "${RUN_LOG}"
fi

# 3. 进入提交目录（无Slurm时回退到仓库目录）
cd "${WORK_DIR}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] 未找到Python脚本: ${SCRIPT_PATH}" | tee -a "${RUN_LOG}"
  exit 1
fi

echo "REPO_DIR=${REPO_DIR}" | tee -a "${RUN_LOG}"
echo "WORK_DIR=${WORK_DIR}" | tee -a "${RUN_LOG}"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${RUN_LOG}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}" | tee -a "${RUN_LOG}"
echo "SCRIPT_PATH=${SCRIPT_PATH}" | tee -a "${RUN_LOG}"
echo "FISHER_TOBS_YR=${FISHER_TOBS_YR}" | tee -a "${RUN_LOG}"
echo "FISHER_DT=${FISHER_DT}" | tee -a "${RUN_LOG}"
echo "FISHER_MAX_BINS=${FISHER_MAX_BINS}" | tee -a "${RUN_LOG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}" | tee -a "${RUN_LOG}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}" | tee -a "${RUN_LOG}"

# 4. 执行Python核心代码
"${PYTHON_BIN}" "${SCRIPT_PATH}" "$@" 2>&1 | tee -a "${RUN_LOG}"

echo "任务结束运行时间: $(date)" | tee -a "${RUN_LOG}"
echo "运行日志文件: ${RUN_LOG}" | tee -a "${RUN_LOG}"
