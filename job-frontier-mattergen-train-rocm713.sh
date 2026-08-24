#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J mattergen-train-rocm713
#SBATCH -o jobOutputs/mattergen-train-rocm713-%j.out
#SBATCH -e jobOutputs/mattergen-train-rocm713-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

set -euo pipefail

# ---------------------------------------------------------------------------
# MatterGen training on Frontier — ROCm 7.13 environment
# ---------------------------------------------------------------------------
REPO_ROOT=/lustre/orion/lrn070/world-shared/mlupopa/MatterGen/mattergen
SCRIPT_DIR="${REPO_ROOT}/installation_scripts"
ENV_PATH="${REPO_ROOT}/MatterGen-Installation-Frontier/mattergen_venv_rocm713"

# Data module and options (override on submission if desired)
DATA_MODULE="${DATA_MODULE:-mp_20}"

cd "${REPO_ROOT}"
mkdir -p jobOutputs

# Clear any conda environment inherited from the login shell
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true

# Load the ROCm 7.13 module stack and activate the venv
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module-to-load-frontier-rocm713.sh"
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
which python

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
export MASTER_PORT=29500
echo "MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

# --- Runtime environment (validated ROCm 7.13 config from lumina-sdk) ---
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp

export ROCM_HOME=${ROCM_PATH}
export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# NCCL / RCCL: disable the aws-ofi-nccl plugin (fails CXI domain creation with
# RC -38 ENOSYS on this stack). RCCL falls back to its built-in transports:
# intra-node xGMI/SHM, inter-node TCP over the HSN NICs. Validated multi-node.
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_PLUGIN=none

echo "Job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} GPU ranks, data_module=${DATA_MODULE}"

srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    bash -c 'export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}; exec mattergen-train data_module='"${DATA_MODULE}"' ~trainer.logger'
