#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J mattergen-train
#SBATCH -o jobOutputs/mattergen-train-%j.out
#SBATCH -e jobOutputs/mattergen-train-%j.out
#SBATCH -t 00:45:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --network=disable_rdzv_get
#SBATCH -C nvme

set -euo pipefail

# ---------------------------------------------------------------------------
# MatterGen training on Frontier with ROCm 7.1.1
# ---------------------------------------------------------------------------
#
# was getting some annoying matplotlib errors 
#
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/$USER/mpl-cache
mkdir -p "$MPLCONFIGDIR"


# REPO_ROOT must be the directory that CONTAINS the top-level "mattergen/"
# package. We prepend it to PYTHONPATH so Python imports the local MatterGen
# checkout first.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${REPO_ROOT}/installation_scripts"

[[ -f "${REPO_ROOT}/pyproject.toml" && -d "${REPO_ROOT}/mattergen" ]] || {
    echo "ERROR: submit this job from the MatterGen repository root." >&2
    exit 1
}

# Packed ROCm 7.1.1 environment created by your setup script
ENV_SOURCE_PATH="${MATTERGEN_ENV_PATH:-/lustre/orion/lrn070/proj-shared/${USER}/envs/mattergen-rocm711}"
ENV_ARCHIVE="${MATTERGEN_ENV_ARCHIVE:-${ENV_SOURCE_PATH}.tar.gz}"

# Node-local NVMe staging location
LOCAL_ENV_ROOT="/mnt/bb/${USER}/mattergen-rocm711-${SLURM_JOB_ID}"
LOCAL_ENV_ARCHIVE="${LOCAL_ENV_ROOT}.tar.gz"

# Options (override on submission if desired)
DATA_MODULE="${DATA_MODULE:-OMat24-v2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

cd "${REPO_ROOT}"
mkdir -p jobOutputs
[[ -s "${ENV_ARCHIVE}" ]] || {
    echo "ERROR: packed MatterGen environment not found: ${ENV_ARCHIVE}" >&2
    echo "Run installation_scripts/setup_mattergen_env_frontier_rocm711.sh first." >&2
    exit 1
}

# Clear any conda environment inherited from the login shell
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
    CONDA_PROMPT_MODIFIER 2>/dev/null || true
unset PYTHONHOME 2>/dev/null || true
export PYTHONNOUSERSITE=1

# Load ROCm 7.1.1 modules
source "${SCRIPT_DIR}/module-to-load-frontier-rocm711.sh"
eval "$(conda shell.bash hook)"

echo "Broadcasting ${ENV_ARCHIVE} to node-local NVMe"
if ! sbcast -pf "${ENV_ARCHIVE}" "${LOCAL_ENV_ARCHIVE}"; then
    echo "ERROR: sbcast failed; refusing to use a potentially partial archive." >&2
    exit 1
fi

# Create and unpack local env on each node
srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --gpu-bind=none \
    mkdir -p "${LOCAL_ENV_ROOT}"

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=56 \
    --gpu-bind=none \
    tar --use-compress-program=pigz -xf "${LOCAL_ENV_ARCHIVE}" -C "${LOCAL_ENV_ROOT}"

conda activate "${LOCAL_ENV_ROOT}"

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --gpu-bind=none \
    conda-unpack

which python

# ---------------------------------------------------------------------------
# Force Python to search the local repo first
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}"
export PROJECT_ROOT="${REPO_ROOT}"
export REPO_ROOT="${REPO_ROOT}"

# Import sanity check in the launcher environment
python - <<'PY'
import os, sys
import mattergen
print("DEBUG mattergen.__file__ =", mattergen.__file__)
print("DEBUG PROJECT_ROOT       =", os.environ.get("PROJECT_ROOT"))
print("DEBUG PYTHONPATH         =", os.environ.get("PYTHONPATH"))
print("DEBUG sys.path[:10]      =", sys.path[:10])
PY

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
echo "MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TMPDIR=/tmp

export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Remove any legacy ROCm 6.x aws-ofi-rccl plugin path contamination
unset PATH_TO_THE_PLUGIN_DIRECTORY
filtered_ld_library_path=""
IFS=: read -r -a ld_library_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_library_entries[@]}"; do
    if [[ -n "${entry}" && "${entry}" != *AWI_OFI_RCCL_ROCm631* ]]; then
        filtered_ld_library_path="${filtered_ld_library_path:+${filtered_ld_library_path}:}${entry}"
    fi
done
export LD_LIBRARY_PATH="${filtered_ld_library_path}"

# ROCm 7.1.1 settings from your working test script:
# use RCCL socket transport instead of rccl-net-plugin/OFI
module unload rccl-net-plugin 2>/dev/null || true
unset NCCL_NET_PLUGIN
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
unset NCCL_PROTO NCCL_ALGO
#export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH,COLL
# export NCCL_DEBUG_FILE="${REPO_ROOT}/jobOutputs/rccl-%h-%p.log"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

echo "Job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} GPU ranks, data_module=${DATA_MODULE}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH:-<none>}"
echo "repo=${REPO_ROOT}"
echo "python=$(command -v python)"
echo "conda_prefix=${CONDA_PREFIX:-<unset>}"
echo "env_archive=${ENV_ARCHIVE} staged_env=${LOCAL_ENV_ROOT}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "effective Frontier network environment:"
env | LC_ALL=C sort | grep -E '^(NCCL_|FI_CXI_|FI_MR_)' || true
module list

export DATA_MODULE
export CHECKPOINT_PATH

srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --kill-on-bad-exit=1 \
    --wait=30 \
    bash -c '
        export PYTHONPATH="'"${PYTHONPATH}"'"
        export PROJECT_ROOT="'"${PROJECT_ROOT}"'"
        export REPO_ROOT="'"${REPO_ROOT}"'"
        export MPLCONFIGDIR="/tmp/matplotlib-'"${SLURM_JOB_ID}"'"
        mkdir -p "${MPLCONFIGDIR}"

        python - <<'"'"'PY'"'"'
import os, sys
import mattergen
print("SRUN DEBUG mattergen.__file__ =", mattergen.__file__)
print("SRUN DEBUG PROJECT_ROOT       =", os.environ.get("PROJECT_ROOT"))
print("SRUN DEBUG sys.path[:10]      =", sys.path[:10])
PY

        if [ -n "${CHECKPOINT_PATH}" ]; then
            exec mattergen-train \
                data_module="${DATA_MODULE}" \
                checkpoint_path="${CHECKPOINT_PATH}" \
                ~trainer.logger
        else
            exec mattergen-train \
                data_module="${DATA_MODULE}" \
                ~trainer.logger
        fi
    '
