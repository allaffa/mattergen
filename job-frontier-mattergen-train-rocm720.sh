#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J mattergen-train
#SBATCH -o jobOutputs/mattergen-train-%j.out
#SBATCH -e jobOutputs/mattergen-train-%j.out
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 512
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

set -euo pipefail

# ---------------------------------------------------------------------------
# MatterGen training on Frontier
# ---------------------------------------------------------------------------
# REPO_ROOT must be the directory that CONTAINS the top-level "mattergen/"
# package. We prepend it to PYTHONPATH so Python imports the local MatterGen
# checkout first. This is useful for debugging local code instead of picking up
# another installed/editable MatterGen from elsewhere.
REPO_ROOT=/lustre/orion/lrn070/proj-shared/patxi/BatchMeanPatxi_Max
ENV_PATH=/lustre/orion/lrn070/proj-shared/patxi/envs/HydraGNN-Installation-Frontier/hydragnn_venv
SCRIPT_DIR=/lustre/orion/lrn070/proj-shared/patxi/BatchMeanPatxi_Max/installation_scripts

# Options (override on submission if desired)
DATA_MODULE="${DATA_MODULE:-OMat24-v2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

cd "${REPO_ROOT}"
mkdir -p jobOutputs

# Clear any conda environment inherited from the login shell
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true

# Load modules and activate the conda env
source "${SCRIPT_DIR}/module-to-load-frontier-rocm720.sh"
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
which python

# ---------------------------------------------------------------------------
# Force Python to search the local repo first
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PROJECT_ROOT="${REPO_ROOT}"

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
export MASTER_PORT=29500
echo "MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

# --- Runtime environment ---
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp

export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# export NCCL_NET_PLUGIN=none


# ## Getting error without these after 20 nodes
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export FI_MR_CACHE_MONITOR=disabled
 
# # these were settings including in pei's original submission scripts
# #export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
# export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
# export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
# export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
# # export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
# # export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
# # export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.



echo "Job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} GPU ranks, data_module=${DATA_MODULE}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH:-<none>}"

export DATA_MODULE
export CHECKPOINT_PATH

ml rccl-net-plugin

srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    bash -c '
        export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
        export PYTHONPATH="'"${PYTHONPATH}"'"
        export PROJECT_ROOT="'"${PROJECT_ROOT}"'"

        python - <<'"'"'PY'"'"'
import os, sys
import mattergen
print("SRUN DEBUG mattergen.__file__ =", mattergen.__file__)
print("SRUN DEBUG PROJECT_ROOT       =", os.environ.get("PROJECT_ROOT"))
print("SRUN DEBUG sys.path[:10]      =", sys.path[:10])
PY

        if [ -n "${CHECKPOINT_PATH}" ]; then
            exec mattergen-train data_module="${DATA_MODULE}" checkpoint_path="${CHECKPOINT_PATH}" ~trainer.logger
        else
            exec mattergen-train data_module="${DATA_MODULE}" ~trainer.logger
        fi
    '
