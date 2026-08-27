#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J JamieTest
#SBATCH -o JamieTest-%j.out
#SBATCH -e JamieTest-%j.out
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -N 64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --network=disable_rdzv_get

set -euo pipefail

# A 64-node submission launches 512 ranks. Override -N with sbatch when doing
# the matched 256-rank test; task counts are always derived from the allocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="${REPO_ROOT}/installation_scripts"
# This is the ROCm 7.2 environment used by the known-working Frontier job.
# MATTERGEN_ENV_PATH can override it without editing this script.
ENV_PATH="${MATTERGEN_ENV_PATH:-/lustre/orion/lrn070/proj-shared/patxi/envs/HydraGNN-Installation-Frontier/hydragnn_venv}"
DATA_MODULE="${DATA_MODULE:-OMat24-v2}"
RANK_LOG_DIR="${REPO_ROOT}/jobOutputs/JamieTest-${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
[[ -f pyproject.toml && -d mattergen ]] || {
    echo "ERROR: submit JamieTest.sh from the top level of your MatterGen checkout." >&2
    exit 1
}
mkdir -p "${RANK_LOG_DIR}"

# Do not let a Conda environment inherited from the submission shell affect
# module loading or activation of the known-working shared environment.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
    CONDA_PROMPT_MODIFIER 2>/dev/null || true

# Load the same ROCm 7.2 stack and environment as the working reference job.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module-to-load-frontier-rocm720.sh"
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export REPO_ROOT
export PROJECT_ROOT="${REPO_ROOT}"
export OUTPUT_DIR="${REPO_ROOT}/outputs/JamieTest-${SLURM_JOB_ID}"

export MASTER_ADDR
MASTER_ADDR="$(hostname -i)"
export MASTER_PORT="${MASTER_PORT:-3442}"

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TMPDIR=/tmp
export MIOPEN_DISABLE_CACHE=1
export GPU_MAX_HW_QUEUES=2
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Use Frontier's module-provided RCCL network plugin. In particular, do not
# prepend the legacy ROCm 6.3.1 aws-ofi-rccl build used by submit.sh.
unset PATH_TO_THE_PLUGIN_DIRECTORY
filtered_ld_library_path=""
IFS=: read -r -a ld_library_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_library_entries[@]}"; do
    if [[ -n "${entry}" && "${entry}" != *AWI_OFI_RCCL_ROCm631* ]]; then
        filtered_ld_library_path="${filtered_ld_library_path:+${filtered_ld_library_path}:}${entry}"
    fi
done
export LD_LIBRARY_PATH="${filtered_ld_library_path}"
module load rccl-net-plugin

# Preserve every Slingshot/libfabric setting supplied by rccl-net-plugin. Force
# OFI so a missing multi-node network plugin fails instead of using sockets.
export NCCL_NET=OFI
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH
export NCCL_DEBUG_FILE="${RANK_LOG_DIR}/rccl-%h-%p.log"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000

echo "repo=${REPO_ROOT}"
echo "python=$(command -v python)"
echo "conda_prefix=${CONDA_PREFIX:-<unset>}"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS}"
echo "master=${MASTER_ADDR}:${MASTER_PORT} data_module=${DATA_MODULE}"
echo "rank_logs=${RANK_LOG_DIR}"
echo "rccl_net=${NCCL_NET} rccl_net_plugin=${NCCL_NET_PLUGIN:-<module/default>}"
echo "gpu_visibility_before_srun: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<set-by-slurm-per-task>} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "effective Frontier network environment:"
env | LC_ALL=C sort | grep -E '^(NCCL_|FI_CXI_|FI_MR_)' || true
module list

python - <<'PY'
import os
from pathlib import Path

import adios2
import mattergen
import torch

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
mattergen_file = Path(mattergen.__file__).resolve()
try:
    mattergen_file.relative_to(repo_root)
except ValueError:
    raise SystemExit(
        f"ERROR: mattergen resolved outside this checkout: {mattergen_file} "
        f"(expected a path under {repo_root})"
    )

print("mattergen", mattergen_file)
print("torch", torch.__version__, "HIP", torch.version.hip)
print("adios2", adios2.__version__, adios2.__file__)
PY

run_mattergen_rank() {
    local rank="${SLURM_PROCID:-unknown}"
    local local_rank="${SLURM_LOCALID:-unknown}"
    local host
    local rc
    host="$(hostname)"

    # Slurm's closest binding exposes the topology-correct GPU as cuda:0.
    printf 'gpu_binding rank=%s local_rank=%s host=%s ROCR_VISIBLE_DEVICES=%s HIP_VISIBLE_DEVICES=%s CUDA_VISIBLE_DEVICES=%s\n' \
        "${rank}" "${local_rank}" "${host}" \
        "${ROCR_VISIBLE_DEVICES:-<unset>}" "${HIP_VISIBLE_DEVICES:-<unset>}" \
        "${CUDA_VISIBLE_DEVICES:-<unset>}"
    # One node-local Matplotlib cache per node prevents hundreds of ranks from
    # contending on the shared home-directory font cache during imports.
    export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
    mkdir -p "${MPLCONFIGDIR}"
    export MATTERGEN_RANK_TRACE_FILE="${RANK_LOG_DIR}/trace-rank-${rank}-${host}.log"
    local status_file="${RANK_LOG_DIR}/status-rank-${rank}-${host}.txt"

    printf 'state=started rank=%s local_rank=%s host=%s pid=%s time=%s\n' \
        "${rank}" "${local_rank}" "${host}" "$$" "$(date --iso-8601=seconds)" \
        > "${status_file}"

    set +e
    mattergen-train \
        "data_module=${DATA_MODULE}" \
        auto_resume=false \
        trainer.devices=8 \
        "trainer.num_nodes=${SLURM_JOB_NUM_NODES}" \
        native_trainer.debug_ddp=true \
        native_trainer.debug_ddp_steps=2 \
        '~trainer.logger'
    rc=$?
    set -e

    if (( rc == 0 )); then
        printf 'state=completed rank=%s local_rank=%s host=%s rc=0 time=%s\n' \
            "${rank}" "${local_rank}" "${host}" "$(date --iso-8601=seconds)" \
            > "${status_file}"
    elif (( rc > 128 )); then
        printf 'state=failed rank=%s local_rank=%s host=%s rc=%s signal=%s time=%s\n' \
            "${rank}" "${local_rank}" "${host}" "${rc}" "$((rc - 128))" \
            "$(date --iso-8601=seconds)" > "${status_file}"
    else
        printf 'state=failed rank=%s local_rank=%s host=%s rc=%s time=%s\n' \
            "${rank}" "${local_rank}" "${host}" "${rc}" \
            "$(date --iso-8601=seconds)" > "${status_file}"
    fi
    return "${rc}"
}
export -f run_mattergen_rank
export RANK_LOG_DIR DATA_MODULE

# A nonzero task exits the step promptly. Slurm stdout/stderr, Python
# breadcrumbs, RCCL logs, and final status markers are all rank-specific.
set +e
srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --kill-on-bad-exit=1 \
    --wait=30 \
    --output="${RANK_LOG_DIR}/slurm-rank-%t-%N.out" \
    --error="${RANK_LOG_DIR}/slurm-rank-%t-%N.out" \
    bash -c run_mattergen_rank
srun_rc=$?
set -e

if (( srun_rc != 0 )); then
    echo "ERROR: srun failed with rc=${srun_rc}. Rank logs: ${RANK_LOG_DIR}" >&2

    # Keep rank-local files for full diagnostics, but also surface one complete
    # error tail in the main Slurm output so failures are not hidden behind the
    # generic "tasks ... Exited with exit code" messages.
    shopt -s nullglob
    rank_output_files=("${RANK_LOG_DIR}"/slurm-rank-*.out)
    shopt -u nullglob
    for rank_output in "${rank_output_files[@]}"; do
        if [[ -s "${rank_output}" ]]; then
            echo "===== representative worker log: ${rank_output} (last 200 lines) =====" >&2
            tail -n 200 "${rank_output}" >&2
            break
        fi
    done

    echo "Run: python analyze_jamie_test.py ${RANK_LOG_DIR}" >&2
    exit "${srun_rc}"
fi
