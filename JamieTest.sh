#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J JamieTest
#SBATCH -o JamieTest-%j.out
#SBATCH -e JamieTest-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

set -euo pipefail

# A 64-node submission launches 512 ranks. Override -N with sbatch when doing
# the matched 256-rank test; task counts are always derived from the allocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_PATH=/lustre/orion/lrn070/proj-shared/patxi/envs/hydragenn_mattergen720
DATA_MODULE="${DATA_MODULE:-OMat24-v2}"
RANK_LOG_DIR="${REPO_ROOT}/jobOutputs/JamieTest-${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
mkdir -p "${RANK_LOG_DIR}"

# Load the ROCm 7.2 stack, then explicitly activate the shared environment.
# shellcheck disable=SC1091
source "${REPO_ROOT}/installation_scripts/module-to-load-frontier-rocm720.sh"
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PROJECT_ROOT="${REPO_ROOT}"
export OUTPUT_DIR="${REPO_ROOT}/outputs/JamieTest-${SLURM_JOB_ID}"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | sed -n '1p')"
export MASTER_PORT="${MASTER_PORT:-29500}"

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MIOPEN_DISABLE_CACHE=1
export GPU_MAX_HW_QUEUES=2
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

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

# First-pass diagnostics use the matched plugin with no hand-tuned FI/CXI/GDR
# overrides. RCCL writes one detailed log per process.
unset FI_MR_CACHE_MONITOR FI_CXI_RDV_PROTO FI_CXI_DEFAULT_CQ_SIZE
unset FI_CXI_DEFAULT_TX_SIZE FI_CXI_RX_MATCH_MODE NCCL_NET_GDR_LEVEL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH
export NCCL_DEBUG_FILE="${RANK_LOG_DIR}/rccl-%h-%p.log"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

echo "repo=${REPO_ROOT}"
echo "python=$(command -v python)"
echo "conda_prefix=${CONDA_PREFIX:-<unset>}"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS}"
echo "master=${MASTER_ADDR}:${MASTER_PORT} data_module=${DATA_MODULE}"
echo "rank_logs=${RANK_LOG_DIR}"
module list

python -c 'import adios2, mattergen, torch; print("mattergen", mattergen.__file__); print("torch", torch.__version__, "HIP", torch.version.hip); print("adios2", adios2.__version__, adios2.__file__)'

run_mattergen_rank() {
    local rank="${SLURM_PROCID:-unknown}"
    local local_rank="${SLURM_LOCALID:-unknown}"
    local host
    local rc
    host="$(hostname)"

    export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"
    export MATTERGEN_RANK_TRACE_FILE="${RANK_LOG_DIR}/trace-rank-${rank}-${host}.log"
    local status_file="${RANK_LOG_DIR}/status-rank-${rank}-${host}.txt"

    printf 'state=started rank=%s local_rank=%s host=%s pid=%s time=%s\n' \
        "${rank}" "${local_rank}" "${host}" "$$" "$(date --iso-8601=seconds)" \
        > "${status_file}"

    set +e
    mattergen-train \
        "data_module=${DATA_MODULE}" \
        auto_resume=false \
        checkpoint_path=null \
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
srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    --wait=30 \
    --output="${RANK_LOG_DIR}/slurm-rank-%t-%N.out" \
    --error="${RANK_LOG_DIR}/slurm-rank-%t-%N.out" \
    bash -c run_mattergen_rank
