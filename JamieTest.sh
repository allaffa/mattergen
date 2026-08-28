#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J JamieTest
#SBATCH -o JamieTest-%j.out
#SBATCH -e JamieTest-%j.out
#SBATCH -t 01:00:00
#SBATCH -q debug
#SBATCH -N 64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --network=disable_rdzv_get
#SBATCH -C nvme

set -euo pipefail

# A 64-node submission launches 512 ranks. Override -N with sbatch when doing
# the matched 256-rank test; task counts are always derived from the allocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="${REPO_ROOT}/installation_scripts"
# The setup script creates this archive. It is broadcast once per node and
# unpacked onto Frontier's NVMe before Python starts.
ENV_SOURCE_PATH="${MATTERGEN_ENV_PATH:-/lustre/orion/lrn070/proj-shared/zb7/envs/mattergen-rocm711}"
ENV_ARCHIVE="${MATTERGEN_ENV_ARCHIVE:-${ENV_SOURCE_PATH}.tar.gz}"
LOCAL_ENV_ROOT="/mnt/bb/${USER}/mattergen-rocm711-${SLURM_JOB_ID}"
LOCAL_ENV_ARCHIVE="${LOCAL_ENV_ROOT}.tar.gz"
DATA_MODULE="${DATA_MODULE:-OMat24-v2}"
RANK_LOG_DIR="${REPO_ROOT}/jobOutputs/JamieTest-${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
[[ -f pyproject.toml && -d mattergen ]] || {
    echo "ERROR: submit JamieTest.sh from the top level of your MatterGen checkout." >&2
    exit 1
}
mkdir -p "${RANK_LOG_DIR}"
[[ -s "${ENV_ARCHIVE}" ]] || {
    echo "ERROR: packed MatterGen environment not found: ${ENV_ARCHIVE}" >&2
    echo "Run installation_scripts/setup_mattergen_env_frontier_rocm711.sh first." >&2
    exit 1
}

# Do not let a Conda environment inherited from the submission shell affect
# module loading or activation of the known-working shared environment.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
    CONDA_PROMPT_MODIFIER 2>/dev/null || true
unset PYTHONHOME 2>/dev/null || true
export PYTHONNOUSERSITE=1

# Load OLCF's recommended PyTorch 2.10/ROCm 7.1.1 stack.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module-to-load-frontier-rocm711.sh"
eval "$(conda shell.bash hook)"

echo "Broadcasting ${ENV_ARCHIVE} to node-local NVMe"
if ! sbcast -pf "${ENV_ARCHIVE}" "${LOCAL_ENV_ARCHIVE}"; then
    echo "ERROR: sbcast failed; refusing to use a potentially partial archive." >&2
    exit 1
fi

# One task per node creates and relocates its local copy. All nodes use the
# same path, so activation in the batch shell exports a valid path to srun.
srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    mkdir -p "${LOCAL_ENV_ROOT}"
srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=56 \
    tar --use-compress-program=pigz -xf "${LOCAL_ENV_ARCHIVE}" -C "${LOCAL_ENV_ROOT}"

conda activate "${LOCAL_ENV_ROOT}"
srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    conda-unpack

# Only the submitted checkout and the staged environment participate in
# imports. Do not inherit a different checkout or user-site installation.
export PYTHONPATH="${REPO_ROOT}"
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
# Do not inherit diagnostic overrides from the submission shell. Let RCCL's
# internal tuner choose the protocol and algorithm for each collective.
unset NCCL_PROTO NCCL_ALGO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH,COLL
export NCCL_DEBUG_FILE="${RANK_LOG_DIR}/rccl-%h-%p.log"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

echo "repo=${REPO_ROOT}"
echo "python=$(command -v python)"
echo "conda_prefix=${CONDA_PREFIX:-<unset>}"
echo "env_archive=${ENV_ARCHIVE} staged_env=${LOCAL_ENV_ROOT}"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS}"
echo "master=${MASTER_ADDR}:${MASTER_PORT} data_module=${DATA_MODULE}"
echo "rank_logs=${RANK_LOG_DIR}"
echo "rccl_net=${NCCL_NET} rccl_net_plugin=${NCCL_NET_PLUGIN:-<module/default>}"
echo "gpu_visibility_before_srun: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<set-by-slurm-per-task>} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "effective Frontier network environment:"
env | LC_ALL=C sort | grep -E '^(NCCL_|FI_CXI_|FI_MR_)' || true
module list

python - <<'PY'
from importlib import metadata
import os
from pathlib import Path
import sys

import adios2
import torch

expected_distributions = {
    "torch-geometric": "2.7.0",
    "torch-scatter-rocm": "2.1.2.post2",
    "torch-sparse-rocm": "0.6.18.post2",
    "torch-cluster-rocm": "1.6.3.post2",
    "adios2": "2.10.2.100774",
}
for distribution, expected_version in expected_distributions.items():
    installed_version = metadata.version(distribution)
    if installed_version != expected_version:
        raise SystemExit(
            f"ERROR: expected {distribution} {expected_version}, "
            f"found {installed_version}"
        )
    print("distribution", distribution, installed_version, flush=True)

for conflicting_distribution in (
    "pyg-lib",
    "pyg-lib-rocm",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-spline-conv-rocm",
):
    try:
        installed_version = metadata.version(conflicting_distribution)
    except metadata.PackageNotFoundError:
        continue
    raise SystemExit(
        f"ERROR: conflicting distribution is installed: "
        f"{conflicting_distribution}=={installed_version}"
    )

print("import torch_scatter", flush=True)
import torch_scatter
print("import torch_sparse", flush=True)
import torch_sparse
print("import torch_cluster", flush=True)
import torch_cluster
print("import torch_geometric", flush=True)
import torch_geometric


def verify_scatter(device: str) -> None:
    source = torch.tensor([1.0, 2.0, 3.0], device=device)
    index = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
    result = torch_scatter.scatter(
        source, index, dim=0, dim_size=2, reduce="sum"
    )
    expected = torch.tensor([4.0, 2.0], device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    if not torch.equal(result.cpu(), expected.cpu()):
        raise SystemExit(
            f"ERROR: torch_scatter produced an incorrect {device} result: {result}"
        )
    print("torch_scatter", device, "verification passed", flush=True)


verify_scatter("cpu")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: no ROCm GPU is visible in the batch allocation")
verify_scatter("cuda")

print("import local MatterGen checkout", flush=True)
import mattergen

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
print("python", sys.version.split()[0], sys.executable)
print("torch", torch.__version__, "HIP", torch.version.hip)
print("adios2", adios2.__version__, adios2.__file__)
print("torch_geometric", torch_geometric.__version__)

torch_release = torch.__version__.split("+", 1)[0]
hip_release = torch.version.hip or ""
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"ERROR: expected Python 3.12, found {sys.version.split()[0]}")
if torch_release != "2.10.0":
    raise SystemExit(f"ERROR: expected torch 2.10.0, found {torch.__version__}")
if not hip_release.startswith("7.1"):
    raise SystemExit(f"ERROR: expected HIP 7.1.x, found {torch.version.hip!r}")
if not adios2.__version__.startswith("2.10.2"):
    raise SystemExit(
        f"ERROR: expected the adios2 2.10.2 series, found {adios2.__version__}"
    )
if not torch.distributed.is_nccl_available():
    raise SystemExit("ERROR: PyTorch was installed without RCCL/NCCL support")
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

# So that it doesn't hang
module unload rccl-net-plugin 2>/dev/null || true
unset NCCL_NET_PLUGIN
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

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
