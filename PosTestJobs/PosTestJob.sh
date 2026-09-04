#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J PosTest
#SBATCH -o PosTestJobs/PosTest-%j.out
#SBATCH -e PosTestJobs/PosTest-%j.out
#SBATCH -t 00:40:00
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --network=disable_rdzv_get
#SBATCH -C nvme

set -euo pipefail

: "${POS_TEST_RUN_INDEX:?POS_TEST_RUN_INDEX must be set by run_sweep.sh}"
: "${POS_TEST_SAMPLES:?POS_TEST_SAMPLES must be set by run_sweep.sh}"
: "${POS_TEST_T_EXP:?POS_TEST_T_EXP must be set by run_sweep.sh}"
: "${POS_TEST_MAX_T:?POS_TEST_MAX_T must be set by run_sweep.sh}"

REPO_ROOT="${MATTERGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRIPT_DIR="${REPO_ROOT}/installation_scripts"
ENV_SOURCE_PATH="${MATTERGEN_ENV_PATH:-/lustre/orion/lrn070/proj-shared/${USER}/envs/mattergen-rocm711}"
ENV_ARCHIVE="${MATTERGEN_ENV_ARCHIVE:-${ENV_SOURCE_PATH}.tar.gz}"
LOCAL_ENV_ROOT="/mnt/bb/${USER}/mattergen-rocm711-${SLURM_JOB_ID}"
LOCAL_ENV_ARCHIVE="${LOCAL_ENV_ROOT}.tar.gz"
OMAT_DATA_PATH="${OMAT_DATA_PATH:-/lustre/orion/world-shared/lrn070/jyc/frontier/dataset_sc26_restripe/OMat24-v2.bp}"
POS_TEST_SEED="${POS_TEST_SEED:-0}"
POS_TEST_MAX_STEPS="${POS_TEST_MAX_STEPS:-600}"
POS_TEST_MAX_TRAIN_SECONDS="${POS_TEST_MAX_TRAIN_SECONDS:-1800}"
RUN_NAME="run-${POS_TEST_RUN_INDEX}-samples-${POS_TEST_SAMPLES}-t-exp-${POS_TEST_T_EXP}-${SLURM_JOB_ID}"
RANK_LOG_DIR="${REPO_ROOT}/jobOutputs/PosTest/${RUN_NAME}"
OUTPUT_DIR="${REPO_ROOT}/outputs/PosTest/${RUN_NAME}"

cd "${REPO_ROOT}"
[[ -f pyproject.toml && -d mattergen ]] || {
    echo "ERROR: submit PosTestJob.sh from the top level of the MatterGen checkout." >&2
    exit 1
}
[[ -s "${ENV_ARCHIVE}" ]] || {
    echo "ERROR: packed MatterGen environment not found: ${ENV_ARCHIVE}" >&2
    echo "Run installation_scripts/setup_mattergen_env_frontier_rocm711.sh first." >&2
    exit 1
}
[[ -e "${OMAT_DATA_PATH}" ]] || {
    echo "ERROR: OMat ADIOS dataset not found: ${OMAT_DATA_PATH}" >&2
    exit 1
}
[[ "${SLURM_NTASKS}" -eq $((SLURM_JOB_NUM_NODES * 8)) ]] || {
    echo "ERROR: expected 8 tasks per node; got ${SLURM_NTASKS} tasks on ${SLURM_JOB_NUM_NODES} nodes." >&2
    exit 1
}

mkdir -p "${RANK_LOG_DIR}" "${OUTPUT_DIR}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
    CONDA_PROMPT_MODIFIER 2>/dev/null || true
unset PYTHONHOME 2>/dev/null || true
export PYTHONNOUSERSITE=1

# Use the same tested Frontier software stack as JamieTest.sh.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module-to-load-frontier-rocm711.sh"
eval "$(conda shell.bash hook)"

echo "Broadcasting ${ENV_ARCHIVE} to node-local NVMe"
if ! sbcast -pf "${ENV_ARCHIVE}" "${LOCAL_ENV_ARCHIVE}"; then
    echo "ERROR: sbcast failed; refusing to use a potentially partial archive." >&2
    exit 1
fi

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

export PYTHONPATH="${REPO_ROOT}"
export REPO_ROOT PROJECT_ROOT="${REPO_ROOT}" OUTPUT_DIR
export MASTER_ADDR
MASTER_ADDR="$(hostname -i)"
MASTER_ADDR="${MASTER_ADDR%% *}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TMPDIR=/tmp
export MIOPEN_DISABLE_CACHE=1
export GPU_MAX_HW_QUEUES=2
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
export ROCFFT_RTC_CACHE_PATH=/dev/null
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Match JamieTest.sh's proven socket transport and avoid an inherited legacy
# ROCm 6.3.1 OFI plugin.
unset PATH_TO_THE_PLUGIN_DIRECTORY
filtered_ld_library_path=""
IFS=: read -r -a ld_library_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_library_entries[@]}"; do
    if [[ -n "${entry}" && "${entry}" != *AWI_OFI_RCCL_ROCm631* ]]; then
        filtered_ld_library_path="${filtered_ld_library_path:+${filtered_ld_library_path}:}${entry}"
    fi
done
export LD_LIBRARY_PATH="${filtered_ld_library_path}"
module unload rccl-net-plugin 2>/dev/null || true
unset NCCL_NET_PLUGIN NCCL_PROTO NCCL_ALGO
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH,COLL
export NCCL_DEBUG_FILE="${RANK_LOG_DIR}/rccl-%h-%p.log"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

echo "run=${POS_TEST_RUN_INDEX} samples=${POS_TEST_SAMPLES} seed=${POS_TEST_SEED}"
echo "timestep_range=[0,2^-${POS_TEST_T_EXP}]=[0,${POS_TEST_MAX_T}]"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} ranks=${SLURM_NTASKS}"
echo "max_steps=${POS_TEST_MAX_STEPS} max_train_seconds=${POS_TEST_MAX_TRAIN_SECONDS}"
echo "dataset=${OMAT_DATA_PATH}"
echo "environment=${ENV_ARCHIVE} staged_environment=${LOCAL_ENV_ROOT}"
echo "output=${OUTPUT_DIR} rank_logs=${RANK_LOG_DIR}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
module list

python - <<'PY'
from importlib import metadata
from pathlib import Path
import os
import sys

import adios2
import torch
import torch_cluster
import torch_geometric
import torch_scatter
import torch_sparse
import mattergen

expected = {
    "torch-geometric": "2.7.0",
    "torch-scatter-rocm": "2.1.2.post2",
    "torch-sparse-rocm": "0.6.18.post2",
    "torch-cluster-rocm": "1.6.3.post2",
    "adios2": "2.10.2.100774",
    "emmet-core": "0.85.1",
    "pymatgen": "2024.10.29",
    "monty": "2024.7.30",
}
for distribution, version in expected.items():
    actual = metadata.version(distribution)
    if actual != version:
        raise SystemExit(f"ERROR: expected {distribution} {version}, found {actual}")

try:
    conflicting_pymatgen = metadata.version("pymatgen-core")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit(
        f"ERROR: conflicting distribution pymatgen-core=={conflicting_pymatgen} "
        "is installed"
    )

from emmet.core.material import PropertyOrigin
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"ERROR: expected Python 3.12, found {sys.version.split()[0]}")
if torch.__version__.split("+", 1)[0] != "2.10.0":
    raise SystemExit(f"ERROR: expected torch 2.10.0, found {torch.__version__}")
if not (torch.version.hip or "").startswith("7.1"):
    raise SystemExit(f"ERROR: expected HIP 7.1.x, found {torch.version.hip!r}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no ROCm GPU is visible in the batch allocation")
if not torch.distributed.is_nccl_available():
    raise SystemExit("ERROR: PyTorch was installed without RCCL/NCCL support")

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
mattergen_file = Path(mattergen.__file__).resolve()
try:
    mattergen_file.relative_to(repo_root)
except ValueError as exc:
    raise SystemExit(
        f"ERROR: mattergen resolved outside this checkout: {mattergen_file}"
    ) from exc

source = torch.tensor([1.0, 2.0, 3.0], device="cuda")
index = torch.tensor([0, 1, 0], dtype=torch.long, device="cuda")
expected_scatter = torch.tensor([4.0, 2.0], device="cuda")
actual_scatter = torch_scatter.scatter(source, index, dim=0, dim_size=2, reduce="sum")
torch.cuda.synchronize()
if not torch.equal(actual_scatter, expected_scatter):
    raise SystemExit(f"ERROR: torch_scatter GPU result is incorrect: {actual_scatter}")

print("Frontier environment and GPU preflight passed")
print("python", sys.version.split()[0], sys.executable)
print("torch", torch.__version__, "HIP", torch.version.hip)
print("adios2", adios2.__version__)
print("mattergen", mattergen_file)
PY

run_pos_test_rank() {
    local rank="${SLURM_PROCID:-unknown}"
    local local_rank="${SLURM_LOCALID:-unknown}"
    local host
    local rc
    host="$(hostname)"

    export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
    mkdir -p "${MPLCONFIGDIR}"
    export MATTERGEN_RANK_TRACE_FILE="${RANK_LOG_DIR}/trace-rank-${rank}-${host}.log"
    local status_file="${RANK_LOG_DIR}/status-rank-${rank}-${host}.txt"
    printf 'state=started rank=%s local_rank=%s host=%s pid=%s time=%s\n' \
        "${rank}" "${local_rank}" "${host}" "$$" "$(date --iso-8601=seconds)" \
        > "${status_file}"

    set +e
    mattergen-train \
        "seed=${POS_TEST_SEED}" \
        data_module=OMat24-v2 \
        "data_module.root_dir=${OMAT_DATA_PATH}" \
        "data_module.train_dataset.max_samples=${POS_TEST_SAMPLES}" \
        data_module.val_dataset=null \
        checkpoint_path=null \
        auto_resume=false \
        trainer.devices=8 \
        "trainer.num_nodes=${SLURM_JOB_NUM_NODES}" \
        "trainer.max_steps=${POS_TEST_MAX_STEPS}" \
        trainer.check_val_every_n_epoch=0 \
        trainer.checkpoint.save_top_k=0 \
        trainer.checkpoint.save_last=false \
        trainer.checkpoint.every_n_epochs=0 \
        trainer.checkpoint.every_n_train_steps=0 \
        model_module.diffusion_module.loss_fn.weights.pos=1.0 \
        model_module.diffusion_module.loss_fn.weights.cell=0.0 \
        model_module.diffusion_module.loss_fn.weights.atomic_numbers=0.0 \
        "+model_module.diffusion_module.timestep_sampler={_target_:mattergen.diffusion.timestep_samplers.UniformTimestepSampler,min_t:0.0,max_t:${POS_TEST_MAX_T}}" \
        "native_trainer.max_train_seconds=${POS_TEST_MAX_TRAIN_SECONDS}" \
        native_trainer.debug_ddp=false \
        native_trainer.log_every_n_steps=1 \
        '~trainer.logger'
    rc=$?
    set -e

    if (( rc == 0 )); then
        printf 'state=completed rank=%s local_rank=%s host=%s rc=0 time=%s\n' \
            "${rank}" "${local_rank}" "${host}" "$(date --iso-8601=seconds)" \
            > "${status_file}"
    else
        printf 'state=failed rank=%s local_rank=%s host=%s rc=%s time=%s\n' \
            "${rank}" "${local_rank}" "${host}" "${rc}" \
            "$(date --iso-8601=seconds)" > "${status_file}"
    fi
    return "${rc}"
}
export -f run_pos_test_rank
export RANK_LOG_DIR POS_TEST_SEED POS_TEST_SAMPLES POS_TEST_MAX_STEPS
export POS_TEST_MAX_TRAIN_SECONDS POS_TEST_T_EXP POS_TEST_MAX_T OMAT_DATA_PATH

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
    bash -c run_pos_test_rank
srun_rc=$?
set -e

if (( srun_rc != 0 )); then
    echo "ERROR: position test failed with rc=${srun_rc}. Logs: ${RANK_LOG_DIR}" >&2
    shopt -s nullglob
    rank_output_files=("${RANK_LOG_DIR}"/slurm-rank-*.out)
    shopt -u nullglob
    for rank_output in "${rank_output_files[@]}"; do
        if [[ -s "${rank_output}" ]]; then
            echo "===== representative worker tail: ${rank_output} =====" >&2
            tail -n 200 "${rank_output}" >&2
            break
        fi
    done
    exit "${srun_rc}"
fi

python "${REPO_ROOT}/PosTestJobs/summarize_pos_loss.py" \
    --repo-root "${REPO_ROOT}" || \
    echo "WARNING: training succeeded, but loss summarization failed." >&2

echo "Position test completed successfully: ${RUN_NAME}"
