#!/usr/bin/env bash
set -euo pipefail

# Local generation analogue of runJogo.sh. Set MODEL_PATH explicitly for the
# checkpoint or model directory to sample from.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"

# The requested path points at the env's Python executable. Conda activation
# needs the env directory, so derive that while still using the requested Python.
CONDA_PYTHON="${CONDA_PYTHON:-/home/3pf/miniconda3/envs/hydragnn/bin/python}"
CONDA_ENV="${CONDA_ENV:-$(cd -- "$(dirname -- "${CONDA_PYTHON}")/.." && pwd)}"

MODEL_PATH="${MODEL_PATH:-/home/3pf/projects/lrn070/mattergen/outputs/singlerun/2026-09-14/10-43-29}"
DATA_MODULE="${DATA_MODULE:-miniOmat24}"
BATCH_SIZE="${BATCH_SIZE:-36}"
NUM_BATCHES="${NUM_BATCHES:-1}"
NUM_ATOMS_DISTRIBUTION="${NUM_ATOMS_DISTRIBUTION:-Omat24}"
CUDA_DEVICE="${CUDA_DEVICE:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

cd "${REPO_ROOT}"

data_module_config="${REPO_ROOT}/mattergen/conf/data_module/${DATA_MODULE}.yaml"
if [[ ! -f "${data_module_config}" ]]; then
    echo "ERROR: data module config not found: ${data_module_config}" >&2
    exit 1
fi

root_dir="$(
    awk '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*root_dir[[:space:]]*:/ {
            sub(/^[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]]+#.*$/, "")
            gsub(/^[[:space:]]+|[[:space:]]+$/, "")
            gsub(/^["'\''"]|["'\''"]$/, "")
            print
            exit
        }
    ' "${data_module_config}"
)"

if [[ -z "${root_dir}" ]]; then
    echo "ERROR: root_dir not found in ${data_module_config}" >&2
    exit 1
fi

RESULTS_PATH="${RESULTS_PATH:-results/unconditioned}"

mkdir -p jobOutputs "${RESULTS_PATH}"
RUN_LOG="${RUN_LOG:-jobOutputs/genJogo-$(date +%Y%m%d-%H%M%S).out}"
exec > >(tee -a "${RUN_LOG}") 2>&1

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true

if [[ -f "/home/3pf/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/home/3pf/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

conda activate "${CONDA_ENV}"

# Prefer this checkout over any installed MatterGen package so local edits win.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PROJECT_ROOT="${REPO_ROOT}"

export MODEL_PATH
export DATA_MODULE
export RESULTS_PATH
export CUDA_DEVICE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "Using Python: $(which python)"
echo "CUDA_DEVICE=${CUDA_DEVICE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_MODULE=${DATA_MODULE}"
echo "Data module config: ${data_module_config}"
echo "Data root_dir=${root_dir}"
echo "RESULTS_PATH=${RESULTS_PATH}"
echo "NUM_ATOMS_DISTRIBUTION=${NUM_ATOMS_DISTRIBUTION}"
echo "BATCH_SIZE=${BATCH_SIZE}, NUM_BATCHES=${NUM_BATCHES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "Logging generation output to: ${RUN_LOG}"

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

generate_command="$(command -v mattergen-generate || true)"
if [[ -z "${generate_command}" ]]; then
    echo "ERROR: mattergen-generate was not found in PATH after activating ${CONDA_ENV}" >&2
    exit 1
fi
echo "Using mattergen-generate: ${generate_command}"

python - <<'PY'
import os, sys
import mattergen
print("DEBUG mattergen.__file__ =", mattergen.__file__)
print("DEBUG PROJECT_ROOT       =", os.environ.get("PROJECT_ROOT"))
print("DEBUG PYTHONPATH         =", os.environ.get("PYTHONPATH"))
print("DEBUG sys.path[:10]      =", sys.path[:10])
PY

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    "${generate_command}" "${RESULTS_PATH}" \
    --model_path="${MODEL_PATH}" \
    --batch_size="${BATCH_SIZE}" \
    --num_batches="${NUM_BATCHES}" \
    --num_atoms_distribution="${NUM_ATOMS_DISTRIBUTION}" \
    "$@"
