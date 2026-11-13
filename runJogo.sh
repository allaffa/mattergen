#!/usr/bin/env bash
set -euo pipefail

# Local, single-GPU analogue of the mattergen-train srun payload in
# job-frontier-mattergen-train-rocm720.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"

# The requested path points at the env's Python executable. Conda activation
# needs the env directory, so derive that while still using the requested Python.
CONDA_PYTHON="${CONDA_PYTHON:-/home/3pf/miniconda3/envs/hydragnn/bin/python}"
CONDA_ENV="${CONDA_ENV:-$(cd -- "$(dirname -- "${CONDA_PYTHON}")/.." && pwd)}"

DATA_MODULE="${DATA_MODULE:-miniOmat24}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINT_SEARCH_ROOT="${CHECKPOINT_SEARCH_ROOT:-${REPO_ROOT}/outputs/singlerun}"
LOOK_FOR_CHECKPOINT="${LOOK_FOR_CHECKPOINT:-true}"
LOOK_FOR_CHECKPOINT=true
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-4,5,6,7}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"

cd "${REPO_ROOT}"
mkdir -p jobOutputs
RUN_LOG="${RUN_LOG:-jobOutputs/runJogo-$(date +%Y%m%d-%H%M%S).out}"
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

export CUDA_DEVICE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "Using Python: $(which python)"
echo "CUDA_DEVICE=${CUDA_DEVICE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Logging run output to: ${RUN_LOG}"
echo "Checkpoint search root: ${CHECKPOINT_SEARCH_ROOT}"
echo "Initial checkpoint lookup: ${LOOK_FOR_CHECKPOINT}"
echo "Restart delay: ${RESTART_DELAY_SECONDS}s; MAX_RESTARTS=${MAX_RESTARTS} (0 means unlimited)"

python - <<'PY'
import os, sys
import mattergen
print("DEBUG mattergen.__file__ =", mattergen.__file__)
print("DEBUG PROJECT_ROOT       =", os.environ.get("PROJECT_ROOT"))
print("DEBUG PYTHONPATH         =", os.environ.get("PYTHONPATH"))
print("DEBUG sys.path[:10]      =", sys.path[:10])
PY

find_latest_checkpoint() {
    [[ -d "${CHECKPOINT_SEARCH_ROOT}" ]] || return 1
    find "${CHECKPOINT_SEARCH_ROOT}" -path '*/checkpoints/last.ckpt' -type f \
        -printf '%T@ %p\n' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "Starting training attempt ${attempt} at $(date --iso-8601=seconds)"

    train_args=(
        "data_module=${DATA_MODULE}"
        "~trainer.logger"
        "trainer.accelerator=gpu"
        "trainer.devices=4"
        "trainer.num_nodes=1"
        "native_trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}"
    )

    if [[ "${LOOK_FOR_CHECKPOINT,,}" == "true" ]]; then
        latest_checkpoint="$(find_latest_checkpoint || true)"
        if [[ -n "${latest_checkpoint}" ]]; then
            echo "Resuming from latest checkpoint: ${latest_checkpoint}"
            train_args+=("++checkpoint_path=${latest_checkpoint}")
        elif [[ -n "${CHECKPOINT_PATH}" ]]; then
            echo "No checkpoint found under ${CHECKPOINT_SEARCH_ROOT}; using CHECKPOINT_PATH=${CHECKPOINT_PATH}"
            train_args+=("++checkpoint_path=${CHECKPOINT_PATH}")
        else
            echo "No checkpoint found under ${CHECKPOINT_SEARCH_ROOT}; starting from scratch"
        fi
    else
        echo "Checkpoint lookup disabled for this attempt; starting from scratch"
    fi

    set +e
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=4 \
        -m mattergen.scripts.run \
        "${train_args[@]}" \
        "$@"
    rc=$?
    set -e
    LOOK_FOR_CHECKPOINT=true

    if (( rc == 0 )); then
        echo "Training completed successfully at $(date --iso-8601=seconds)"
        exit 0
    fi

    echo "Training attempt ${attempt} failed with exit code ${rc} at $(date --iso-8601=seconds)"
    if (( MAX_RESTARTS > 0 && attempt >= MAX_RESTARTS )); then
        echo "Reached MAX_RESTARTS=${MAX_RESTARTS}; exiting with code ${rc}"
        exit "${rc}"
    fi
    echo "Restarting in ${RESTART_DELAY_SECONDS}s..."
    sleep "${RESTART_DELAY_SECONDS}"
done
