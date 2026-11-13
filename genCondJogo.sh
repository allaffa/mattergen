#!/usr/bin/env bash
set -euo pipefail

# Conditional generation sweep for the locally trained MatterGen model.
# Generates one force_rms-conditioned batch per target force and application target.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"

# The requested path points at the env's Python executable. Conda activation
# needs the env directory, so derive that while still using the requested Python.
CONDA_PYTHON="${CONDA_PYTHON:-/home/3pf/miniconda3/envs/hydragnn/bin/python}"
CONDA_ENV="${CONDA_ENV:-$(cd -- "$(dirname -- "${CONDA_PYTHON}")/.." && pwd)}"

MODEL_PATH="${MODEL_PATH:-/home/3pf/projects/lrn070/mattergen/outputs/singlerun/2026-09-14/10-43-29}"
DATA_MODULE="${DATA_MODULE:-miniOmat24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_BATCHES="${NUM_BATCHES:-1}"
NUM_ATOMS_DISTRIBUTION="${NUM_ATOMS_DISTRIBUTION:-Omat24}"
DEFAULT_CONDITIONING_TARGETS="nuclear_fission_materials|U-O;nuclear_fission_materials|U-N;nuclear_fission_materials|U-Zr;fusion_materials|W-Cr-Y;fusion_materials|Li-Pb;fusion_materials|Li-Ti-O;battery_and_electrochemical_materials|Li-Fe-P-O;battery_and_electrochemical_materials|Li-Ni-Mn-Co-O;battery_and_electrochemical_materials|Li-La-Zr-O;catalysts_and_electrocatalysts|Pt-Ni;catalysts_and_electrocatalysts|Cu-Zn-O;catalysts_and_electrocatalysts|Co-N-C;hydrogen_materials|Mg-H;hydrogen_materials|La-Ni-H;hydrogen_materials|Pd-H;critical_mineral_compounds|Nd-Fe-B;critical_mineral_compounds|Ga-N;critical_mineral_compounds|Pt-Pd-Rh;structural_alloys|Fe-Cr-C;structural_alloys|Al-Mg-Si;structural_alloys|Ti-Al-V;materials_for_extreme_environments|Zr-B-C;materials_for_extreme_environments|Ta-C;materials_for_extreme_environments|Al-Cr-O;carbon_management_materials|Mg-O-C;carbon_management_materials|Ca-O-C;carbon_management_materials|Si-Al-O;solar_and_photochemical_materials|Cs-Pb-I-Br;solar_and_photochemical_materials|Cu-In-Ga-Se;solar_and_photochemical_materials|Ti-O;geological_and_subsurface_materials|Na-Al-Si-O-H;geological_and_subsurface_materials|Ca-C-O;geological_and_subsurface_materials|Fe-S;manufacturing_materials|Al-Si-Mg;manufacturing_materials|Ni-Cr-Al;manufacturing_materials|Zr-Si-O;semiconductor_and_quantum_materials|Si-Ge;semiconductor_and_quantum_materials|Ga-N;semiconductor_and_quantum_materials|Mo-S"
CHEMICAL_SYSTEMS="${CHEMICAL_SYSTEMS:-${CHEMICAL_SYSTEM:-}}"
CONDITIONING_TARGETS="${CONDITIONING_TARGETS:-}"
if [[ -z "${CONDITIONING_TARGETS}" && -z "${CHEMICAL_SYSTEMS}" ]]; then
    CONDITIONING_TARGETS="${DEFAULT_CONDITIONING_TARGETS}"
fi
FORCE_RMS_VALUES="${FORCE_RMS_VALUES:-0.01348962883 0.02169280885 0.03488442581 0.05609799875 0.09021176043 0.1450704464 0.2332892553 0.3751548163 0.6032902628 0.9701571869 1.560119606 2.508844153 4.034497714 6.487916669 10.43328456 16.77787066 26.98066388 43.38787907 69.77248814 112.2018454}"
DIFFUSION_GUIDANCE_FACTOR="${DIFFUSION_GUIDANCE_FACTOR:-1.0}"
CUDA_DEVICE="${CUDA_DEVICE:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RESULTS_RUN_TAG="${RESULTS_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

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

root_basename="$(basename -- "${root_dir}")"
results_folder="${root_basename%.bp}"
if [[ "${results_folder}" == "${root_basename}" ]]; then
    echo "ERROR: root_dir basename does not end in .bp: ${root_basename}" >&2
    exit 1
fi

slugify() {
    local value="$1"
    value="${value//[^A-Za-z0-9._-]/_}"
    printf "%s" "${value}"
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf "%s" "${value}"
}

find_existing_extxyz() {
    local condition_label="$1"
    local force_label="$2"

    [[ -d "${RESULTS_ROOT}" ]] || return 1
    find "${RESULTS_ROOT}" \
        -path "*/chemical_system_${condition_label}/*/force_rms_${force_label}/generated_crystals.extxyz" \
        -type f \
        -print \
        -quit
}

conditioning_targets=()
if [[ -n "${CONDITIONING_TARGETS}" ]]; then
    IFS=";" read -r -a conditioning_targets <<< "${CONDITIONING_TARGETS}"
else
    for chemical_system in ${CHEMICAL_SYSTEMS}; do
        conditioning_targets+=("uncategorized|${chemical_system}")
    done
fi

if [[ "${#conditioning_targets[@]}" -eq 0 ]]; then
    echo "ERROR: no conditioning targets were configured." >&2
    exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-results/${results_folder}/conditioned}"

mkdir -p jobOutputs "${RESULTS_ROOT}"
RUN_LOG="${RUN_LOG:-jobOutputs/genCondJogo-${RESULTS_RUN_TAG}.out}"
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
export RESULTS_ROOT
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
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "CONDITIONING_TARGETS=${CONDITIONING_TARGETS:-derived from CHEMICAL_SYSTEMS=${CHEMICAL_SYSTEMS}}"
echo "FORCE_RMS_VALUES=${FORCE_RMS_VALUES}"
echo "DIFFUSION_GUIDANCE_FACTOR=${DIFFUSION_GUIDANCE_FACTOR}"
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

for conditioning_target in "${conditioning_targets[@]}"; do
    if [[ "${conditioning_target}" == *"|"* ]]; then
        application="$(trim "${conditioning_target%%|*}")"
        chemical_system="$(trim "${conditioning_target#*|}")"
    else
        application="uncategorized"
        chemical_system="$(trim "${conditioning_target}")"
    fi

    if [[ -z "${application}" || -z "${chemical_system}" ]]; then
        echo "ERROR: invalid conditioning target: ${conditioning_target}" >&2
        exit 1
    fi

    application_label="$(slugify "${application}")"
    condition_label="$(slugify "${chemical_system}")"
    chemical_results_root="${RESULTS_ROOT}/${application_label}/chemical_system_${condition_label}/${RESULTS_RUN_TAG}"

    for force_rms in ${FORCE_RMS_VALUES}; do
        force_label="${force_rms//./p}"
        force_label="${force_label//[^A-Za-z0-9._-]/_}"
        results_path="${chemical_results_root}/force_rms_${force_label}"
        properties_to_condition_on="{'force_rms':${force_rms},'chemical_system':'${chemical_system}'}"

        existing_extxyz="$(find_existing_extxyz "${condition_label}" "${force_label}" || true)"
        if [[ -n "${existing_extxyz}" ]]; then
            echo
            echo "Skipping conditional generation at $(date --iso-8601=seconds)"
            echo "application=${application}"
            echo "chemical_system=${chemical_system}"
            echo "force_rms=${force_rms}"
            echo "Found existing extxyz: ${existing_extxyz}"
            continue
        fi

        mkdir -p "${results_path}"

        echo
        echo "Starting conditional generation at $(date --iso-8601=seconds)"
        echo "application=${application}"
        echo "chemical_system=${chemical_system}"
        echo "force_rms=${force_rms}"
        echo "properties_to_condition_on=${properties_to_condition_on}"
        echo "RESULTS_PATH=${results_path}"

        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc-per-node="${NPROC_PER_NODE}" \
            "${generate_command}" "${results_path}" \
            --model_path="${MODEL_PATH}" \
            --batch_size="${BATCH_SIZE}" \
            --num_batches="${NUM_BATCHES}" \
            --num_atoms_distribution="${NUM_ATOMS_DISTRIBUTION}" \
            --properties_to_condition_on="${properties_to_condition_on}" \
            --diffusion_guidance_factor="${DIFFUSION_GUIDANCE_FACTOR}" \
            --record_trajectories=False \
            "$@"

        echo "Completed chemical_system=${chemical_system} force_rms=${force_rms} at $(date --iso-8601=seconds)"
    done
done

echo "Conditional sweep completed at $(date --iso-8601=seconds)"
echo "Results root: ${RESULTS_ROOT}"
