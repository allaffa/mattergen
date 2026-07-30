#!/usr/bin/env bash
# =============================================================================
# MatterGen environment setup for Frontier — ROCm 7.13.0
# =============================================================================
# Builds a fresh Python 3.12 environment using the AMD-published gfx90a PyTorch
# 2.11 wheels for ROCm 7.13.0 (repo.amd.com). Mirrors the validated lumina-sdk
# ROCm 7.13 approach: the AMD wheel bundles an RCCL build matched to ROCm 7.13,
# which fixes the HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION crash seen with the old
# ROCm 7.1.x / 7.2.x RCCL kernels.
#
# Usage:
#   ./installation_scripts/setup_mattergen_env_frontier_rocm713.sh [options]
#
# Options:
#   --env-path <path>     Conda env path (default: ./MatterGen-Installation-Frontier/mattergen_venv_rocm713)
#   --python-version <v>  Python version (default: 3.12)
#   --recreate            Remove and recreate the environment if it exists
#   --pyg-from-source     Build torch-scatter/sparse/cluster/spline-conv from ROCm forks
#   --pyg-from-wheels     Install the AMD ROCm PyG wheels (default)
#   -h, --help            Show this message
# =============================================================================
set -Eeuo pipefail

hr()        { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner()    { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }
die()       { echo "ERROR: $*" >&2; exit 1; }

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
INSTALL_ROOT="${INSTALL_ROOT:-$REPO_ROOT/MatterGen-Installation-Frontier}"
ENV_PATH="${ENV_PATH:-$INSTALL_ROOT/mattergen_venv_rocm713}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RECREATE_ENV=0
BUILD_PYG_FROM_SOURCE="${BUILD_PYG_FROM_SOURCE:-0}"
EXPECTED_ROCM_MM="7.13"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-path)        ENV_PATH="$2"; shift 2 ;;
    --python-version)  PYTHON_VERSION="$2"; shift 2 ;;
    --recreate)        RECREATE_ENV=1; shift ;;
    --pyg-from-source) BUILD_PYG_FROM_SOURCE=1; shift ;;
    --pyg-from-wheels) BUILD_PYG_FROM_SOURCE=0; shift ;;
    -h|--help)         sed -n '2,25p' "$0"; exit 0 ;;
    *)                 die "Unknown option: $1" ;;
  esac
done

cd "$REPO_ROOT" || die "Cannot cd to repository root: $REPO_ROOT"
[[ -f pyproject.toml ]] || die "Run from the MatterGen repo root (pyproject.toml not found)."

SCRIPT_DIR="$REPO_ROOT/installation_scripts"

pip_retry() {
  local n=0
  until python -m pip install "$@"; do
    n=$((n+1)); [[ $n -ge 3 ]] && die "pip install failed after 3 attempts: $*"
    echo "pip install failed (attempt $n), retrying in 10s..."; sleep 10
  done
}

banner "MatterGen ROCm 7.13 environment setup started ($(date))"

# ----------------------------------------------------------------------------
banner "Configure module stack (ROCm 7.13.0)"
# ----------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
  [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/module-to-load-frontier-rocm713.sh"

# ----------------------------------------------------------------------------
banner "Verify ROCm version"
# ----------------------------------------------------------------------------
detect_rocm_mm() {
  local v=""
  v="$(module -t list 2>&1 | grep -Eo 'rocm/[0-9]+\.[0-9]+' | head -n1 | cut -d/ -f2 || true)"
  if [[ -z "$v" && -n "${ROCM_PATH:-}" ]]; then
    v="$(echo "$ROCM_PATH" | grep -Eo '[0-9]+\.[0-9]+' | head -n1 || true)"
  fi
  echo "$v"
}
ROCM_MM="${ROCM_MM:-$(detect_rocm_mm)}"
echo "Detected ROCm: ${ROCM_MM:-<unknown>}"
[[ "$ROCM_MM" == "$EXPECTED_ROCM_MM" ]] || die "ROCm mismatch: detected '$ROCM_MM', expected '$EXPECTED_ROCM_MM'."

# ----------------------------------------------------------------------------
banner "Create/activate conda environment"
# ----------------------------------------------------------------------------
mkdir -p "$INSTALL_ROOT"
command -v conda >/dev/null 2>&1 || die "conda not found (load miniforge3 module)."
eval "$(conda shell.bash hook)"

if [[ -d "$ENV_PATH" && "$RECREATE_ENV" == "1" ]]; then
  subbanner "Removing existing environment: $ENV_PATH"
  conda env remove -p "$ENV_PATH" -y >/dev/null 2>&1 || rm -rf "$ENV_PATH"
fi
if [[ ! -d "$ENV_PATH" ]]; then
  subbanner "Creating environment: $ENV_PATH (python=$PYTHON_VERSION)"
  conda create -y -p "$ENV_PATH" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_PATH"
echo "Python: $(which python) ($(python --version))"

banner "Install Python build tooling"
pip_retry -U pip setuptools wheel cmake ninja packaging

# ----------------------------------------------------------------------------
banner "Install AMD gfx90a PyTorch 2.11 wheels for ROCm 7.13.0"
# ----------------------------------------------------------------------------
AMD_ROCM_INDEX_URL="https://repo.amd.com/rocm/whl/gfx90a/"
subbanner "Using index: ${AMD_ROCM_INDEX_URL}"
pip_retry --index-url "${AMD_ROCM_INDEX_URL}" \
  "torch==2.11.0+rocm7.13.0" \
  "torchvision==0.26.0+rocm7.13.0" \
  "torchaudio==2.11.0+rocm7.13.0"

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
print("arch list =", torch.cuda.get_arch_list())
PY

# ----------------------------------------------------------------------------
banner "Install PyTorch-Geometric stack"
# ----------------------------------------------------------------------------
if [[ "$BUILD_PYG_FROM_SOURCE" == "1" ]]; then
  subbanner "Building torch-scatter/sparse/cluster/spline-conv from ROCm forks"
  PYG_FRONTIER="${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}"
  mkdir -p "$PYG_FRONTIER"; cd "$PYG_FRONTIER"

  build_pyg() {  # name repo_url ref
    local name="$1" url="$2" ref="$3"
    if [[ ! -d "$name/.git" ]]; then git clone --recursive "$url" "$name"; fi
    pushd "$name" >/dev/null
    git fetch --all; git checkout "$ref"; git submodule update --init --recursive
    rm -rf build
    CC=gcc CXX=g++ python setup.py build
    CC=gcc CXX=g++ python setup.py install
    popd >/dev/null
  }
  build_pyg pytorch_scatter      https://github.com/Looong01/pytorch_scatter-rocm.git 9799c51
  build_pyg pytorch_sparse       https://github.com/Looong01/pytorch_sparse-rocm.git  2340737
  build_pyg pytorch_cluster      https://github.com/rusty1s/pytorch_cluster.git        1.6.3-11-g4126a52
  build_pyg pytorch_spline_conv  https://github.com/rusty1s/pytorch_spline_conv.git    1.2.2-9-ga6d1020
  cd "$REPO_ROOT"
  pip_retry torch-geometric
else
  subbanner "Installing AMD ROCm PyG wheels (gfx90a)"
  pip_retry scipy
  pip_retry torch-geometric torch-scatter-rocm torch-sparse-rocm \
    torch-cluster-rocm torch-spline-conv-rocm pyg-lib-rocm
fi

# ----------------------------------------------------------------------------
banner "Install MatterGen dependencies from pyproject (excluding torch/PyG)"
# ----------------------------------------------------------------------------
python - <<'PY'
import subprocess
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

deps = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
skip = {
    "torch", "torchvision", "torchaudio",
    "torch-scatter", "torch-sparse", "torch-cluster",
    "torch-spline-conv", "torch-geometric", "pyg-lib",
}
out = []
for dep in deps:
    base = dep.split(";")[0].split("[")[0]
    for op in ("<=", ">=", "==", "!=", "~=", ">", "<"):
        if op in base:
            base = base.split(op)[0]; break
    if base.strip().replace("_", "-").lower() in skip:
        continue
    out.append(dep)
if out:
    subprocess.check_call(["python", "-m", "pip", "install", *out])
PY

banner "Install MatterGen package (editable, no extra deps)"
pip_retry -e "$REPO_ROOT" --no-deps

# ----------------------------------------------------------------------------
banner "Post-install verification"
# ----------------------------------------------------------------------------
python - <<'PY'
import importlib
for m in ["hydra", "omegaconf", "torch", "torch_geometric",
          "torch_scatter", "torch_sparse", "mattergen",
          "mattergen.scripts.run"]:
    try:
        importlib.import_module(m)
        print(f"  import OK: {m}")
    except Exception as e:
        print(f"  import FAIL: {m} -> {e}")
import torch
print("GPU visible:", torch.cuda.is_available())
PY

command -v mattergen-train >/dev/null 2>&1 || die "mattergen-train not on PATH after install"
command -v csv-to-dataset  >/dev/null 2>&1 || die "csv-to-dataset not on PATH after install"

banner "Setup complete"
cat <<EOF
Environment path:  $ENV_PATH
Activate with:     conda activate $ENV_PATH
Smoke test:        mattergen-train --help
EOF
