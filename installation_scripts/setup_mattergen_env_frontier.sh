#!/usr/bin/env bash
set -Eeuo pipefail

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: ./installation_scripts/setup_mattergen_env_frontier.sh [options]

Options:
  --env-path <path>         Conda environment path (default: ./MatterGen-Installation-Frontier/mattergen_venv)
  --python-version <ver>    Python version for environment (default: 3.11)
  --recreate                Remove and recreate environment if it already exists
  --skip-modules            Do not load Frontier module stack
  --skip-rocm-torch         Do not force ROCm torch install
  --pyg-from-source         Build torch-scatter/sparse/cluster/spline-conv from source (default)
  --pyg-from-wheels         Install torch-scatter/sparse/cluster/spline-conv from wheels
  -h, --help                Show this message

Environment overrides:
  INSTALL_ROOT
  ENV_PATH
  PYTHON_VERSION
  LOAD_FRONTIER_MODULES (1/0)
  INSTALL_ROCM_TORCH (1/0)
  BUILD_PYG_FROM_SOURCE (1/0)
EOF
}

INSTALL_ROOT="${INSTALL_ROOT:-$PWD/MatterGen-Installation-Frontier}"
ENV_PATH="${ENV_PATH:-$INSTALL_ROOT/mattergen_venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RECREATE_ENV=0
LOAD_FRONTIER_MODULES="${LOAD_FRONTIER_MODULES:-1}"
INSTALL_ROCM_TORCH="${INSTALL_ROCM_TORCH:-1}"
BUILD_PYG_FROM_SOURCE="${BUILD_PYG_FROM_SOURCE:-1}"
ROCM_MM="${ROCM_MM:-7.2}"
PYG_SCATTER_SHA="n/a"
PYG_SPARSE_SHA="n/a"
PYG_CLUSTER_SHA="n/a"
PYG_SPLINE_SHA="n/a"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-path)
      ENV_PATH="$2"; shift 2 ;;
    --python-version)
      PYTHON_VERSION="$2"; shift 2 ;;
    --recreate)
      RECREATE_ENV=1; shift ;;
    --skip-modules)
      LOAD_FRONTIER_MODULES=0; shift ;;
    --skip-rocm-torch)
      INSTALL_ROCM_TORCH=0; shift ;;
    --pyg-from-source)
      BUILD_PYG_FROM_SOURCE=1; shift ;;
    --pyg-from-wheels)
      BUILD_PYG_FROM_SOURCE=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "Unknown option: $1" ;;
  esac
done

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT" || die "Cannot cd to repository root: $REPO_ROOT"
[[ -f pyproject.toml ]] || die "Run this script from MatterGen repository root (pyproject.toml not found)."

safe_ml() {
  local module_name="$1"
  if ! ml "$module_name" >/dev/null 2>&1; then
    echo "WARN: failed to load module '$module_name'"
  fi
}

banner "MatterGen environment setup started ($(date))"

if [[ "$LOAD_FRONTIER_MODULES" == "1" ]]; then
  banner "Configure module stack"
  if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
  fi

  if command -v module >/dev/null 2>&1; then
    module reset || true
    safe_ml cpe/24.07
    safe_ml cce/18.0.0
    safe_ml rocm/7.2.0
    safe_ml amd-mixed/7.2.0
    safe_ml craype-accel-amd-gfx90a
    safe_ml PrgEnv-gnu
    safe_ml miniforge3/23.11.0-0
    safe_ml git-lfs
    module unload darshan-runtime || true
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
  else
    echo "WARN: module command not available; continuing without module loads"
  fi
fi

banner "Create/activate conda environment"
mkdir -p "$INSTALL_ROOT"

command -v conda >/dev/null 2>&1 || die "conda not found. Load miniforge/anaconda first."
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
echo "Python: $(which python)"
python --version

banner "Install Python build tooling"
python -m pip install -U pip setuptools wheel cmake ninja

if [[ "$INSTALL_ROCM_TORCH" == "1" ]]; then
  banner "Install ROCm PyTorch"
  ROCM_MM=""
  if command -v module >/dev/null 2>&1; then
    ROCM_MM="$(module -t list 2>&1 | grep -Eo 'rocm/[0-9]+\.[0-9]+' | head -n1 | cut -d/ -f2 || true)"
  fi
  if [[ -z "$ROCM_MM" && -n "${ROCM_VERSION:-}" ]]; then
    ROCM_MM="$(echo "$ROCM_VERSION" | awk -F. '{print $1"."$2}')"
  fi
  [[ -n "$ROCM_MM" ]] || ROCM_MM="7.2"

  PYTORCH_ROCM_INDEX_URL="https://download.pytorch.org/whl/rocm${ROCM_MM}"
  subbanner "Using index: ${PYTORCH_ROCM_INDEX_URL}"
  python -m pip install --index-url "$PYTORCH_ROCM_INDEX_URL" torch torchvision torchaudio
fi

banner "Install MatterGen dependencies from pyproject"
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
    "torch-scatter", "torch_sparse", "torch-sparse",
    "torch-cluster", "torch_cluster", "torch-geometric", "torch_geometric",
}

normalized = []
for dep in deps:
    base = dep.split(";")[0].split("[")[0]
    for op in ("<=", ">=", "==", "!=", "~=", ">", "<"):
        if op in base:
            base = base.split(op)[0]
            break
    name = base.strip().replace("_", "-").lower()
    if name in skip:
        continue
    normalized.append(dep)

if normalized:
    subprocess.check_call(["python", "-m", "pip", "install", *normalized])
PY

banner "Install PyG stack"
if [[ "$BUILD_PYG_FROM_SOURCE" == "1" ]]; then
  PYG_FRONTIER="${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}"
  mkdir -p "$PYG_FRONTIER"
  cd "$PYG_FRONTIER"

  subbanner "pytorch_scatter (ROCm fork pinned to 9799c51)"
  if [[ ! -d pytorch_scatter/.git ]]; then
    git clone --recursive https://github.com/Looong01/pytorch_scatter-rocm.git pytorch_scatter
  fi
  pushd pytorch_scatter >/dev/null
  git fetch --all
  git checkout 9799c51
  git submodule update --init --recursive
  rm -rf build
  CC=gcc CXX=g++ python setup.py build
  CC=gcc CXX=g++ python setup.py install
  PYG_SCATTER_SHA="$(git rev-parse HEAD)"
  popd >/dev/null

  subbanner "pytorch_sparse (ROCm fork pinned to 2340737)"
  if [[ ! -d pytorch_sparse/.git ]]; then
    git clone --recursive https://github.com/Looong01/pytorch_sparse-rocm.git pytorch_sparse
  fi
  pushd pytorch_sparse >/dev/null
  git fetch --all
  git checkout 2340737
  git submodule update --init --recursive
  rm -rf build
  CC=gcc CXX=g++ python setup.py build
  CC=gcc CXX=g++ python setup.py install
  PYG_SPARSE_SHA="$(git rev-parse HEAD)"
  popd >/dev/null

  subbanner "pytorch_cluster (official pinned to 1.6.3-11-g4126a52)"
  if [[ ! -d pytorch_cluster/.git ]]; then
    git clone --recursive https://github.com/rusty1s/pytorch_cluster.git
  fi
  pushd pytorch_cluster >/dev/null
  git fetch --all
  git checkout 1.6.3-11-g4126a52
  git submodule update --init --recursive
  rm -rf build
  CC=gcc CXX=g++ python setup.py build
  CC=gcc CXX=g++ python setup.py install
  PYG_CLUSTER_SHA="$(git rev-parse HEAD)"
  popd >/dev/null

  subbanner "pytorch_spline_conv (official pinned to 1.2.2-9-ga6d1020)"
  if [[ ! -d pytorch_spline_conv/.git ]]; then
    git clone --recursive https://github.com/rusty1s/pytorch_spline_conv.git
  fi
  pushd pytorch_spline_conv >/dev/null
  git fetch --all
  git checkout 1.2.2-9-ga6d1020
  git submodule update --init --recursive
  rm -rf build
  CC=gcc CXX=g++ python setup.py build
  CC=gcc CXX=g++ python setup.py install
  PYG_SPLINE_SHA="$(git rev-parse HEAD)"
  popd >/dev/null

  cd "$REPO_ROOT"
  python -m pip install torch-geometric
else
  subbanner "Installing wheel-based PyG stack"
  python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
  PYG_SCATTER_SHA="wheel"
  PYG_SPARSE_SHA="wheel"
  PYG_CLUSTER_SHA="wheel"
  PYG_SPLINE_SHA="wheel"
fi

banner "Install MatterGen package"
python -m pip install -e . --no-deps

banner "Post-install verification"
python - <<'PY'
import importlib
mods = [
    "hydra",
    "omegaconf",
    "torch",
    "torch_geometric",
    "torch_spline_conv",
    "mattergen",
    "mattergen.scripts.run",
    "mattergen.scripts.csv_to_dataset",
]
for m in mods:
    importlib.import_module(m)
print("Python imports OK")
PY

if ! command -v mattergen-train >/dev/null 2>&1; then
  die "mattergen-train is not on PATH after install"
fi
if ! command -v csv-to-dataset >/dev/null 2>&1; then
  die "csv-to-dataset is not on PATH after install"
fi

if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local >/dev/null 2>&1 || true
fi

banner "Setup complete"
cat <<EOF
Final Summary
Base install:        $INSTALL_ROOT
Virtual environment: $ENV_PATH
PyTorch-Geometric:   ${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}
  - pytorch_scatter:     $PYG_SCATTER_SHA
  - pytorch_sparse:      $PYG_SPARSE_SHA
  - pytorch_cluster:     $PYG_CLUSTER_SHA
  - pytorch_spline_conv: $PYG_SPLINE_SHA
EOF

echo "Environment path: $ENV_PATH"
echo "Activate with: conda activate $ENV_PATH"
echo "Smoke test: mattergen-train --help"
