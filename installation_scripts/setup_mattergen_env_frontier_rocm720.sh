#!/usr/bin/env bash
# =============================================================================
# MatterGen environment setup for Frontier
# =============================================================================
# Creates a Conda-based environment for MatterGen on Frontier, with optional
# ROCm PyTorch installation and PyTorch-Geometric built either from source or
# from wheels.
#
# This script also installs several supporting packages/tools used in your
# workflow:
#   - pymatgen (from GitHub)
#   - mpi4py 3.1.5
#   - ADIOS2 v2.10.2
#   - DDStore
#   - DeepHyper (develop)
#   - GPTL + gptl4py
#   - HydraGNN
#
# Usage:
#   ./installation_scripts/setup_mattergen_env_frontier.sh [options]
#
# Options:
#   --env-path <path>         Conda env path
#                             (default: ./MatterGen-Installation-Frontier/mattergen_venv)
#   --python-version <ver>    Python version (default: 3.11)
#   --recreate                Remove and recreate the environment if it exists
#   --skip-modules            Do not load Frontier module stack
#   --skip-rocm-torch         Do not install ROCm PyTorch
#   --pyg-from-source         Build torch-scatter/sparse/cluster/spline-conv from source
#                             (default)
#   --pyg-from-wheels         Install torch-scatter/sparse/cluster/spline-conv from wheels
#   -h, --help                Show this message
#
# Environment overrides:
#   INSTALL_ROOT
#   ENV_PATH
#   PYTHON_VERSION
#   LOAD_FRONTIER_MODULES
#   INSTALL_ROCM_TORCH
#   BUILD_PYG_FROM_SOURCE
#   ROCM_MM
# =============================================================================
set -Eeuo pipefail

hr()        { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner()    { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }
die()       { echo "ERROR: $*" >&2; exit 1; }

usage() {
  sed -n '2,36p' "$0"
}

pip_retry() {
  local n=0
  until python -m pip install "$@"; do
    n=$((n + 1))
    [[ $n -ge 3 ]] && die "pip install failed after 3 attempts: $*"
    echo "pip install failed (attempt $n), retrying in 10s..."
    sleep 10
  done
}

safe_ml() {
  local module_name="$1"
  if ! ml "$module_name" >/dev/null 2>&1; then
    echo "WARN: failed to load module '$module_name'"
  fi
}

assert_numpy_1264() {
  python - <<'PY'
import sys
try:
    import numpy as np
    print("numpy version:", np.__version__)
except Exception as e:
    sys.exit(f"numpy import failed: {e!r}")
PY
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
INSTALL_ROOT="${INSTALL_ROOT:-$REPO_ROOT/MatterGen-Installation-Frontier}"
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
    --env-path)         ENV_PATH="$2"; shift 2 ;;
    --python-version)   PYTHON_VERSION="$2"; shift 2 ;;
    --recreate)         RECREATE_ENV=1; shift ;;
    --skip-modules)     LOAD_FRONTIER_MODULES=0; shift ;;
    --skip-rocm-torch)  INSTALL_ROCM_TORCH=0; shift ;;
    --pyg-from-source)  BUILD_PYG_FROM_SOURCE=1; shift ;;
    --pyg-from-wheels)  BUILD_PYG_FROM_SOURCE=0; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  die "Unknown option: $1" ;;
  esac
done

cd "$REPO_ROOT" || die "Cannot cd to repository root: $REPO_ROOT"
[[ -f pyproject.toml ]] || die "Run from the MatterGen repo root (pyproject.toml not found)."

banner "MatterGen environment setup started ($(date))"

# Clear any inherited conda environment from the launching shell
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
      CONDA_PROMPT_MODIFIER 2>/dev/null || true

# -----------------------------------------------------------------------------
banner "Configure module stack"
# -----------------------------------------------------------------------------
if [[ "$LOAD_FRONTIER_MODULES" == "1" ]]; then
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

# -----------------------------------------------------------------------------
banner "Create/activate conda environment"
# -----------------------------------------------------------------------------
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
PYTHON_BIN="$(which python)"
echo "Python: $PYTHON_BIN ($(python --version))"

banner "Install Python build tooling"
pip_retry -U pip setuptools wheel cmake ninja packaging

# -----------------------------------------------------------------------------
banner "Install ROCm PyTorch"
# -----------------------------------------------------------------------------
if [[ "$INSTALL_ROCM_TORCH" == "1" ]]; then
  DETECTED_ROCM_MM=""
  if command -v module >/dev/null 2>&1; then
    DETECTED_ROCM_MM="$(module -t list 2>&1 | grep -Eo 'rocm/[0-9]+\.[0-9]+' | head -n1 | cut -d/ -f2 || true)"
  fi
  if [[ -z "$DETECTED_ROCM_MM" && -n "${ROCM_VERSION:-}" ]]; then
    DETECTED_ROCM_MM="$(echo "$ROCM_VERSION" | awk -F. '{print $1"."$2}')"
  fi
  [[ -n "$DETECTED_ROCM_MM" ]] && ROCM_MM="$DETECTED_ROCM_MM"

  PYTORCH_ROCM_INDEX_URL="https://download.pytorch.org/whl/rocm${ROCM_MM}"
  subbanner "Using index: ${PYTORCH_ROCM_INDEX_URL}"
  pip_retry --index-url "$PYTORCH_ROCM_INDEX_URL" torch torchvision torchaudio

  python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
PY
else
  subbanner "Skipping ROCm PyTorch install"
fi

# -----------------------------------------------------------------------------
banner "Install MatterGen dependencies from pyproject (excluding torch/PyG)"
# -----------------------------------------------------------------------------
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
            base = base.split(op)[0]
            break
    if base.strip().replace("_", "-").lower() in skip:
        continue
    out.append(dep)
if out:
    subprocess.check_call(["python", "-m", "pip", "install", *out])
PY

# -----------------------------------------------------------------------------
banner "Install PyTorch-Geometric stack"
# -----------------------------------------------------------------------------
if [[ "$BUILD_PYG_FROM_SOURCE" == "1" ]]; then
  PYG_FRONTIER="${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}"
  mkdir -p "$PYG_FRONTIER"
  cd "$PYG_FRONTIER"

  build_pyg() {  # name repo_url ref
    local name="$1" url="$2" ref="$3"
    if [[ ! -d "$name/.git" ]]; then
      git clone --recursive "$url" "$name"
    fi
    pushd "$name" >/dev/null
    git fetch --all
    git checkout "$ref"
    git submodule update --init --recursive
    rm -rf build
    CC=gcc CXX=g++ python setup.py build
    CC=gcc CXX=g++ python setup.py install
    git rev-parse HEAD
    popd >/dev/null
  }

  subbanner "Building torch-scatter from ROCm fork"
  PYG_SCATTER_SHA="$(build_pyg pytorch_scatter https://github.com/Looong01/pytorch_scatter-rocm.git 9799c51 | tail -n1)"

  subbanner "Building torch-sparse from ROCm fork"
  PYG_SPARSE_SHA="$(build_pyg pytorch_sparse https://github.com/Looong01/pytorch_sparse-rocm.git 2340737 | tail -n1)"

  subbanner "Building torch-cluster from official repo"
  PYG_CLUSTER_SHA="$(build_pyg pytorch_cluster https://github.com/rusty1s/pytorch_cluster.git 1.6.3-11-g4126a52 | tail -n1)"

  subbanner "Building torch-spline-conv from official repo"
  PYG_SPLINE_SHA="$(build_pyg pytorch_spline_conv https://github.com/rusty1s/pytorch_spline_conv.git 1.2.2-9-ga6d1020 | tail -n1)"

  cd "$REPO_ROOT"
  pip_retry torch-geometric
else
  subbanner "Installing wheel-based PyG stack"
  pip_retry torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
  PYG_SCATTER_SHA="wheel"
  PYG_SPARSE_SHA="wheel"
  PYG_CLUSTER_SHA="wheel"
  PYG_SPLINE_SHA="wheel"
fi

banner "Install MatterGen package (editable, no extra deps)"
pip_retry -e "$REPO_ROOT" --no-deps

# =============================================================================
# pymatgen reinstall from GitHub + verification
# =============================================================================
banner "pymatgen reinstall from GitHub"

subbanner "Remove existing pymatgen"
python -m pip uninstall -y pymatgen || true

subbanner "Remove any leftover pymatgen files from site-packages"
python - <<'PY'
import site
import sys
from pathlib import Path
import shutil

paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass

user_site = site.getusersitepackages()
if user_site:
    paths.append(user_site)

seen = set()
for p in paths:
    if not p or p in seen:
        continue
    seen.add(p)
    sp = Path(p)
    if not sp.exists():
        continue
    for target in sp.glob("pymatgen*"):
        print(f"Removing leftover: {target}")
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
PY

subbanner "Install pymatgen from GitHub"
python -m pip install --no-deps --no-cache-dir "git+https://github.com/materialsproject/pymatgen.git"

subbanner "Verify pymatgen graph modules"
python - <<'PY'
import importlib.util
import sys

mods = [
    "pymatgen",
    "pymatgen.analysis.graphs",
    "pymatgen.core.graphs",
]

failed = False
for name in mods:
    try:
        spec = importlib.util.find_spec(name)
        print(f"{name} -> {spec}")
        if spec is not None:
            print(f"  origin: {spec.origin}")
        else:
            failed = True
    except Exception as e:
        print(f"{name} -> ERROR: {e!r}")
        failed = True

print("\nImport test:")
try:
    from pymatgen.core.graphs import StructureGraph
    print("pymatgen.core.graphs.StructureGraph -> OK", StructureGraph)
except Exception as e:
    print("core.graphs import failed:", repr(e))
    failed = True

try:
    from pymatgen.analysis.graphs import StructureGraph as SG2
    print("pymatgen.analysis.graphs.StructureGraph -> OK", SG2)
except Exception as e:
    print("analysis.graphs import failed:", repr(e))
    failed = True

if failed:
    sys.exit("pymatgen verification failed")
PY

# =============================================================================
# mpi4py 3.1.5
# =============================================================================
banner "mpi4py 3.1.5"
MPI4PY_FRONTIER="${INSTALL_ROOT}/MPI4PY-Frontier"
export MPI4PY_FRONTIER
mkdir -p "$MPI4PY_FRONTIER"
cd "$MPI4PY_FRONTIER"

if [[ ! -d mpi4py/.git ]]; then
  git clone -b 3.1.5 https://github.com/mpi4py/mpi4py.git
fi

pushd mpi4py >/dev/null
rm -rf build
CC=cc MPICC=cc "$PYTHON_BIN" -m pip install . --no-build-isolation -v
popd >/dev/null

# =============================================================================
# ADIOS2
# =============================================================================
banner "ADIOS2 (v2.10.2)"
ADIOS2_FRONTIER="${INSTALL_ROOT}/ADIOS2-Frontier"
export ADIOS2_FRONTIER
mkdir -p "$ADIOS2_FRONTIER"
cd "$ADIOS2_FRONTIER"

if [[ ! -d ADIOS2/.git ]]; then
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git
fi

mkdir -p adios2-build

CC=cc CXX=CC FC=ftn \
cmake -DCMAKE_INSTALL_PREFIX="$ENV_PATH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DADIOS2_USE_MPI=ON \
      -DADIOS2_USE_Fortran=OFF \
      -DADIOS2_BUILD_EXAMPLES_EXPERIMENTAL=OFF \
      -DADIOS2_BUILD_TESTING=OFF \
      -DADIOS2_USE_HDF5=OFF \
      -DADIOS2_USE_SST=OFF \
      -DADIOS2_USE_BZip2=OFF \
      -DADIOS2_USE_PNG=OFF \
      -DADIOS2_USE_DataSpaces=OFF \
      -DADIOS2_USE_Python=ON \
      -DPython_EXECUTABLE="$(which python)" \
      -B adios2-build -S ADIOS2

cmake --build adios2-build -j32
cmake --install adios2_build 2>/dev/null || cmake --install adios2-build

# =============================================================================
# DDStore
# =============================================================================
banner "DDStore"
DDSTORE_FRONTIER="${INSTALL_ROOT}/DDStore-Frontier"
export DDSTORE_FRONTIER
mkdir -p "$DDSTORE_FRONTIER"
cd "$DDSTORE_FRONTIER"

git clone git@github.com:ORNL/DDStore.git || true
pushd DDStore >/dev/null
CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# =============================================================================
# DeepHyper
# =============================================================================
banner "DeepHyper (develop branch)"
DEEPHYPER_FRONTIER="${INSTALL_ROOT}/DeepHyperFrontier"
export DEEPHYPER_FRONTIER
mkdir -p "$DEEPHYPER_FRONTIER"
cd "$DEEPHYPER_FRONTIER"

git clone https://github.com/deephyper/deephyper.git || true
cd deephyper
git fetch origin develop
git checkout develop
pip_retry -e ".[hps,hps-tl]" --verbose
assert_numpy_1264

# =============================================================================
# GPTL
# =============================================================================
banner "GPTL"
GPTL_FRONTIER="${INSTALL_ROOT}/GPTLFrontier"
export GPTL_FRONTIER
mkdir -p "$GPTL_FRONTIER"
cd "$GPTL_FRONTIER"

wget -nc https://github.com/jmrosinski/GPTL/releases/download/v8.1.1/gptl-8.1.1.tar.gz
tar xvf gptl-8.1.1.tar.gz || true

pushd gptl-8.1.1 >/dev/null
./configure --prefix="$INSTALL_ROOT" --disable-libunwind CC=cc CXX=CC FC=ftn
make install
popd >/dev/null

git clone git@github.com:jychoi-hpc/gptl4py.git || true
pushd gptl4py >/dev/null
GPTL_DIR="$INSTALL_ROOT" CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# =============================================================================
# HydraGNN
# =============================================================================
banner "HydraGNN"
HYDRAGNN_FRONTIER="${INSTALL_ROOT}/HydraGNN"
export HYDRAGNN_FRONTIER
mkdir -p "$HYDRAGNN_FRONTIER"
cd "$HYDRAGNN_FRONTIER"

git clone https://github.com/ORNL/HydraGNN.git || true
cd HydraGNN

"$ENV_PATH/bin/python" -m pip uninstall -y hydragnn || true
"$ENV_PATH/bin/python" -m pip install -e . --verbose
assert_numpy_1264

cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
banner "Post-install verification"
# -----------------------------------------------------------------------------
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
failed = False
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  import OK: {m}")
    except Exception as e:
        print(f"  import FAIL: {m} -> {e}")
        failed = True
if failed:
    raise SystemExit("Post-install import verification failed")

import torch
print("GPU visible:", torch.cuda.is_available())
PY

command -v mattergen-train >/dev/null 2>&1 || die "mattergen-train not on PATH after install"
command -v csv-to-dataset  >/dev/null 2>&1 || die "csv-to-dataset not on PATH after install"

if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local >/dev/null 2>&1 || true
fi

banner "Setup complete"
cat <<EOF
Environment path:  $ENV_PATH
Activate with:     conda activate $ENV_PATH
Smoke test:        mattergen-train --help

Final Summary
Base install:        $INSTALL_ROOT
Virtual environment: $ENV_PATH
PyTorch-Geometric:   ${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}
  - pytorch_scatter:     $PYG_SCATTER_SHA
  - pytorch_sparse:      $PYG_SPARSE_SHA
  - pytorch_cluster:     $PYG_CLUSTER_SHA
  - pytorch_spline_conv: $PYG_SPLINE_SHA
EOF
