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

# Clear any conda environment inherited from the launching shell. Otherwise the
# miniforge module's deactivate hook runs `conda deactivate` in this
# non-interactive shell and fails ("Run 'conda init' before 'conda deactivate'"),
# which aborts the script under `set -e`.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
      CONDA_PROMPT_MODIFIER 2>/dev/null || true

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
  subbanner "Removing prebuilt PyG wheels (ABI-incompatible with torch 2.11+rocm7.13)"
  # The prebuilt gfx90a wheels — especially pyg_lib — are linked against a
  # different libtorch ABI and fail to load with:
  #   libpyg.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
  # torch_sparse.typing does `import pyg_lib` and only catches ImportError, so
  # the leftover broken pyg_lib turns into a hard OSError. Uninstall the whole
  # prebuilt set (both plain and -rocm names) before rebuilding from source.
  # pyg_lib is intentionally NOT rebuilt/reinstalled: torch_geometric and the
  # source-built torch_sparse work without it (accelerated sampling ops only).
  python -m pip uninstall -y \
    pyg-lib pyg-lib-rocm \
    torch-scatter torch-scatter-rocm \
    torch-sparse torch-sparse-rocm \
    torch-cluster torch-cluster-rocm \
    torch-spline-conv torch-spline-conv-rocm 2>/dev/null || true

  subbanner "Building torch-scatter/sparse/cluster/spline-conv from ROCm forks"
  PYG_FRONTIER="${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}"
  mkdir -p "$PYG_FRONTIER"; cd "$PYG_FRONTIER"

  build_pyg() {  # name repo_url ref
    local name="$1" url="$2" ref="$3"
    if [[ ! -d "$name/.git" ]]; then git clone --recursive "$url" "$name"; fi
    pushd "$name" >/dev/null
    git fetch --all; git checkout "$ref"; git submodule update --init --recursive
    rm -rf build
    CC=gcc CXX=g++ PYTORCH_ROCM_ARCH=gfx90a FORCE_CUDA=1 python setup.py build
    CC=gcc CXX=g++ PYTORCH_ROCM_ARCH=gfx90a FORCE_CUDA=1 python setup.py install
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



# ============================================================
# pymatgen reinstall from GitHub + verification
# ============================================================
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

############################################################
# mpi4py 3.1.5
############################################################
MPI4PY_FRONTIER="${INSTALL_ROOT}/MPI4PY-Frontier"
export MPI4PY_FRONTIER
mkdir -p "$MPI4PY_FRONTIER"
cd "$MPI4PY_FRONTIER"

if [[ ! -d mpi4py/.git ]]; then
  git clone -b 3.1.5 https://github.com/mpi4py/mpi4py.git
fi

pushd mpi4py >/dev/null
rm -rf build

# keep setuptools<70 active in the venv for this build
CC=cc MPICC=cc "$PYTHON_BIN" -m pip install . --no-build-isolation -v

popd >/dev/null


# ============================================================
# ADIOS2
# ============================================================
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
cmake -DCMAKE_INSTALL_PREFIX=$ENV_PATH \
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
    -DPython_EXECUTABLE=$(which python) \
    -B adios2-build -S ADIOS2

cmake --build adios2-build -j32
cmake --install adios2_build 2>/dev/null || cmake --install adios2-build

# ============================================================
# DDStore
# ============================================================
banner "DDStore"
DDSTORE_FRONTIER="${INSTALL_ROOT}/DDStore-Frontier"
export DDSTORE_FRONTIER
mkdir -p "$DDSTORE_FRONTIER"
cd "$DDSTORE_FRONTIER"

git clone git@github.com:ORNL/DDStore.git || true
pushd DDStore >/dev/null
CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null

# ============================================================
# DeepHyper
# ============================================================
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

# ============================================================
# GPTL
# ============================================================
banner "GPTL"
GPTL_FRONTIER="${INSTALL_ROOT}/GPTLFrontier"
export GPTL_FRONTIER
mkdir -p "$GPTL_FRONTIER"
cd "$GPTL_FRONTIER"

wget https://github.com/jmrosinski/GPTL/releases/download/v8.1.1/gptl-8.1.1.tar.gz
tar xvf gptl-8.1.1.tar.gz
pushd gptl-8.1.1 >/dev/null
./configure --prefix=$INSTALL_ROOT --disable-libunwind CC=cc CXX=CC FC=ftn
make install
popd >/dev/null

git clone git@github.com:jychoi-hpc/gptl4py.git || true
pushd gptl4py >/dev/null
GPTL_DIR=$INSTALL_ROOT CC=cc CXX=CC pip_retry . --no-build-isolation --verbose
popd >/dev/null


# ============================================================
# hydragnn install
# ============================================================
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
