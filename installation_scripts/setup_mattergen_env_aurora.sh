#!/usr/bin/env bash

# Build the MatterGen runtime on Aurora, following HydraGNN's Aurora recipe
# (scripts/hpc/alcf/aurora/installation/install.sh):
#   - PyTorch (XPU), NumPy, SciPy, mpi4py come from `module load frameworks`;
#     a --system-site-packages venv inherits them and they are never reinstalled.
#   - torch_scatter / torch_sparse / torch_cluster are built from source (CPU
#     kernels) per ALCF's PyG guidance; torch_geometric is pure Python.
#   - ADIOS2 is built from source with MPI + Python bindings.
#   - MatterGen is installed editable with --no-deps.
#
# Run (login node or compute node):
#   nohup installation_scripts/setup_mattergen_env_aurora.sh > install_aurora.log 2>&1 &
set -Eeuo pipefail

hr()        { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner()    { hr; echo ">>> $1"; hr; }
die()       { echo "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: installation_scripts/setup_mattergen_env_aurora.sh [options]

Options:
  --install-root PATH  Directory for the venv, ADIOS2 and build trees.
  --venv-path PATH     Install the venv into PATH (default: INSTALL_ROOT/mattergen_venv).
  --recreate           Remove and recreate an existing venv.
  -h, --help           Show this help message.

Environment overrides:
  INSTALL_ROOT, VENV_PATH, ADIOS2_VERSION (default v2.10.2), BUILD_JOBS (default 16),
  PYG_SCATTER_REF, PYG_SPARSE_REF, PYG_CLUSTER_REF (git refs; default: repo HEAD)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$(dirname "${REPO_ROOT}")/MatterGen-Installation-Aurora}"
VENV_PATH="${VENV_PATH:-}"
RECREATE=0

while (( $# > 0 )); do
    case "$1" in
        --install-root)
            (( $# >= 2 )) || die "--install-root requires a path"
            INSTALL_ROOT="$2"; shift 2 ;;
        --venv-path)
            (( $# >= 2 )) || die "--venv-path requires a path"
            VENV_PATH="$2"; shift 2 ;;
        --recreate)
            RECREATE=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "unknown option: $1" ;;
    esac
done

VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/mattergen_venv}"
ADIOS2_VERSION="${ADIOS2_VERSION:-v2.10.2}"
BUILD_JOBS="${BUILD_JOBS:-16}"

[[ -f "${REPO_ROOT}/pyproject.toml" && -d "${REPO_ROOT}/mattergen" ]] || \
    die "could not locate the MatterGen repository root"

banner "MatterGen Aurora environment setup ($(date))"
echo "REPO_ROOT    = ${REPO_ROOT}"
echo "INSTALL_ROOT = ${INSTALL_ROOT}"
echo "VENV_PATH    = ${VENV_PATH}"

# Aurora compute nodes reach the internet only through the ALCF proxy.
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-${HTTP_PROXY}}"
export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}"
export no_proxy="${no_proxy:-admin,*.hostmgmt.cm.aurora.alcf.anl.gov,*.alcf.anl.gov,localhost}"

# Keep caches on Lustre instead of the small home filesystem.
mkdir -p "${INSTALL_ROOT}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${INSTALL_ROOT}/.pip-cache}"
mkdir -p "${PIP_CACHE_DIR}"

unset PYTHONHOME PYTHONPATH 2>/dev/null || true
export PYTHONNOUSERSITE=1

banner "Modules"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module-to-load-aurora.sh"
module -t list 2>&1 || true
PYTHON_BIN="$(command -v python3 || true)"
[[ -n "${PYTHON_BIN}" ]] || die "python3 unavailable after 'module load frameworks'"

banner "Create venv (inherits frameworks site-packages)"
if (( RECREATE )) && [[ -e "${VENV_PATH}" ]]; then
    echo "Removing existing venv: ${VENV_PATH}"
    rm -rf "${VENV_PATH}"
fi
if [[ ! -e "${VENV_PATH}" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_PATH}"
elif [[ ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
    die "${VENV_PATH} exists but is not a venv"
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
VENV_SITEPKG="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
echo "python       = $(command -v python) ($(python --version 2>&1))"
echo "VENV_SITEPKG = ${VENV_SITEPKG}"

pip_retry() {
    local n=0
    until python -m pip install "$@"; do
        n=$((n + 1))
        (( n >= 3 )) && die "pip install failed after 3 attempts: $*"
        echo "pip install failed (attempt ${n}), retrying in 10s..."
        sleep 10
    done
}

banner "Pin frameworks packages so pip never replaces them"
python -m pip install --upgrade pip setuptools wheel packaging
CONSTRAINTS_FILE="${INSTALL_ROOT}/pip-constraints.txt"
python - "${CONSTRAINTS_FILE}" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

# Frameworks-provided packages compiled against Aurora's oneAPI/MPICH stack.
frameworks = ["torch", "torchvision", "numpy", "scipy", "mpi4py", "matplotlib", "pandas", "h5py"]
lines = [f"{name}=={metadata.version(name)}" for name in frameworks]
# emmet-core 0.87.x needs pymatgen.core.graphs, absent from this pymatgen stack.
lines += ["emmet-core==0.85.1", "pymatgen==2024.10.29", "monty==2024.7.30", "torch-geometric==2.7.0"]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
PY
export PIP_CONSTRAINT="${CONSTRAINTS_FILE}"
cat "${CONSTRAINTS_FILE}"

python - <<'PY'
import torch
print("torch", torch.__version__, torch.__file__)
print("xpu available:", hasattr(torch, "xpu") and torch.xpu.is_available())
PY

banner "ADIOS2 ${ADIOS2_VERSION} (build from source, MPI + Python)"
pip_retry pybind11
ADIOS2_SRC="${INSTALL_ROOT}/ADIOS2-src"
ADIOS2_BUILD="${ADIOS2_SRC}/build"
ADIOS2_INSTALL="${INSTALL_ROOT}/adios2"
[[ -d "${ADIOS2_SRC}/.git" ]] || git clone https://github.com/ornladios/ADIOS2.git "${ADIOS2_SRC}"
git -C "${ADIOS2_SRC}" fetch --all --tags
git -C "${ADIOS2_SRC}" checkout "${ADIOS2_VERSION}"

MPICC_BIN="$(command -v mpicc || true)"
MPICXX_BIN="$(command -v mpicxx || true)"
[[ -n "${MPICC_BIN}" && -n "${MPICXX_BIN}" ]] || die "mpicc/mpicxx not found"

cmake -S "${ADIOS2_SRC}" -B "${ADIOS2_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${ADIOS2_INSTALL}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DADIOS2_BUILD_EXAMPLES=OFF \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_Fortran=OFF \
    -DADIOS2_USE_Python=ON \
    -DPython_EXECUTABLE="$(command -v python)" \
    -DPython3_EXECUTABLE="$(command -v python)" \
    -DADIOS2_USE_HDF5=OFF \
    -DADIOS2_USE_SST=OFF \
    -DADIOS2_USE_BZip2=OFF \
    -DADIOS2_USE_PNG=OFF \
    -DADIOS2_USE_DataSpaces=OFF \
    -DADIOS2_USE_DataMan=OFF \
    -DADIOS2_USE_CUDA=OFF \
    -DADIOS2_USE_HIP=OFF \
    -DADIOS2_USE_SYCL=OFF \
    -DCMAKE_C_COMPILER="${MPICC_BIN}" \
    -DCMAKE_CXX_COMPILER="${MPICXX_BIN}"
cmake --build "${ADIOS2_BUILD}" -j"${BUILD_JOBS}"
cmake --install "${ADIOS2_BUILD}"

if [[ -d "${ADIOS2_INSTALL}/lib64" ]]; then ADIOS2_LIBDIR="${ADIOS2_INSTALL}/lib64"; else ADIOS2_LIBDIR="${ADIOS2_INSTALL}/lib"; fi
# ADIOS2 may nest the bindings under ${ADIOS2_INSTALL}${VENV_SITEPKG}; link whichever exists.
ADIOS2_PY_PKG="$(find "${ADIOS2_INSTALL}" -type d -name adios2 -path '*site-packages/adios2' -print -quit)"
[[ -n "${ADIOS2_PY_PKG}" ]] || die "could not locate the ADIOS2 Python package under ${ADIOS2_INSTALL}"
rm -rf "${VENV_SITEPKG}/adios2"
ln -s "${ADIOS2_PY_PKG}" "${VENV_SITEPKG}/adios2"
echo "Linked ${VENV_SITEPKG}/adios2 -> ${ADIOS2_PY_PKG}"

# Make ADIOS2 visible on every venv activation.
ACTIVATE_HOOK="# mattergen-aurora: ADIOS2 runtime"
if ! grep -qF "${ACTIVATE_HOOK}" "${VENV_PATH}/bin/activate"; then
    cat >> "${VENV_PATH}/bin/activate" <<EOF
${ACTIVATE_HOOK}
export ADIOS2_DIR="${ADIOS2_INSTALL}"
export PATH="${ADIOS2_INSTALL}/bin:\${PATH}"
export LD_LIBRARY_PATH="${ADIOS2_LIBDIR}:\${LD_LIBRARY_PATH:-}"
EOF
fi
export ADIOS2_DIR="${ADIOS2_INSTALL}"
export PATH="${ADIOS2_INSTALL}/bin:${PATH}"
export LD_LIBRARY_PATH="${ADIOS2_LIBDIR}:${LD_LIBRARY_PATH:-}"

banner "PyTorch Geometric + auxiliary packages (CPU builds per ALCF guidance)"
pip_retry torch-geometric==2.7.0

TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
export FORCE_ONLY_CPU=1 MAX_JOBS="${BUILD_JOBS}"

PYG_SRCDIR="${INSTALL_ROOT}/pyg-aux-src"
mkdir -p "${PYG_SRCDIR}"
declare -A PYG_REFS=(
    [pytorch_scatter]="${PYG_SCATTER_REF:-}"
    [pytorch_sparse]="${PYG_SPARSE_REF:-}"
    [pytorch_cluster]="${PYG_CLUSTER_REF:-}"
)
# MatterGen imports torch_scatter/torch_sparse only; mattersim needs no spline_conv or pyg_lib.
for lib in pytorch_scatter pytorch_sparse pytorch_cluster; do
    rm -rf "${PYG_SRCDIR:?}/${lib}"
    git clone "https://github.com/rusty1s/${lib}.git" "${PYG_SRCDIR}/${lib}"
    pushd "${PYG_SRCDIR}/${lib}" >/dev/null
    [[ -n "${PYG_REFS[${lib}]}" ]] && git checkout "${PYG_REFS[${lib}]}"
    git submodule update --init --recursive
    echo "${lib} @ $(git rev-parse --short HEAD)"
    # ALCF: skip the OpenMP backend probe, which misdetects frameworks' torch.
    sed -i "s|if ('backend: OpenMP' in info and 'OpenMP not found' not in info|if (False|g" setup.py
    # Retry: Lustre occasionally fails the compiler's .o.tmp -> .o rename.
    pip_retry --no-build-isolation --no-deps --force-reinstall -v .
    popd >/dev/null
    rm -rf "${PYG_SRCDIR:?}/${lib}"
done

banner "MatterGen runtime dependencies"
cd "${REPO_ROOT}"
# Install pyproject dependencies except: backend packages (frameworks / built above),
# dev tools, numpy<2 and matplotlib==3.8.4 (frameworks' torch needs NumPy 2), and
# mattersim (installed below without its heavy, unused dependency tree).
python - <<'PY'
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
excluded = {
    canonicalize_name(name)
    for name in (
        "torch", "torchvision", "torchaudio", "torch-cluster", "torch-geometric",
        "torch-scatter", "torch-sparse", "numpy", "matplotlib", "mattersim",
        "autopep8", "jupyterlab", "notebook", "pylint", "pytest",
    )
}
runtime = [d for d in dependencies if canonicalize_name(Requirement(d).name) not in excluded]
print("installing:", " ".join(runtime), flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", *runtime])
PY

# Packages that declare a torch requirement go in with --no-deps: pip rejects
# frameworks' torch 2.13.0a0 for some specifiers and would pull a CUDA wheel.
# This set covers what mattergen imports from mattersim (potential, batch_relax).
pip_retry deprecated loguru prettytable scikit-learn opt-einsum lightning-utilities sympy
pip_retry --no-deps "mattersim>=1.1" "torch-ema>=0.3" "torchmetrics>=0.10" \
    "torch-runstats>=0.2" "e3nn>=0.5" opt-einsum-fx

banner "Install MatterGen (editable, --no-deps)"
pip_retry --no-deps -e "${REPO_ROOT}"

banner "Verify environment"
export VENV_PATH REPO_ROOT
python - <<'PY'
from importlib import metadata
import os
from pathlib import Path
import sys

import torch

venv = os.environ["VENV_PATH"]
import mpi4py
import numpy
for module in (torch, mpi4py, numpy):
    print(module.__name__, module.__version__, module.__file__)
    if module.__file__.startswith(venv):
        raise SystemExit(f"{module.__name__} is shadowed by a venv install; pip uninstall it from the venv")

for distribution in ("torch-geometric", "torch-scatter", "torch-sparse", "torch-cluster",
                     "emmet-core", "pymatgen", "monty", "mattersim"):
    print("distribution", distribution, metadata.version(distribution))

from emmet.core.material import PropertyOrigin
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
import adios2
import torch_cluster
import torch_geometric
import torch_scatter
import torch_sparse
from mattersim.forcefield.potential import MatterSimCalculator

if not hasattr(adios2, "FileReader"):
    raise SystemExit("adios2.FileReader is unavailable")
print("adios2", adios2.__version__, adios2.__file__)


def verify_scatter(device: str) -> None:
    source = torch.tensor([1.0, 2.0, 3.0], device=device)
    index = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
    result = torch_scatter.scatter(source, index, dim=0, dim_size=2, reduce="sum")
    if not torch.equal(result.cpu(), torch.tensor([4.0, 2.0])):
        raise SystemExit(f"torch_scatter produced an incorrect {device} result: {result}")
    print("torch_scatter", device, "verification passed")


verify_scatter("cpu")
xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
if xpu:
    verify_scatter("xpu")
else:
    print("XPU not visible (login node?); XPU verification deferred to a compute-node job")
if not torch.distributed.is_xccl_available():
    raise SystemExit("PyTorch was built without the XCCL backend")

import mattergen
import mattergen.scripts.run

mattergen_file = Path(mattergen.__file__).resolve()
mattergen_file.relative_to(Path(os.environ["REPO_ROOT"]).resolve())
train_entrypoint = Path(sys.executable).parent / "mattergen-train"
if not train_entrypoint.is_file():
    raise SystemExit(f"MatterGen training entry point is missing: {train_entrypoint}")

print("MatterGen environment verification passed")
print("python", sys.version.split()[0], sys.executable)
print("torch", torch.__version__, "xpu_available", xpu,
      "device_count", torch.xpu.device_count() if xpu else 0)
print("torch_geometric", torch_geometric.__version__)
print("mattergen", mattergen_file)
print("mattergen_train", train_entrypoint)
PY

banner "Done"
cat <<EOF
Venv:   ${VENV_PATH}
ADIOS2: ${ADIOS2_INSTALL} (${ADIOS2_VERSION})

To activate later:
  source ${SCRIPT_DIR}/module-to-load-aurora.sh
  source ${VENV_PATH}/bin/activate
EOF
