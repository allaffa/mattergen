#!/usr/bin/env bash

# Build and pack the focused MatterGen runtime recommended by OLCF for
# Frontier: Python 3.12, PyTorch 2.10.0, ROCm 7.1.1, and ROCm PyG wheels.
set -Eeuo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: installation_scripts/setup_mattergen_env_frontier_rocm711.sh [options]

Options:
  --env-path PATH      Install into PATH instead of the project-Lustre default.
  --archive-path PATH  Write the conda-pack archive to PATH.
  --recreate           Remove and recreate an existing conda environment.
  -h, --help           Show this help message.

The default environment is:
  /lustre/orion/lrn070/proj-shared/zb7/envs/mattergen-rocm711

MATTERGEN_ENV_PATH may also set the environment path.
MATTERGEN_ENV_ARCHIVE may set the archive path. By default it is ENV_PATH.tar.gz.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PATH="${MATTERGEN_ENV_PATH:-/lustre/orion/lrn070/proj-shared/zb7/envs/mattergen-rocm711}"
ARCHIVE_PATH="${MATTERGEN_ENV_ARCHIVE:-}"
RECREATE=0

while (( $# > 0 )); do
    case "$1" in
        --env-path)
            (( $# >= 2 )) || die "--env-path requires a path"
            ENV_PATH="$2"
            shift 2
            ;;
        --archive-path)
            (( $# >= 2 )) || die "--archive-path requires a path"
            ARCHIVE_PATH="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

ARCHIVE_PATH="${ARCHIVE_PATH:-${ENV_PATH}.tar.gz}"
ENV_PARENT="$(dirname "${ENV_PATH}")"

[[ -f "${REPO_ROOT}/pyproject.toml" && -d "${REPO_ROOT}/mattergen" ]] || \
    die "could not locate the MatterGen repository root"

# Keep installation caches on project Lustre instead of the user's NFS home.
mkdir -p "${ENV_PARENT}"
export CONDA_PKGS_DIRS="${MATTERGEN_CONDA_PKGS_DIRS:-${ENV_PARENT}/.conda-pkgs}"
export PIP_CACHE_DIR="${MATTERGEN_PIP_CACHE_DIR:-${ENV_PARENT}/.pip-cache}"
mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

# Do not let the caller's active Python/Conda environment participate in this
# build. The miniforge module below supplies the conda executable.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
    CONDA_PROMPT_MODIFIER PYTHONHOME PYTHONPATH 2>/dev/null || true
export PYTHONNOUSERSITE=1

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module-to-load-frontier-rocm711.sh"
command -v conda >/dev/null 2>&1 || die "conda is unavailable after loading miniforge3"
eval "$(conda shell.bash hook)"

if (( RECREATE )) && [[ -e "${ENV_PATH}" ]]; then
    echo "Removing existing conda environment: ${ENV_PATH}"
    conda env remove --prefix "${ENV_PATH}" --yes || \
        die "could not remove ${ENV_PATH}; remove or rename it manually"
    [[ ! -e "${ENV_PATH}" ]] || die "conda left files behind in ${ENV_PATH}"
fi

if [[ ! -e "${ENV_PATH}" ]]; then
    echo "Creating Python 3.12 environment: ${ENV_PATH}"
    conda create --yes --prefix "${ENV_PATH}" --channel conda-forge python=3.12
elif [[ ! -f "${ENV_PATH}/conda-meta/history" ]]; then
    die "${ENV_PATH} exists but is not a conda environment"
fi

conda activate "${ENV_PATH}"
cd "${REPO_ROOT}"

conda install --yes --prefix "${ENV_PATH}" --channel conda-forge conda-pack
python -m pip install --upgrade pip setuptools wheel ninja packaging scipy
python -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/rocm7.1

python -m pip install \
    --constraint "${SCRIPT_DIR}/frontier-rocm711-constraints.txt" \
    --only-binary=:all: \
    adios2==2.10.2.100774

# Install MatterGen's regular runtime dependencies before the compiled PyG
# extensions. The backend exclusions prevent MatterGen's generic CUDA/CPU PyG
# requirements from taking precedence over the Frontier-specific ROCm wheels.
python - "${SCRIPT_DIR}/frontier-rocm711-constraints.txt" <<'PY'
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

constraints = Path(sys.argv[1])
dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
backend_packages = {
    canonicalize_name(name)
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "torch-cluster",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-spline-conv",
        "pyg-lib",
    )
}
development_packages = {
    canonicalize_name(name)
    for name in ("autopep8", "jupyterlab", "notebook", "pylint", "pytest")
}
runtime_dependencies = [
    dependency
    for dependency in dependencies
    if canonicalize_name(Requirement(dependency).name)
    not in backend_packages | development_packages
]
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--constraint",
        str(constraints),
        *runtime_dependencies,
    ]
)
PY

# torch-geometric itself is pure Python. Install it with its normal runtime
# dependencies, then remove every potentially overlapping compiled PyG
# distribution before installing the ABI-matched ROCm wheels last.
python -m pip install \
    --constraint "${SCRIPT_DIR}/frontier-rocm711-constraints.txt" \
    torch-geometric==2.7.0

python -m pip uninstall --yes \
    pyg-lib pyg-lib-rocm \
    torch-scatter torch-scatter-rocm \
    torch-sparse torch-sparse-rocm \
    torch-cluster torch-cluster-rocm \
    torch-spline-conv torch-spline-conv-rocm

# These post2 wheels were published for PyTorch 2.10 and provide the standard
# torch_scatter/torch_sparse/torch_cluster import names on ROCm. MatterGen does
# not require pyg-lib or torch-spline-conv, so neither is installed.
python -m pip install \
    --no-deps \
    --only-binary=:all: \
    --constraint "${SCRIPT_DIR}/frontier-rocm711-constraints.txt" \
    torch-scatter-rocm==2.1.2.post2 \
    torch-sparse-rocm==0.6.18.post2 \
    torch-cluster-rocm==1.6.3.post2

# Install a non-editable copy only to provide MatterGen's console entry points
# inside the packed environment. JamieTest.sh sets PYTHONPATH to REPO_ROOT, so
# training always imports the user's current checkout after staging.
python -m pip install --no-deps "${REPO_ROOT}"

export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}"
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
            f"expected {distribution} {expected_version}, found {installed_version}"
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
        f"conflicting distribution is installed: "
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
            f"torch_scatter produced an incorrect {device} result: {result}"
        )
    print("torch_scatter", device, "verification passed", flush=True)


verify_scatter("cpu")
if torch.cuda.is_available():
    verify_scatter("cuda")
else:
    print("ROCm GPU verification deferred to JamieTest.sh batch preflight", flush=True)

print("import local MatterGen checkout", flush=True)
import mattergen
import mattergen.scripts.run

expected_repo = Path(os.environ["REPO_ROOT"]).resolve()
mattergen_file = Path(mattergen.__file__).resolve()
try:
    mattergen_file.relative_to(expected_repo)
except ValueError as exc:
    raise SystemExit(
        f"mattergen resolved outside the checkout: {mattergen_file}"
    ) from exc

torch_release = torch.__version__.split("+", 1)[0]
hip_release = torch.version.hip or ""
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, found {sys.version.split()[0]}")
if torch_release != "2.10.0":
    raise SystemExit(f"expected torch 2.10.0, found {torch.__version__}")
if not hip_release.startswith("7.1"):
    raise SystemExit(f"expected HIP 7.1.x, found {torch.version.hip!r}")
if not adios2.__version__.startswith("2.10.2"):
    raise SystemExit(f"expected the adios2 2.10.2 series, found {adios2.__version__}")
if not hasattr(adios2, "FileReader"):
    raise SystemExit("adios2.FileReader is unavailable")
if not torch.distributed.is_nccl_available():
    raise SystemExit("PyTorch was installed without RCCL/NCCL support")
if os.environ.get("SLURM_JOB_ID") and not torch.cuda.is_available():
    raise SystemExit("ROCm GPU is unavailable inside the Slurm allocation")

train_entrypoint = Path(sys.executable).parent / "mattergen-train"
if not train_entrypoint.is_file():
    raise SystemExit(f"MatterGen training entry point is missing: {train_entrypoint}")

print("MatterGen environment verification passed")
print("python", sys.version.split()[0], sys.executable)
print("torch", torch.__version__, "HIP", torch.version.hip)
print("adios2", adios2.__version__, adios2.__file__)
print("torch_geometric", torch_geometric.__version__)
print("mattergen", mattergen_file)
print("mattergen_train", train_entrypoint)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

echo "Installing conda-pack and writing relocatable environment archive"
mkdir -p "$(dirname "${ARCHIVE_PATH}")"
conda pack \
    --format tar.gz \
    --n-threads -1 \
    --force \
    --prefix "${ENV_PATH}" \
    --output "${ARCHIVE_PATH}"
[[ -s "${ARCHIVE_PATH}" ]] || die "conda-pack did not create ${ARCHIVE_PATH}"

echo "Environment ready: ${ENV_PATH}"
echo "Packed archive: ${ARCHIVE_PATH}"
echo "Submit with: MATTERGEN_ENV_ARCHIVE=${ARCHIVE_PATH} sbatch JamieTest.sh"
