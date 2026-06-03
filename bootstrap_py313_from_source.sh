#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./bootstrap_py313_from_source.sh [--venv <path>] [--python <python-bin>] [--keep-existing]

Bootstraps a Python 3.13 environment from source-friendly installs and installs MatterGen.

Options:
  --venv <path>       Virtual environment path (default: .venv)
  --python <bin>      Python interpreter to use (default: python3.13)
  --keep-existing     Do not delete an existing venv directory
  -h, --help          Show this help message
EOF
}

VENV_PATH=".venv"
PYTHON_BIN="python3.13"
KEEP_EXISTING="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --keep-existing)
      KEEP_EXISTING="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root (where pyproject.toml exists)." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' was not found in PATH." >&2
  exit 1
fi

if [[ -d "$VENV_PATH" && "$KEEP_EXISTING" != "true" ]]; then
  echo "Removing existing virtual environment at $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

PYTHON="$VENV_PATH/bin/python"
PIP="$PYTHON -m pip"
export BOOTSTRAP_VENV_PATH="$VENV_PATH"

$PIP install --upgrade pip
$PIP install "setuptools==80.9.0" wheel cmake ninja flit_core

PYG_SOURCE_PKGS=(
  torch-scatter
  torch-sparse
  torch-cluster
  torch-geometric
)

# Install dependencies declared in pyproject.toml except project editable install and PyG source packages.
"$PYTHON" - <<'PY'
import os
import subprocess
import tomllib
from pathlib import Path

pyproject = Path("pyproject.toml")
obj = tomllib.loads(pyproject.read_text())
deps = obj["project"]["dependencies"]

skip = {
    "torch_scatter", "torch-scatter",
    "torch_sparse", "torch-sparse",
    "torch_cluster", "torch-cluster",
    "torch_geometric", "torch-geometric",
}

filtered = []
for dep in deps:
    name = dep.split(";")[0].split("[")[0]
    for sep in ["<=", ">=", "==", "!=", "~=", ">", "<"]:
        if sep in name:
            name = name.split(sep)[0]
            break
    normalized = name.strip().replace("_", "-").lower()
    if normalized in skip:
        continue
    filtered.append(dep)

if filtered:
    venv_path = os.environ["BOOTSTRAP_VENV_PATH"]
    cmd = [str(Path(venv_path) / "bin" / "python"), "-m", "pip", "install", *filtered]
    subprocess.check_call(cmd)
PY

MAX_JOBS_DEFAULT="$($PYTHON - <<'PY'
import os
print(max(1, (os.cpu_count() or 1) // 2))
PY
)"
export MAX_JOBS="${MAX_JOBS:-$MAX_JOBS_DEFAULT}"

echo "Building PyG extensions from source with MAX_JOBS=$MAX_JOBS"
$PIP install --no-cache-dir --force-reinstall --no-binary=torch-scatter --no-build-isolation torch-scatter
$PIP install --no-cache-dir --force-reinstall --no-binary=torch-sparse --no-build-isolation torch-sparse
$PIP install --no-cache-dir --force-reinstall --no-binary=torch-cluster --no-build-isolation torch-cluster
$PIP install --no-cache-dir --force-reinstall --no-binary=torch-geometric --no-build-isolation torch-geometric

$PIP install -e . --no-deps

echo
echo "Bootstrap complete. Activate with:"
echo "  source $VENV_PATH/bin/activate"
