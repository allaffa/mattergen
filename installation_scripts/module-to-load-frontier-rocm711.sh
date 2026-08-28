#!/usr/bin/env bash

# OLCF's recommended stable PyTorch/PyG module stack for Frontier.
module reset
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module unload darshan-runtime 2>/dev/null || true

# cpe/26.03 is not Frontier's system-default CPE, so expose its libraries to
# Python extension modules and the ROCm wheels.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
