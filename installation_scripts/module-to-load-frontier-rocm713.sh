# MatterGen module stack for Frontier — ROCm 7.13.0
# Mirrors the validated lumina-sdk ROCm 7.13 stack. Source this at the top of
# the installer and every job script.
module reset
ml PrgEnv-gnu/8.7.0
ml cray-mpich
ml cpe/26.03
ml miniforge3/23.11.0-0
ml amd-mixed/7.13.0
ml rocm/7.13.0
ml craype-accel-amd-gfx90a
ml git-lfs
module unload darshan-runtime || true
ml rccl-net-plugin

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
