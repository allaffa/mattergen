module reset
#module load gcc/12.3.0
ml cpe/24.07
ml cce/18.0.0
ml rocm/7.2.0
ml amd-mixed/7.2.0
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
#ml cmake/3.27.9
module unload darshan-runtime
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
conda deactivate
conda activate /lustre/orion/lrn070/proj-shared/zhangp/hydragnn_venv