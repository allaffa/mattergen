#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J generateRHEA
#SBATCH -o ../jobOutputs/gen/mattergen_rhea_gen-%j.out
#SBATCH -e ../jobOutputs/gen/mattergen_rhea_gen-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch 
##SBATCH -q debug
#SBATCH -N 32
##SBATCH -S 1

 
CASE_ROOT=/lustre/orion/lrn070/proj-shared/patxi/patxi/mattergen
source $CASE_ROOT/module-to-load-frontier-rocm720.sh

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/HydraGNN-Installation-Frontier/ADIOS2-Frontier/adios2-build/lib/python3.11/site-packages/:$PYTHONPATH

which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"

module unload darshan-runtime
module list

export all_proxy=http://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

## Getting error without these after 20 nodes
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export FI_MR_CACHE_MONITOR=disabled

## aws-ofi-rccl plugin settings
export TORCH_NCCL_HIGH_PRIORITY=1
export FI_CXI_RDV_PROTO=alt_read

export PATH_TO_THE_PLUGIN_DIRECTORY=/lustre/orion/lrn070/world-shared/mlupopa/AWI_OFI_RCCL_ROCm631/aws-ofi-rccl/lib
export LD_LIBRARY_PATH=${PATH_TO_THE_PLUGIN_DIRECTORY}:$LD_LIBRARY_PATH
 
export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
 
export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

# this is the path to the best "scratch" checkpoint from training on legacy mattergen
export MODEL_PATH=/lustre/orion/lrn070/proj-shared/patxi/patxi/checkpoints/scratch/2026-04-19/01-20-12
export RESULTS_PATH=/lustre/orion/lrn070/proj-shared/patxi/patxi/results/rhea_scratch_weighted  # Samples will be written to this directory
mkdir -p $RESULTS_PATH
#srun --ntasks-per-node=8 mattergen-generate $RESULTS_PATH --model_path=$MODEL_PATH  --batch_size=8 --num_batches=1 --sampling_config_overrides='["++condition_loader_partial.num_atoms_distribution=RHEA_DATA"]'
srun --ntasks-per-node=8 mattergen-generate $RESULTS_PATH --model_path=$MODEL_PATH  --batch_size=1024 --num_batches=1 --num_atoms_distribution="RSSA"



# unzip ${RESULTS_PATH}/generated_trajectories.zip -d ${RESULTS_PATH}
# python visualizeGen.py $RESULTS_PATH 15 10


# export MODEL_PATH=/lustre/orion/lrn070/proj-shared/patxi/mattergen/outputs/singlerun/2026-03-18/11-10-24
# export RESULTS_PATH=results/rhea_fineTune
# mkdir -p $RESULTS_PATH
# srun --ntasks-per-node=8 mattergen-generate $RESULTS_PATH --model_path=$MODEL_PATH  --batch_size=16 --num_batches=1

# unzip ${RESULTS_PATH}/generated_trajectories.zip -d ${RESULTS_PATH}
# python visualizeGen.py $RESULTS_PATH 15 10