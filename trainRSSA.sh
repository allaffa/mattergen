#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J mattergen_rhea_train
#SBATCH -o jobOutputs/mattergen_rhea_train_cell-%j.out
#SBATCH -e jobOutputs/mattergen_rhea_train_cell-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -N 1
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

# this might be weird...
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

#mattergen-generate ./results/ --model_path ./checkpoints/mattergen_base --batch_size 16 --num_batches 1 --sampling_config_path ./sampling_conf 

####data preparation####
#git lfs pull -I data-release/mp-20/ --exclude=""
#unzip data-release/mp-20/mp_20.zip -d datasets
#csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

#srun --ntasks-per-node=8  mattergen-train data_module=mp_20 ~trainer.logger

python ckptLRAdjust.py /lustre/orion/lrn070/proj-shared/patxi/patxi/mattergen/mattergen/conf/default.yaml 1e-5

srun --ntasks-per-node=8  mattergen-train data_module=RSSA_data ~trainer.logger

