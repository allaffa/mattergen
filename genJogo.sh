#!/bin/bash 

export RESUTLS_PATH=/home/3pf/projects/mattergen_jaime/mattergen/results/rhea_scratch_weighted
export PYTHONPATH=~/miniconda3/envs/mattergen/bin/python

source ~/miniconda3/etc/profile.d/conda.sh

conda deactivate

conda activate /home/3pf/miniconda3/envs/mattergen

#torchrun --nproc_per_node=1 -m mattergen.scripts.generate $RESUTLS_PATH --num_atoms_distribution=RSSA --batch_size=8 --num_batches=1 --model_path=/home/3pf/projects/mattergen_lightningfree/outputs


export CUDA_VISIBLE_DEVICES="4,5"
#python -m mattergen.scripts.generate $RESUTLS_PATH --num_atoms_distribution=RSSA --batch_size=8 --num_batches=1 --model_path=/home/3pf/projects/mattergen_lightningfree/outputs
torchrun --nproc_per_node=2 -m mattergen.scripts.generate $RESUTLS_PATH --num_atoms_distribution=RSSA --batch_size=32 --num_batches=1 --model_path=/home/3pf/projects/mattergen_lightningfree/outputs