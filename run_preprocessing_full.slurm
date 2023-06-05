#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=16:0:0
#SBATCH --qos=gpu
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

cd tools/data
srun python affwild_preproc_full.py
deactivate
