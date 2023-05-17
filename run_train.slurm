#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug

srun tools/dist_train.sh configs/posec3d/slowonly_r50_affwild_xsub/joint.py 8 --validate --test-last --test-best
deactivate
