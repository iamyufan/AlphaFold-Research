#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=conv2d_ae_1
#SBATCH --ntasks=1
#SBATCH -p publicgpu
#SBATCH -o conv2d_ae_1.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

source activate /home/hgao53/miniconda3/envs/af2_model

python3 main.py