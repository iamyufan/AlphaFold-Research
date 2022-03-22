#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=zero_padding2048
#SBATCH --ntasks=1
#SBATCH -o zero_padding2048.out
#SBATCH --time=48:00:00

source activate /home/hgao53/miniconda3/envs/af2_model

python3 zero_padding.py