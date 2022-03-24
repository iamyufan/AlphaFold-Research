#!/bin/bash

#SBATCH --job-name=conv2d_ae
#SBATCH --output=conv2d_ae.out
#SBATCH --error=conv2d_ae.err
#SBATCH -t 0-72:00:00
#SBATCH -p publicgpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH --cpu_per_task=8
#SBATCH --mem=100G

source /home/hgao53/miniconda3/bin/activate /home/hgao53/miniconda3/envs/af2_test

python3 main.py
