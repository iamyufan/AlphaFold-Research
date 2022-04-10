#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 100
#SBATCH --job-name=ae_2d
#SBATCH --output=ae_2d.out
#SBATCH --error=ae_2d.err
#SBATCH -t 0-72:00:00
#SBATCH -p publicgpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

source /home/hgao53/miniconda3/bin/activate /home/hgao53/miniconda3/envs/af2_test

python3 main.py
