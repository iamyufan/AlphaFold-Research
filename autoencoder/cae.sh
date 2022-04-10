#!/bin/bash

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=ae_2d
#SBATCH --output=ae_2d.%j.out
#SBATCH --error=ae_2d.%j.err
#SBATCH -t 0-72:00:00
#SBATCH -p publicgpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G

source /home/hgao53/miniconda3/bin/activate /home/hgao53/miniconda3/envs/af2_test

python3 main.py
