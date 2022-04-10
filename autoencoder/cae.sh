#!/bin/bash

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=ae_2d_RTX2080
#SBATCH --output=ae_2d_RTX2080.%j.out
#SBATCH --error=ae_2d_RTX2080.%j.err
#SBATCH -t 0-72:00:00
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:2
#SBATCH -C RTX2080
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G

source /home/hgao53/miniconda3/bin/activate /home/hgao53/miniconda3/envs/af2_test

python3 main.py
