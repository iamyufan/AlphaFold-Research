#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-72:00:00
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --job-name=conv2d_ae_1
#SBATCH -o conv2d_ae_1.out
#SBATCH -J con2d_ae_1
#SBATCH --mem=100G

source /home/hgao53/miniconda3/bin/activate /home/hgao53/miniconda3/envs/af2_test

python3 main.py
