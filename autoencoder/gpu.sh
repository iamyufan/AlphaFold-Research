#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=GTX1080
#SBATCH --output=GTX1080.%j.out
#SBATCH --error=GTX1080.%j.err
#SBATCH -t 0-72:00:00
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:2
#SBATCH -C GTX1080
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G

source /home/hgao53/miniconda3/bin/activate /home/hgao53/miniconda3/envs/af2_test

nvidia-smi
