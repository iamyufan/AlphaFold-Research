#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-72:00:00
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --job-name=nv
#SBATCH -o nv.out
#SBATCH -J nv

nvidia-smi