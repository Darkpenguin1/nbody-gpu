#!/bin/bash
#SBATCH --job-name=nbody_gpu
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=nbody_gpu_%j.out
#SBATCH --error=nbody_gpu_%j.err

module load cuda/12.4

./nbody 100000 0.01 10 100000 128