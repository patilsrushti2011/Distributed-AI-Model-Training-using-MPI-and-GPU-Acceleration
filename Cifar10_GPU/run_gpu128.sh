#!/bin/bash
#SBATCH --job-name=cifar10_128_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=cifar10_128_gpu_%j.out
#SBATCH --error=cifar10_128_gpu_%j.err

module purge
conda activate py395_env  
cd /home/chuk372/cpu_serial
python cifar10_128batch.py

