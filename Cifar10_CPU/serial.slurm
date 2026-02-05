#!/bin/bash
#SBATCH --job-name=cifar10_mobilenet_cpu
#SBATCH --output=logs_cifar10_cpu_%j.out
#SBATCH --error=logs_cifar10_cpu_%j.err
#SBATCH --time=15:00:00          # adjust as needed
#SBATCH --partition=cpu          # your CPU partition name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=16G                # RAM

# Load modules / activate env (adapt to your cluster)
module load python/3.10
module load anaconda3/anaconda3 

conda activate py395_env   
/home/chuk372/cpu_serial

python cifar10_serial_mobilenet_224.py

