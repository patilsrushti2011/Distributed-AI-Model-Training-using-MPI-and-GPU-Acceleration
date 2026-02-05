#!/bin/bash
#SBATCH --job-name=cifar_mpi_gpu
#SBATCH --partition=gpu          # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=2               # 1 MPI rank per GPU
#SBATCH --gres=gpu:2             # each gpu node has 2 GPUs
#SBATCH --time=12:00:00
#SBATCH --output=cifar_mpi_gpu128_%j.out
#SBATCH --error=cifar_mpi_gpu128_%j.err

module purge
module load anaconda3/anaconda3
module load openmpi/4.1.1
source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate py395_env        # env with CUDA-capable torch + mpi4py

cd "$HOME/cifar_10" || exit 1

mpirun --mca pml ob1 --mca btl tcp,self -np 2 \
  python cifar10_mpi_mobilenet_224.py

