#!/bin/bash               
#SBATCH -J vr646
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --comment pytorch

echo "Slurm Job on `hostname`"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


module purge
module load cuda/12.1

torchrun --standalone --nnodes=1 --nproc_per_node 4 --no_python sevenn input.yaml -d 


