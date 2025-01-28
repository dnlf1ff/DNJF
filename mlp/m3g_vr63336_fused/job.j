#!/bin/bash
#
#SBATCH --job-name=vr_cg_test
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=48:00:00
#SBATCH --output hello-%j.out

echo "Slurm Job on `hostname`"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load modules and run your programs here
#
source $HOME/.bashrc
conda activate hk_cg_af

module list

module purge
module load gnu12/12.2.0
module load cuda/12.2.1

module list

#sevenn input.yaml
torchrun --standalone --nnodes=1 --nproc_per_node 4 --no_python sevenn input.yaml -d 
#torchrun --standalone --nnodes=1 --nproc_per_node 4 --no_python sevenn input_vanila.yaml -d 
