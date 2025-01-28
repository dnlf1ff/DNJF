#!/bin/bash
#
#SBATCH --job-name=l4i3_vr646
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --nodelist=syn08
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
conda activate hk_blue

module list

module purge
module load cuda/12.1

module list

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0,2,6,7

#sevenn input.yaml
torchrun --standalone --nnodes=1 --nproc_per_node 4 --no_python sevenn input.yaml -d 
