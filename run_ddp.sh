#!/bin/bash
#SBATCH --job-name=ffu_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=ddp_%j_practice.out
#SBATCH --error=logs/%x_%j_practice.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=noberoi1@umd.edu

# source activate your_env  # activate your python env
cd ~/scratch/zt1/project/msml612/user/noberoi1/FFU

module purge
module load python/3.10.10
module load cuda-new/x86_64/12.3.0

pip install --user regex ftfy
pip install --user torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121

# Debug info
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python ddp_updated.py
