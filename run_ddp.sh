#!/bin/bash
#SBATCH --job-name=ddp_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ddp_%j_1.out

# source activate your_env  # activate your python env
cd ~/scratch/zt1/project/msml612/user/noberoi1/FFU

module load cuda/12.1.1

torchrun --nproc_per_node=4 run_DDP.py
