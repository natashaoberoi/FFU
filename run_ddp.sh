#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition=gpu
#SBATCH --mail-user=indro@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# source activate your_env  # activate your python env
cd ~/scratch/zt1/project/msml612/user/indro/FFU

module purge
module load python/3.10.10
module load cuda-new/x86_64/12.3.0

pip install --user regex ftfy
pip install --user torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --user transformers
pip install --user pycocotools

# Debug info
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python ddp_updated.py
