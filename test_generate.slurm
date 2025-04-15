#!/bin/bash
#SBATCH --job-name=llada-a100
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=val_test.log
#SBATCH --error=val_test.log


export PYTHONUNBUFFERED=1


echo "🌱 Setting up environment..."
module load miniconda
source /vast/palmer/apps/avx2/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate llada_env

echo "🔥 Trying to import torch..."
python -c "import torch; print('Torch version:', torch.__version__)"

echo "🚀 Launching Gradio app"
/home/alm293/.conda/envs/llada_env/bin/python ~/llada_lora_scheduler/LLaDA/val_test.py