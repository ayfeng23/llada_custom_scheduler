#!/bin/bash
#SBATCH --job-name=llada-a100
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=llada_output.log
#SBATCH --error=llada_output.log

# SBATCH --nodelist=r815u35n07  # uncomment if you want to pin to a specific one


export PYTHONUNBUFFERED=1


echo "🌱 Setting up environment..."
module load miniconda
source /vast/palmer/apps/avx2/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate llada_env

echo "🔥 Trying to import torch..."
python -c "import torch; print('Torch version:', torch.__version__)"

echo "🚀 Launching Gradio app"
/home/alm293/.conda/envs/llada_env/bin/python ~/llada_lora_scheduler/LLaDA/train_llada_lora.py
