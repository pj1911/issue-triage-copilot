#!/bin/bash
#SBATCH --job-name=triage-ablation
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -o ablation_%j.out
#SBATCH -e ablation_%j.err

cd PROJECT_ROOT

source CONDA_INIT_SCRIPT
conda activate YOUR_ENV

echo "Started: $(date)"
python -m src.models.ablation_title_only
echo "Done: $(date)"
