#!/bin/bash
#SBATCH --job-name=triage-ablation
#SBATCH -p <your-gpu-partition>
#SBATCH -A <your-account>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -o ablation_%j.out
#SBATCH -e ablation_%j.err

cd <path-to-repo>

source <path-to-conda>/etc/profile.d/conda.sh
conda activate <your-env>

echo "Started: $(date)"
python -m src.models.ablation_title_only
echo "Done: $(date)"
