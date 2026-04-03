#!/bin/bash
#SBATCH --job-name=triage-sweep
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -A YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -o sweep_%j.out
#SBATCH -e sweep_%j.err

cd PROJECT_ROOT

source CONDA_INIT_SCRIPT
conda activate YOUR_ENV

echo "Started: $(date)"
python -m src.models.sweep_threshold
echo "Done: $(date)"
