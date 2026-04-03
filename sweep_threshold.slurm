#!/bin/bash
#SBATCH --job-name=triage-sweep
#SBATCH -p <your-gpu-partition>
#SBATCH -A <your-account>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -o sweep_%j.out
#SBATCH -e sweep_%j.err

cd <path-to-repo>

source <path-to-conda>/etc/profile.d/conda.sh
conda activate <your-env>

echo "Started: $(date)"
python -m src.models.sweep_threshold
echo "Done: $(date)"
