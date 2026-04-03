#!/bin/bash
#SBATCH --job-name=issue-triage-encoder
#SBATCH -p <your-gpu-partition>
#SBATCH -A <your-account>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -o encoder_%j.out
#SBATCH -e encoder_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>

cd <path-to-repo>

# Activate your conda environment
source <path-to-conda>/etc/profile.d/conda.sh
conda activate <your-env>

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"

python -m src.models.train_encoder \
    --model_name distilbert-base-uncased \
    --epochs 6 \
    --batch_size 64 \
    --lr 5e-6 \
    --warmup_pct 0.10 \
    --max_length 256

echo "Training done: $(date)"

python -m src.models.evaluate_encoder

echo "Evaluation done: $(date)"
