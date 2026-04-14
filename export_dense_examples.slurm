#!/bin/bash
#SBATCH --job-name=dense-examples
#SBATCH -p nvidia                      # GPU partition (nvidia on this cluster)
#SBATCH -A <your-account>              # e.g. cities
#SBATCH --qos=nvidia                   # QOS matching the partition
#SBATCH --gres=gpu:a100:1             # request one A100
#SBATCH --constrain=80g               # 80GB variant
#SBATCH -c 4
#SBATCH -t 0:15:00
#SBATCH -o export_dense_examples_%j.out
#SBATCH -e export_dense_examples_%j.err

cd /scratch/<your-username>/issue-triage-copilot

# Use full python path — torch-gpu env at /home/<your-username>/.conda/envs/torch-gpu
export PYTHON=/home/<your-username>/.conda/envs/torch-gpu/bin/python
export HF_HOME=/scratch/<your-username>/hf

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"

$PYTHON -m src.retrieval.export_dense_examples

echo "Done: $(date)"
