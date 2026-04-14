#!/bin/bash
#SBATCH --job-name=dense-retriever
#SBATCH -p nvidia                      # GPU partition (nvidia on this cluster)
#SBATCH -A <your-account>              # e.g. cities
#SBATCH --qos=nvidia                   # QOS matching the partition
#SBATCH --gres=gpu:a100:1             # request one A100
#SBATCH --constrain=80g               # 80GB variant
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH -o dense_retriever_%j.out
#SBATCH -e dense_retriever_%j.err

# Repo lives on scratch — faster I/O than home
cd /scratch/<your-username>/issue-triage-copilot

# Use full python path to bypass conda activate in non-interactive shells.
# Env: torch-gpu at /home/<your-username>/.conda/envs/torch-gpu
# (has torch 2.7+cu126, transformers, pandas, numpy)
export PYTHON=/home/<your-username>/.conda/envs/torch-gpu/bin/python

# Point HF model cache to scratch to avoid filling home quota
# Model: BAAI/bge-small-en-v1.5 (~90 MB) — downloaded once and cached
export HF_HOME=/scratch/<your-username>/hf

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

$PYTHON -m src.retrieval.dense_retriever

echo "Index build done: $(date)"
