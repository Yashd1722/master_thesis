#!/bin/bash
#SBATCH --job-name=ews_multihead
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s466553/Master_thesis/master_thesis/logs/slurm_train_multihead_cnn_%j.out
#SBATCH --error=/home/s466553/Master_thesis/master_thesis/logs/slurm_train_multihead_cnn_%j.err

source ~/Master_thesis/myenv/bin/activate
cd ~/Master_thesis/master_thesis

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $CUDA_VISIBLE_DEVICES"
echo "Start     : $(date)"
echo "=============================="

python training/train.py --model multihead_cnn --mode bury --config config.yaml

echo "Done: $(date) | exit=$?"
