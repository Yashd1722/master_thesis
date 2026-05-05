#!/bin/bash
#SBATCH --job-name=ews_test_multihead
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s466553/Master_thesis/master_thesis/inference_logs/slurm_test_multihead_%j.out
#SBATCH --error=/home/s466553/Master_thesis/master_thesis/inference_logs/slurm_test_multihead_%j.err

source ~/Master_thesis/myenv/bin/activate
cd ~/Master_thesis/master_thesis

echo "=============================================="
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Model    : multihead_cnn"
echo "  Node     : $SLURMD_NODENAME"
echo "  GPU      : $CUDA_VISIBLE_DEVICES"
echo "  Start    : $(date)"
echo "=============================================="

mkdir -p results test_results test_logs inference_logs metrics

echo ">>> Step 1/3: Inference on PANGAEA cores ..."
python testing/evaluate.py --model multihead_cnn --dataset ts_500 --target pangaea --config config.yaml

echo ">>> Step 2/3: Computing ROC/AUC metrics ..."
python testing/compute_metrics.py --model multihead_cnn --dataset ts_500 --config config.yaml

echo ">>> Step 3/3: Generating figures ..."
python testing/plot_figures.py --model multihead_cnn --dataset ts_500 --config config.yaml

echo "=============================================="
echo "  Done: $(date) | exit=$?"
echo "=============================================="
