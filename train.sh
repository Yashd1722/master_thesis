#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ts_train
#SBATCH --output=/home/%u/Master_thesis/Main/logs/slurm_%j.out
#SBATCH --error=/home/%u/Master_thesis/Main/logs/slurm_%j.err

set -e

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Go to project directory
cd /home/$USER/Master_thesis/Main

# Create logs directory
mkdir -p logs results checkpoints

# Activate virtual environment
echo "Activating virtual environment..."
source /home/$USER/Master_thesis/myenv/bin/activate

# GPU check
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo ""

echo "========================================="
echo "Starting experiments"
echo "========================================="

MODEL=lstm
DATASET=ts_500
METRIC=f1_macro

# -----------------------------------------
# 1️⃣ BASELINE (no trend, no seasonality)
# -----------------------------------------
echo ""
echo "Running BASELINE experiment"
echo ""

python3 training/train.py \
  --model $MODEL \
  --dataset $DATASET \
  --metric $METRIC \
  --epochs 20 \
  --early_stop \
  --patience 5 \
  --seq_len 500 \
  --data_seed 42


# -----------------------------------------
# 2️⃣ TREND
# -----------------------------------------
echo ""
echo "Running TREND experiment"
echo ""

python3 training/train.py \
  --model $MODEL \
  --dataset $DATASET \
  --metric $METRIC \
  --epochs 20 \
  --early_stop \
  --patience 5 \
  --seq_len 500 \
  --data_seed 42 \
  --trend


# -----------------------------------------
# 3️⃣ SEASONALITY
# -----------------------------------------
echo ""
echo "Running SEASONALITY experiment"
echo ""

python3 training/train.py \
  --model $MODEL \
  --dataset $DATASET \
  --metric $METRIC \
  --epochs 20 \
  --early_stop \
  --patience 5 \
  --seq_len 500 \
  --data_seed 42 \
  --seasonality


# -----------------------------------------
# 4️⃣ TREND + SEASONALITY
# -----------------------------------------
echo ""
echo "Running TREND + SEASONALITY experiment"
echo ""

python3 training/train.py \
  --model $MODEL \
  --dataset $DATASET \
  --metric $METRIC \
  --epochs 20 \
  --early_stop \
  --patience 5 \
  --seq_len 500 \
  --data_seed 42 \
  --trend \
  --seasonality


echo ""
echo "========================================="
echo "All experiments finished"
echo "Job finished at: $(date)"
echo "========================================="
