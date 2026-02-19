#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=lstm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Create logs directory if it doesn't exist
mkdir -p ~/Master_thesis/Main/logs

# Activate virtual environment
echo "Activating virtual environment..."
source ~/Master_thesis/myenv/bin/activate

# Navigate to project directory
cd ~/Master_thesis/Main

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training
echo "Starting training..."
python3 training/train.py \
    --dataset ts_500 \
    --model lstm \
    --metric f1_macro \
    --epochs 20 \
    --early_stop \
    --patience 5

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="
