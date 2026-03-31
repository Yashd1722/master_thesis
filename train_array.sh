#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ts_train_array
#SBATCH --array=0-3
#SBATCH --output=/home/%u/Master_thesis/Main/logs/slurm_%A_%a.out
#SBATCH --error=/home/%u/Master_thesis/Main/logs/slurm_%A_%a.err

set -e

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

ROOT_DIR="/home/$USER/Master_thesis"
if [ -d "$ROOT_DIR/master_thesis" ]; then
    PROJECT_DIR="$ROOT_DIR/master_thesis"
else
    PROJECT_DIR="$ROOT_DIR/Main"
fi

cd "$PROJECT_DIR"

mkdir -p logs results checkpoints

echo "Activating virtual environment..."
source /home/$USER/Master_thesis/myenv/bin/activate

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo ""

MODEL=lstm
DATASET=ts_500
METRIC=f1_macro
COMMON_ARGS="--model $MODEL --dataset $DATASET --metric $METRIC --epochs 20 --early_stop --patience 5 --seq_len 500 --data_seed 42"

case $SLURM_ARRAY_TASK_ID in
    0)
        EXP_NAME="baseline"
        EXTRA_ARGS=""
        ;;
    1)
        EXP_NAME="trend"
        EXTRA_ARGS="--trend"
        ;;
    2)
        EXP_NAME="seasonality"
        EXTRA_ARGS="--seasonality"
        ;;
    3)
        EXP_NAME="trend_seasonality"
        EXTRA_ARGS="--trend --seasonality"
        ;;
    *)
        echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Running experiment: $EXP_NAME"
echo "Command: python3 training/train.py $COMMON_ARGS $EXTRA_ARGS"

python3 training/train.py $COMMON_ARGS $EXTRA_ARGS

echo ""
echo "========================================="
echo "Finished experiment: $EXP_NAME"
echo "Job finished at: $(date)"
echo "========================================="
