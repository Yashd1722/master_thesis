#!/bin/bash
#SBATCH --job-name=ews_dl_fixup
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -o pipefail

DATASETS=(ts_500 ts_1500)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: inceptiontime on $DATASET"
echo "Node: $(hostname)"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"
mkdir -p logs results checkpoints

python -u training/train.py --model inceptiontime --dataset "$DATASET" --force
