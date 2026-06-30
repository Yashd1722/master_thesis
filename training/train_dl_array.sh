#!/bin/bash
#SBATCH --job-name=ews_dl_train
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Match thread counts to allocated CPUs (h100 partition: 4 CPUs).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# DL model roster: 3 models x 2 datasets = 6 array tasks.
MODELS=(cnn_lstm cnn_lstm lstm lstm inceptiontime inceptiontime)
DATASETS=(ts_500 ts_1500 ts_500 ts_1500 ts_500 ts_1500)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK  GPU: $CUDA_VISIBLE_DEVICES"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"

mkdir -p logs

python -u training/train.py --model "$MODEL" --dataset "$DATASET" \
  2>&1 | tee "logs/${MODEL}_${DATASET}_train.log"
