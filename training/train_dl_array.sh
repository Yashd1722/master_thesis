#!/bin/bash
#SBATCH --job-name=ews_dl_train
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -o pipefail

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODELS=(cnn_lstm lstm inceptiontime patchtst resnet tcn rnn_fcn)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK  GPU: $CUDA_VISIBLE_DEVICES"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"
mkdir -p logs

python -u training/train.py --model "$MODEL" --dataset "$DATASET" --force \
  2>&1 | tee "logs/${MODEL}_${DATASET}_train.log"
