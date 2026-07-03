#!/bin/bash
#SBATCH --job-name=ews_tsc_train
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 7 TSC models x 2 datasets = 14 tasks (0-13)
# task_id = model_idx * 2 + dataset_idx
MODELS=(minirocket multirocket arsenal drcif rocket rdst weasel2)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID / 2]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID % 2]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"

mkdir -p logs

/usr/bin/time -v python -u training/train.py --model "$MODEL" --dataset "$DATASET" \
  2>&1 | tee "logs/${MODEL}_${DATASET}_train.log"
