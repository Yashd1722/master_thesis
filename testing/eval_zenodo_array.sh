#!/bin/bash
#SBATCH --job-name=ews_eval_zenodo
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# 10 models x 2 datasets = 20 tasks (0-19)
# task_id = model_idx * 2 + dataset_idx
MODELS=(minirocket multirocket arsenal drcif rocket rdst weasel2 cnn_lstm lstm inceptiontime)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID / 2]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID % 2]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL | $DATASET | zenodo"
echo "Node: $(hostname)"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"

mkdir -p logs test_result

# Match numba/OMP thread counts to allocated CPUs.
# Without this, MiniRocket's set_num_threads(n_jobs_from_training) can exceed
# the numba pool limit and crash.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Already-done tasks are skipped automatically (no --force).
# Add --force to re-evaluate from scratch.
python -u testing/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --target zenodo \
    --config config.yaml
