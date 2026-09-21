#!/bin/bash
#SBATCH --job-name=ews_eval_fixup
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -o pipefail

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODELS=(drcif tde pf cif grsf tsf tsbf lps fastshapelet inceptiontime)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL | $DATASET | pangaea (fixup re-eval)"
echo "Node: $(hostname)"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"
mkdir -p logs test_result

python -u testing/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --target pangaea \
    --config config.yaml --force

python -u testing/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --target zenodo \
    --config config.yaml --force
