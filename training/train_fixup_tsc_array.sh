#!/bin/bash
#SBATCH --job-name=ews_tsc_fixup
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -o pipefail

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODELS=(drcif tde pf cif grsf tsf tsbf lps fastshapelet)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"
mkdir -p logs results checkpoints

python -u training/train.py --model "$MODEL" --dataset "$DATASET" --force
