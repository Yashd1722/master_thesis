#!/bin/bash
#SBATCH --job-name=ews_tsc_train
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-43
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -o pipefail

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODELS=(minirocket multirocket arsenal rdst weasel2 drcif rocket \
        tsf st ls boss \
        bop saxvsm tsbf lps fastshapelet catch22 tde pf \
        cif mrsqm grsf)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"
mkdir -p logs

/usr/bin/time -v python -u training/train.py --model "$MODEL" --dataset "$DATASET" --force \
  2>&1 | tee "logs/${MODEL}_${DATASET}_train.log"
