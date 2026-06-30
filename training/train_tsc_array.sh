#!/bin/bash
#SBATCH --job-name=ews_tsc_train
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --array=0-15
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Set ALL thread-pool sizes equal to SLURM_CPUS_PER_TASK BEFORE Python starts.
# numba, OpenMP, and MKL each read their var at import time, so this must be
# done here in the shell — not inside train.py.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# TSC model roster (Phase 2 pruned: 8 models x 2 datasets = 16 array tasks).
# Array index = model_idx * 2 + dataset_idx
MODELS=(
  minirocket   minirocket
  multirocket  multirocket
  arsenal      arsenal
  drcif        drcif
  rocket       rocket
  hydra_multirocket hydra_multirocket
  rdst         rdst
  weasel2      weasel2
)

DATASETS=(
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK  NUMBA/OMP/MKL: $NUMBA_NUM_THREADS"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"

mkdir -p logs

# -u: unbuffered stdout so logs stream in real time
/usr/bin/time -v python -u training/train.py --model "$MODEL" --dataset "$DATASET" \
  2>&1 | tee "logs/${MODEL}_${DATASET}_train.log"
