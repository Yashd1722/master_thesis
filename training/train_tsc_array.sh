#!/bin/bash
#SBATCH --job-name=ews_tsc_train
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-25
#SBATCH --output=/home/s466553/Master_thesis/master_thesis/logs/tsc_train_%A_%a.out
#SBATCH --error=/home/s466553/Master_thesis/master_thesis/logs/tsc_train_%A_%a.err

# Fix numba thread pool BEFORE Python starts — must match n_jobs in model files
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

MODELS=(
  rocket       rocket
  minirocket   minirocket
  multirocket  multirocket
  arsenal      arsenal
  knn_dtw      knn_dtw
  boss         boss
  weasel       weasel
  shapelet     shapelet
  proximity_forest proximity_forest
  ts_chief     ts_chief
  drcif        drcif
  tde          tde
  hivecote     hivecote
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
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
  ts_500 ts_1500
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"
echo "Node: $(hostname)  NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS"

cd /home/s466553/Master_thesis/master_thesis
source /home/s466553/Master_thesis/myenv/bin/activate

mkdir -p /home/s466553/Master_thesis/master_thesis/logs
python training/train.py --model "$MODEL" --dataset "$DATASET"
