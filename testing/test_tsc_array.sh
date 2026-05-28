#!/bin/bash
#SBATCH --job-name=ews_tsc_test
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-25
#SBATCH --output=/home/s466553/Master_thesis/master_thesis/logs/tsc_test_%A_%a.out
#SBATCH --error=/home/s466553/Master_thesis/master_thesis/logs/tsc_test_%A_%a.err

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

cd /home/s466553/Master_thesis/master_thesis
source /home/s466553/Master_thesis/myenv/bin/activate

python testing/evaluate.py --model "$MODEL" --dataset "$DATASET" --target zenodo
python testing/evaluate.py --model "$MODEL" --dataset "$DATASET" --target pangaea
