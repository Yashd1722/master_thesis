#!/bin/bash
#SBATCH --job-name=ews_dl_test
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/s466553/Master_thesis/master_thesis/logs/dl_test_%A_%a.out
#SBATCH --error=/home/s466553/Master_thesis/master_thesis/logs/dl_test_%A_%a.err

MODELS=(cnn_lstm cnn_lstm lstm lstm cnn cnn multihead_cnn multihead_cnn)
DATASETS=(ts_500 ts_1500 ts_500 ts_1500 ts_500 ts_1500 ts_500 ts_1500)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL on $DATASET"

cd /home/s466553/Master_thesis/master_thesis
source /home/s466553/Master_thesis/myenv/bin/activate

python testing/evaluate.py --model "$MODEL" --dataset "$DATASET" --target zenodo
python testing/evaluate.py --model "$MODEL" --dataset "$DATASET" --target pangaea
