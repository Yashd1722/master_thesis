#!/bin/bash
#SBATCH --job-name=ews_eval_zenodo
#SBATCH --partition=large_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2:00:00
#SBATCH --array=0-57
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -o pipefail

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODELS=(cnn_lstm lstm inceptiontime patchtst \
        minirocket multirocket arsenal rdst weasel2 drcif rocket \
        tsf st ls boss \
        bop saxvsm tsbf lps fastshapelet catch22 tde pf \
        resnet tcn rnn_fcn \
        cif mrsqm grsf)
DATASETS=(ts_500 ts_1500)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL | $DATASET | zenodo"
echo "Node: $(hostname)"

cd "$HOME/Master_thesis/master_thesis" || exit 1
source "$HOME/Master_thesis/myenv/bin/activate"
mkdir -p logs test_result

python -u testing/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --target zenodo \
    --config config.yaml ${EVAL_FORCE:-}
