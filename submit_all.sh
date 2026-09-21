#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ENV="$HOME/Master_thesis/myenv/bin/activate"

DL=$(sbatch --parsable training/train_dl_array.sh)
TSC=$(sbatch --parsable training/train_tsc_array.sh)
echo "train DL  array : $DL"
echo "train TSC array : $TSC"

ZEN=$(sbatch --parsable --dependency=afterany:"$DL":"$TSC" testing/eval_zenodo_array.sh)
PAN=$(sbatch --parsable --dependency=afterany:"$DL":"$TSC" testing/eval_pangaea_array.sh)
echo "eval zenodo  array : $ZEN"
echo "eval pangaea array : $PAN"

AGG=$(sbatch --parsable --partition=small_cpu --time=00:30:00 --mem=8G \
      --job-name=ews_aggregate \
      --output=logs/%x_%j.out --error=logs/%x_%j.err \
      --dependency=afterany:"$ZEN":"$PAN" \
      --wrap "cd $PWD && source $ENV && python testing/collect_results.py && python testing/plot_figures.py --config config.yaml")
echo "aggregate job   : $AGG"

echo
echo "watch:  squeue -u \$USER"
echo "after:  cat results/summary/coverage.csv"
