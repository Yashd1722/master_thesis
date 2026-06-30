#!/bin/bash
# Evaluate all models on all datasets and targets, then generate figures.
# Run from the repo root: bash testing/run_all_evaluations.sh

source ~/Master_thesis/myenv/bin/activate
cd ~/Master_thesis/master_thesis || exit 1

# Current model roster (Phase 2 pruned + Phase 2 additions).
# TSC: 8 models  DL: 3 models
TSC_MODELS=(minirocket multirocket arsenal drcif rocket hydra_multirocket rdst weasel2)
DL_MODELS=(cnn_lstm lstm inceptiontime)
ALL_MODELS=("${TSC_MODELS[@]}" "${DL_MODELS[@]}")

DATASETS=(ts_500 ts_1500)
TARGETS=(zenodo pangaea)

for model in "${ALL_MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for target in "${TARGETS[@]}"; do
      echo "=================================================="
      echo ">>> Evaluating: $model | $dataset | $target"
      echo "=================================================="
      python testing/evaluate.py --model "$model" --dataset "$dataset" --target "$target"
      EXIT_CODE=$?
      if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: $model/$dataset/$target failed (checkpoint missing or error)"
      fi
    done
  done
done

echo ""
echo "All evaluations done. Generating summary figures..."
python testing/plot_figures.py --config config.yaml
echo "Figures saved to results/comparison/"
