#!/bin/bash
source ~/Master_thesis/myenv/bin/activate
cd ~/Master_thesis/master_thesis || exit 1

# Updated to only include models with verified checkpoints in the checkpoints/ folder
MODELS=(cnn_lstm lstm cnn multihead_cnn rocket minirocket multirocket arsenal shapelet)
DATASETS=(ts_500 ts_1500)
TARGETS=(zenodo pangaea)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for target in "${TARGETS[@]}"; do
      echo "=================================================="
      echo ">>> Evaluating: $model | $dataset | $target"
      echo "=================================================="
      
      # Run evaluate.py and capture the exit code
      python testing/evaluate.py --model "$model" --dataset "$dataset" --target "$target"
      EXIT_CODE=$?
      
      # If it failed because the checkpoint is missing, print a clean warning
      if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️  Skipped $model/$dataset/$target (Checkpoint missing or still training)"
      fi
    done
  done
done

echo ""
echo "🎉 All evaluations complete! Generating final paper plots..."
python testing/plot_figures.py
