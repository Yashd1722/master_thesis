#!/bin/bash
# EWS Thesis Live Dashboard
# Auto-refreshes every 5 seconds to show the status of all models.

watch -n 5 -c '
echo "======================================================================"
echo "  EWS THESIS TRAINING DASHBOARD  |  $(date +%H:%M:%S)"
echo "======================================================================"

echo ">>> ACTIVE PYTHON TRAINING LOGS (Last 6 lines per model):"
for log in logs/*_train.log; do
    if [ -f "$log" ]; then
        # Extract clean name (e.g., "minirocket_ts_500")
        name=$(basename "$log" .log | sed "s/_train//")
        echo -e "\n\033[1;34m--- $name ---\033[0m"
        tail -n 6 "$log" | sed "s/^/  /"
    fi
done

echo -e "\n\033[1;31m>>> RECENT SLURM ERRORS (OOM Kills / Aeon Crashes):\033[0m"
# Check the 10 most recently modified .err files
for err in $(ls -t logs/tsc_train_*.err logs/dl_train_*.err 2>/dev/null | head -n 10); do
    if [ -s "$err" ]; then
        echo -e "\n\033[1;33m--- $(basename $err) ---\033[0m"
        tail -n 5 "$err" | sed "s/^/  /"
    fi
done
'
