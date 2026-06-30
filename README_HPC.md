# HPC Guide — University of Würzburg (julia2)

Practical reference for running jobs on the julia2 cluster.
Update paths/partitions if you move to a different cluster.

---

## Environment Setup (run once)

```bash
# On the login node — create virtual env in your home directory
python3 -m venv ~/Master_thesis/myenv
source ~/Master_thesis/myenv/bin/activate

pip install --upgrade pip
pip install -r ~/Master_thesis/master_thesis/requirements.txt
pip install aeon==1.4.0    # TSC models — may take 5–10 minutes
```

The SLURM scripts activate `~/Master_thesis/myenv/bin/activate` automatically.

---

## Partition Summary

| Partition | Use case | CPUs | RAM | GPU | Max walltime |
|---|---|---|---|---|---|
| `large_cpu` | TSC training | 16 | 60 GB | — | configurable |
| `h100` | DL training | 4 | 32 GB | 1 × H100 | configurable |

Submit from the **login node** with `sbatch`. Do NOT run heavy jobs directly on the
login node.

---

## TSC Training (`train_tsc_array.sh`)

```bash
sbatch training/train_tsc_array.sh
```

- **Array:** `--array=0-15` (8 models × 2 datasets = 16 tasks)
- **Partition:** `large_cpu`, 16 CPUs, 60 GB, 4 h walltime
- **Thread control:** `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` all
  set to `$SLURM_CPUS_PER_TASK` (= 16) in the script header — do NOT change inside
  Python code. Numba reads this var at import time; setting it later has no effect.

Task-to-model mapping:

| Task ID | Model | Dataset |
|---|---|---|
| 0 | minirocket | ts_500 |
| 1 | minirocket | ts_1500 |
| 2 | multirocket | ts_500 |
| 3 | multirocket | ts_1500 |
| 4 | arsenal | ts_500 |
| 5 | arsenal | ts_1500 |
| 6 | drcif | ts_500 |
| 7 | drcif | ts_1500 |
| 8 | rocket | ts_500 |
| 9 | rocket | ts_1500 |
| 10 | hydra_multirocket | ts_500 |
| 11 | hydra_multirocket | ts_1500 |
| 12 | rdst | ts_500 |
| 13 | rdst | ts_1500 |
| 14 | weasel2 | ts_500 |
| 15 | weasel2 | ts_1500 |

**To retrain a single model only** (without cancelling the whole array):

```bash
sbatch --array=0 training/train_tsc_array.sh   # minirocket / ts_500 only
```

---

## DL Training (`train_dl_array.sh`)

```bash
sbatch training/train_dl_array.sh
```

- **Array:** `--array=0-5` (3 models × 2 datasets = 6 tasks)
- **Partition:** `h100`, 1 GPU, 4 CPUs, 32 GB, 24 h walltime

| Task ID | Model | Dataset |
|---|---|---|
| 0 | cnn_lstm | ts_500 |
| 1 | cnn_lstm | ts_1500 |
| 2 | lstm | ts_500 |
| 3 | lstm | ts_1500 |
| 4 | inceptiontime | ts_500 |
| 5 | inceptiontime | ts_1500 |

---

## Monitoring Jobs

```bash
squeue -u $USER                         # list your running/pending jobs
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
scancel <JOBID>                         # cancel a job
scontrol show job <JOBID>               # inspect a job (node, reason for pending, etc.)
```

**Log files** are written to the `logs/` directory:
- SLURM stdout/stderr: `logs/ews_tsc_train_{ARRAY_JOB_ID}_{TASK_ID}.out/.err`
- Python training log (tee'd): `logs/{model}_{dataset}_train.log`

To tail a live log:

```bash
tail -f logs/minirocket_ts_500_train.log
```

---

## Troubleshooting

### Job pending: `(Resources)` or `(Priority)`

Normal. Jobs wait in queue until resources are free. If pending for >12 h,
check if the partition is overloaded:

```bash
sinfo -p large_cpu
```

### Job fails immediately: `ModuleNotFoundError`

The virtualenv is not activated or packages are missing. Check:

```bash
source ~/Master_thesis/myenv/bin/activate
python -c "import aeon; print(aeon.__version__)"
```

### Job killed: `OOM kill` or `Exceeded job memory limit`

Reduce `MAX_TRAIN_SAMPLES` in `models/__init__.py` for the failing model,
or increase `--mem` in the SLURM script (requires checking available RAM on
`large_cpu` nodes — use `sinfo -p large_cpu -o "%n %m %c"` to see node specs).

### Numba thread count too high

If you see a warning like `numba: Requested N threads, but only M available`,
the `NUMBA_NUM_THREADS` export in the SLURM header is not being respected.
Ensure no other code sets `NUMBA_NUM_THREADS` after startup.

### WEASEL v2 Unicode error on some nodes

WEASEL v2 uses dictionary compression. If you see a Unicode/encoding error,
add `export PYTHONIOENCODING=utf-8` to the SLURM script header.

---

## Useful SLURM Snippets

```bash
# Submit with a dependency (run evaluation only after training finishes)
TRAIN_JOB=$(sbatch --parsable training/train_tsc_array.sh)
sbatch --dependency=afterok:$TRAIN_JOB testing/run_all_evaluations.sh

# Check GPU allocation on h100 nodes
srun -p h100 --gres=gpu:1 --pty nvidia-smi

# Interactive session for debugging (CPU)
srun -p large_cpu --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash

# Interactive session for debugging (GPU)
srun -p h100 --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=1:00:00 --pty bash
```

---

## Checkpoint Storage

Checkpoints are stored in `~/Master_thesis/master_thesis/checkpoints/` and
are **gitignored**. Back up any validated checkpoints to a persistent location
before the cluster auto-cleans scratch space.

```bash
# Example: copy all checkpoints to long-term storage (update path)
rsync -avz checkpoints/ /path/to/long_term_storage/checkpoints/
```
