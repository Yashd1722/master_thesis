# Master Thesis: Early Warning Signals in Time Series

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep learning framework for detecting early warning signals in time series data, with a focus on critical transitions and tipping points.

---

## 📁 Project Structure

```
master_thesis/
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
│
├── dataset/                       # Datasets (not tracked in git)
│   ├── ts_500/                   # 500-timestep dataset
│   │   ├── ts_500_train.csv
│   │   ├── ts_500_val.csv
│   │   └── ts_500_test.csv
│   ├── ts_1500/                  # 1500-timestep dataset
│   │   ├── ts_1500_train.csv
│   │   ├── ts_1500_val.csv
│   │   └── ts_1500_test.csv
│   └── pangaea_923197/           # Real-world paleoclimate data
│       └── datasets/
│           └── clean_dataset/
│
├── src/                           # Source code
│   ├── dataset_builder.py        # Build datasets from raw files
│   ├── dataset_loader.py         # Load and preprocess datasets
│   ├── pangea_cleaner.py         # Clean PANGAEA data
│   ├── splits.py                 # Train/val/test splitting utilities
│   └── quick_check.py            # Quick dataset verification tool
│
├── training/                      # Training scripts (on HPC)
│   ├── train.py                  # Main training script
│
│
│
├── logs/                          # Training logs (SLURM output)
│   ├── train_*.out               # Standard output logs
│   └── train_*.err               # Error logs
│
├── checkpoints/                   # Saved model checkpoints
│   └── *.pth                     # PyTorch model files
│
├── metrics/                       # Training metrics and results
│   └── *.json                     # Metric logs
|
├── Models/                         # Saved model Architecture
│   └── CNN.py                     # Different Neural Network Arch.
|
|
│
├── results/                       # Analysis results
│   └── *.csv                     # Result summaries
│
├── train.sh                      # SLURM script for training
```

---

## 🚀 Quick Start

### **Prerequisites**

- Python 3.8+
- PyTorch
- NumPy, Pandas, scikit-learn
- Access to HPC cluster (for GPU training)

### **Installation**

```bash
# Clone the repository
cd ~/Master_thesis/Main

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install torch numpy pandas scikit-learn tqdm matplotlib
```

---

## 📊 Datasets

### **1. ts_500 (Synthetic - 500 timesteps)**

- **Sequences**: ~350 time series
- **Features**: x, Residuals
- **Classes**: Binary or multi-class labels
- **Use case**: Quick testing and prototyping

### **2. ts_1500 (Synthetic - 1500 timesteps)**

- **Sequences**: ~350 time series
- **Features**: x, Residuals
- **Classes**: multi-class labels
- **Use case**: Longer temporal dependencies

### **3. pangaea_923197 (Real paleoclimate data)**

- **Sequences**: 6 core samples
- **Features**: Al, Ba, Mo, Ti, U (mg/kg)
- **Classes**: None (unsupervised)
- **Use case**: Real-world early warning signal detection

---

## 🛠️ Usage

### **1. Verify Dataset**

Check if dataset loads correctly:

```bash
python3 src/quick_check.py --dataset ts_500
python3 src/quick_check.py --dataset ts_1500
python3 src/quick_check.py --dataset pangaea_923197
```

**Options:**

- `--dataset`: Dataset to load (ts_500, ts_1500, pangaea_923197)
- `--train`: Training split ratio (default: 0.7)
- `--val`: Validation split ratio (default: 0.15)
- `--test`: Test split ratio (default: 0.15)
- `--seed`: Random seed (default: 42)

### **2. Local Training (CPU)**

For quick testing on your local machine or HPC login node:

```bash
# Activate environment
source myenv/bin/activate

# Run training (replace with actual training command based on your train.py)
python3 training/train.py \
    --dataset ts_500 \
    --model lstm \
    --metric f1_macro \
    --epochs 20 \
    --early_stop \
    --patience 5
```

### **3. HPC GPU Training**

#### **Option A: Single Job**

```bash
# Submit a single training job
sbatch train_ts_500.sh

# Check job status
squeue -u $USER

# Monitor logs in real-time
tail -f logs/train_*.out
```

# Check status

squeue -u $USER

````

#### **Option B: Submit All Jobs**

Submit individual jobs for each dataset:

```bash
# Make script executable
chmod +x submit_all.sh

# Submit all jobs
./submit_all.sh
````

---

## 📝 SLURM Script Configuration

Each training script follows this structure:

```bash
#!/bin/bash
#SBATCH --partition=h100              # GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=02:00:00               # Max runtime (2 hours)
#SBATCH --mem=16G                     # Memory request
#SBATCH --cpus-per-task=4             # CPU cores
#SBATCH --job-name=lstm_train         # Job name
#SBATCH --output=logs/train_%j.out    # Output log
#SBATCH --error=logs/train_%j.err     # Error log
```

### **Available Partitions**

- `h100` - H100 GPUs (fastest, 1-day limit)
- `gpu_computervision` - Computer vision GPUs (2-day limit)
- `gpu_computervision_long` - Extended runs (4-day limit)
- `standard` - Standard CPUs (no GPU)

---

## 🔍 Monitoring Jobs

### **Check Job Queue**

```bash
squeue -u $USER
```

### **View Real-time Logs**

```bash
tail -f logs/train_JOBID.out
```

### **Check Job Details**

```bash
scontrol show job JOBID
```

### **Cancel Job**

```bash
scancel JOBID
```

### **View All Available GPUs**

```bash
sinfo -p h100,gpu_computervision
```

---

## 📈 Results

After training completes, check:

**Logs:**

```bash
ls -lh logs/
cat logs/train_JOBID.out
```

**Models:**

```bash
ls -lh models/
```

**Metrics:**

```bash
ls -lh metrics/
cat metrics/*.json
```

---

## 🔧 Customizing Training

### **Modify Training Parameters**

Edit the `.sh` script or pass arguments:

```bash
python3 training/train.py \
    --dataset ts_1500 \
    --model gru \              # Change model architecture
    --epochs 50 \              # More epochs
    --batch_size 32 \          # Batch size
    --lr 0.001 \               # Learning rate
    --patience 10              # Early stopping patience
```

### **Available Models**

- `lstm` - Long Short-Term Memory
- `gru` - Gated Recurrent Unit
- `cnn_lstm` - Hybrid CNN-LSTM

---

## 🐛 Troubleshooting

### **Import Errors**

```bash
# Make sure you're in the right directory
cd ~/Master_thesis/Main

# Activate virtual environment
source myenv/bin/activate

# Check Python path
python3 -c "import sys; print(sys.path)"
```

### **Dataset Not Found**

```bash
# Verify dataset exists
ls -lh dataset/ts_500/

# Check file paths in dataset_loader.py
```

### **GPU Not Available**

```bash
# Check GPU availability
nvidia-smi

# Note: GPUs only available on compute nodes, not login nodes
# Must submit job via sbatch or srun
```

### **Job Stuck in Queue**

```bash
# Check why job is pending
squeue -u $USER --start

# Try different partition
sbatch --partition=standard train.sh
```

---

## 📄 File Descriptions

| File                 | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `dataset_loader.py`  | Universal loader for ts_500, ts_1500, pangaea_923197 |
| `dataset_builder.py` | Build datasets from raw simulation outputs           |
| `pangea_cleaner.py`  | Preprocess PANGAEA paleoclimate data                 |
| `splits.py`          | Create train/val/test splits                         |
| `quick_check.py`     | Verify dataset loading and splits                    |
| `train_*.sh`         | SLURM batch scripts for HPC training                 |
| `train.sh`           | Convenience script to submit jobs                    |

---

---

## 📧 Contact

For questions or issues, contact: [yashkumar-sanjaybhai.dhameliya@stud-mail.uni-wuerzburg.de]

---

## 🔄 Git Workflow

### **Push Changes**

```bash
# Check status
git status

# Add files (excluding large datasets)
git add src/ logs/ metrics/ models/ *.sh

# Commit
git commit -m "Add training scripts and results"

# Push
git push origin main
```

### **Pull Latest Changes**

```bash
git pull origin main
```

---

## 📊 Example Workflow

```bash
# 1. Check dataset
python3 src/quick_check.py --dataset ts_500

# 2. Submit training job
sbatch train.sh

# 3. Monitor progress
watch -n 10 'squeue -u $USER'

# 4. View results when complete
cat logs/train_*.out

# 5. Analyze metrics
python3 -c "import json; print(json.load(open('metrics/results.json')))"

# 6. Push to git
git add logs/ metrics/ models/
git commit -m "Training results for ts_500"
git push
```

---

**Last Updated:** January 2026  
**Status:** Active Development
