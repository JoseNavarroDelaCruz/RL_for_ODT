#!/bin/bash
#SBATCH --job-name=grpo_training
#SBATCH --output=/home/n/navarrodelacruz/two_stage_neural_diving/logs/grpo_%j.out
#SBATCH --error=/home/n/navarrodelacruz/two_stage_neural_diving/logs/grpo_%j.err
#SBATCH -p IMSE
#SBATCH --nodelist GPU51
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --mail-user=navarrodelacruz@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,ALL

# ============================================================================
# Run script for GRPO training with E2E accuracy loss
# Uses: two_stage_gps_small.py + hpc_train_grpo_small.py
#
# Key GRPO differences from supervised-only (run_small.sh):
#   LOSS_METHOD=e2e_accuracy → Policy gradient based on end-to-end accuracy
#   GRPO hyperparams (num_candidates, temperature, clip_epsilon, e2e_weight)
#   are set in main() of hpc_train_grpo_small.py — edit there to tune them.
#
# Available LOSS_METHOD options:
#   - e2e_accuracy: End-to-end accuracy-based policy gradient (DEFAULT for GRPO)
#   - supervised_only: Pure supervised learning (cross-entropy only, no GRPO)
#   - log_prob: Original GRPO with log-probability ratios
#   - value_ratio: Element-wise value ratios
# ============================================================================

export HOME_REPO="$HOME/two_stage_neural_diving"
export LOG_DIR="$HOME/two_stage_neural_diving/logs"
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LOG_FILE="$SCRIPT_DIR/grpo_${SLURM_JOB_ID}.log"
export TMPDIR=/tmp

# --- Dataset selection ---
export DATASETS="banknote"
# Multi-dataset: export DATASETS="seeds,glass,banknote"

# --- Reproducibility ---
SEED=${SEED:-42}
export SEED

# --- GRPO Loss Method ---
# Set to e2e_accuracy for policy gradient training based on end-to-end classification accuracy
# This is the key difference from run_small.sh (which uses supervised_only)
LOSS_METHOD=${LOSS_METHOD:-e2e_accuracy}
export LOSS_METHOD

# --- Batch size ---
# GRPO works with batches for group normalization (unlike PPO which needs batch_size=1)
# Larger batches provide better baseline estimates for advantage normalization
BATCH_SIZE=${BATCH_SIZE:-10}
export BATCH_SIZE

# --- Dataset size (100, 250, 500, 1000) ---
NUM_FILES=${NUM_FILES:-1000}
export NUM_FILES

# --- Gradient accumulation ---
# 1 = no accumulation (update every batch) - DEFAULT
# N = accumulate gradients across N batches before stepping
# 0 = auto (accumulate across all batches in epoch)
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
export GRAD_ACCUM_STEPS

# ============================================================================
echo "=========================================="
echo "GRPO TRAINING (E2E Accuracy Loss)"
echo "=========================================="
echo "Script:          hpc_train_grpo_small.py"
echo "Datasets:        $DATASETS"
echo "NUM_FILES:       $NUM_FILES"
echo "BATCH_SIZE:      $BATCH_SIZE"
echo "LOSS_METHOD:     $LOSS_METHOD  (e2e_accuracy = policy gradient)"
echo "SEED:            $SEED"
echo "GRAD_ACCUM:      $GRAD_ACCUM_STEPS"
echo "=========================================="

# Create logs directory
mkdir -p "$LOG_DIR" || { echo "ERROR: Failed to create $LOG_DIR"; exit 1; }

# Redirect all output to both log file and console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== HPC JOB STARTING =========="
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "CUDA Devices Available: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"

# Load environment
source ~/.bashrc
conda activate neural_diving_pytorch
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment" >&2
    exit 1
fi

# Navigate to project directory
cd "$HOME_REPO" || { echo "ERROR: Failed to cd into $HOME_REPO"; exit 1; }

echo "Starting GRPO training with $LOSS_METHOD loss method..."
TRAIN_SCRIPT="hpc_train_grpo_small.py"

python "$TRAIN_SCRIPT" --config=config_train_pytorch2.py > "$LOG_DIR/train_output_$SLURM_JOB_ID.log" 2>&1
EXIT_CODE=$?

echo "CUDA Devices Available:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader || echo "nvidia-smi found but failed"
else
    echo "nvidia-smi not found"
fi

echo "Checking CPU usage..."
top -b -n 1 | head -20

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script failed with exit code $EXIT_CODE" >&2
    echo "Check $LOG_DIR/train_output_$SLURM_JOB_ID.log for details."
    exit $EXIT_CODE
fi

echo "========== HPC JOB COMPLETED =========="
