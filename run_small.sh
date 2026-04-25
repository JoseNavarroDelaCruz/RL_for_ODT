#!/bin/bash
#SBATCH --job-name=gcn_small
#SBATCH --output=/home/n/navarrodelacruz/two_stage_neural_diving/logs/gcn_small_%j.out
#SBATCH --error=/home/n/navarrodelacruz/two_stage_neural_diving/logs/gcn_small_%j.err
#SBATCH -p nopreempt
#SBATCH --nodelist GPU2
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
# Run script for SIMPLIFIED model with explicit tree routing
# Uses: two_stage_gps_small.py + hpc_train_grpo_small.py
# ============================================================================

export HOME_REPO="$HOME/two_stage_neural_diving"
export LOG_DIR="$HOME/two_stage_neural_diving/logs"
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LOG_FILE="$SCRIPT_DIR/gcn_small_${SLURM_JOB_ID}.log"
export TMPDIR=/tmp # Set the temporary directory to the node's local /tmp folder

# Set DATASETS
export DATASETS="small_toy"
# If all datasets needed, do:
#export DATASETS="seeds,glass,banknote"

# Set SEED: random seed for reproducibility
# Use different seeds to run multiple experiments
SEED=${SEED:-42}
export SEED

# Set BATCH_SIZE: number of problems per batch
BATCH_SIZE=${BATCH_SIZE:-10}
export BATCH_SIZE

# Set NUM_FILES: dataset size (100, 250, 500, 1000)
NUM_FILES=${NUM_FILES:-1000}
export NUM_FILES

# Set GRAD_ACCUM_STEPS: gradient accumulation
# 1 = no accumulation (update every batch) - DEFAULT
# N = accumulate gradients across N batches before stepping
# 0 = auto (accumulate across all batches in epoch)
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
export GRAD_ACCUM_STEPS

echo "=========================================="
echo "SIMPLIFIED MODEL (Explicit Tree Routing)"
echo "=========================================="
echo "Training with NUM_FILES=$NUM_FILES, BATCH_SIZE=$BATCH_SIZE, SEED=$SEED"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR" || { echo "ERROR: Failed to create $LOG_DIR"; exit 1; }

# Redirect all output to both log file and console
exec > >(tee -a "$LOG_FILE") 2>&1



# 1) Log allocated resources (for debugging)
echo "========== HPC JOB STARTING =========="
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "CUDA Devices Available: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"

# 2) Load the environment
source ~/.bashrc  # Ensure Conda is available
conda activate neural_diving_pytorch  # or however you activate your environment
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment" >&2
    exit 1
fi

# 3) Navigate to the project directory
cd "$HOME_REPO" || { echo "ERROR: Failed to cd into $HOME_REPO"; exit 1; }


# 4) Run simplified training script
echo "Starting SIMPLIFIED GRPO training (explicit tree routing)..."
TRAIN_SCRIPT="hpc_train_grpo_small.py"

python "$TRAIN_SCRIPT" --config=config_train_pytorch2.py > "$LOG_DIR/train_output_$SLURM_JOB_ID.log" 2>&1
EXIT_CODE=$?


echo "CUDA Devices Available:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader || echo "nvidia-smi found but failed (likely no GPUs on this node)"
else
    echo "nvidia-smi not found"
fi


echo "Checking CPU usage..."
top -b -n 1 | head -20

# 5) Check if Python script ran successfully
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script failed with exit code $EXIT_CODE" >&2
    echo "Check $LOG_DIR/train_output_$SLURM_JOB_ID.log for details."
    exit $EXIT_CODE
fi

echo "========== HPC JOB COMPLETED =========="
