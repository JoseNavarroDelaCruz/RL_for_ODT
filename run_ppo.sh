#!/bin/bash
#SBATCH --job-name=train_ppo
#SBATCH --output=/home/n/navarrodelacruz/two_stage_neural_diving/logs/ppo_small_%j.out
#SBATCH --error=/home/n/navarrodelacruz/two_stage_neural_diving/logs/ppo_small_%j.err
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
# Run script for PPO training
# Uses: two_stage_gps_small.py + hpc_train_ppo_small.py + ppo_utils.py
#
# Key PPO differences from GRPO run_small.sh:
#   BATCH_SIZE=1  → instance-level advantages (one problem per PPO rollout)
#   Inner epoch loop + KL early stopping handled inside hpc_train_ppo_small.py
#   PPO hyperparams (clip_epsilon, kl_coef, n_inner_epochs, entropy_coef)
#   are set in main() of hpc_train_ppo_small.py — edit there to tune them.
# ============================================================================

export HOME_REPO="$HOME/two_stage_neural_diving"
export LOG_DIR="$HOME/two_stage_neural_diving/logs"
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LOG_FILE="$SCRIPT_DIR/ppo_small_${SLURM_JOB_ID}.log"
export TMPDIR=/tmp

# --- Dataset selection ---
export DATASETS="glass"
# Multi-dataset: export DATASETS="seeds,glass,banknote"

# --- Reproducibility ---
SEED=${SEED:-42}
export SEED

# --- PPO requires batch_size=1 for instance-level advantages ---
# Each PPO rollout samples G candidates for a single problem instance,
# normalizes advantages within that group, then runs N_inner_epochs updates.
# Larger batch sizes would mix instances and break per-instance advantage normalization.
BATCH_SIZE=${BATCH_SIZE:-1}
export BATCH_SIZE

# --- Dataset size (100, 250, 500, 1000) ---
NUM_FILES=${NUM_FILES:-1000}
export NUM_FILES

# ============================================================================
echo "=========================================="
echo "PPO TRAINING (Explicit Tree Routing)"
echo "=========================================="
echo "Script:     hpc_train_ppo_small.py"
echo "Datasets:   $DATASETS"
echo "NUM_FILES:  $NUM_FILES"
echo "BATCH_SIZE: $BATCH_SIZE  (1 = instance-level PPO advantages)"
echo "SEED:       $SEED"
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

# Verify required files are present
for f in hpc_train_ppo_small.py ppo_utils.py two_stage_gps_small.py config_train_pytorch2.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file missing: $f" >&2
        exit 1
    fi
done

echo "Starting PPO training..."
TRAIN_SCRIPT="hpc_train_ppo_small.py"

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
