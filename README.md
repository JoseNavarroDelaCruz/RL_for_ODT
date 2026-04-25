# PPO and GRPO for NeurODT

Fine-tuning a GPS-based neural surrogate for Optimal Decision Tree (ODT)
construction using Proximal Policy Optimisation (PPO) and Group Relative
Policy Optimisation (GRPO).

## Overview

NeurODT replaces a MIP solver (CPLEX) at inference time by training a
dual GPS (Graph Positional and Structural) architecture to predict optimal
decision tree variable assignments directly from a graph encoding of the
problem instance. This repository extends NeurODT with RL fine-tuning via
PPO and GRPO, evaluated on three classification datasets: `small_toy`,
`seeds`, and `glass`.

## Repository Structure

```
├── hpc_train_ppo_small.py   # PPO training loop with GPS critic
├── hpc_train_grpo_small.py  # GRPO training loop with group-relative advantages
├── two_stage_gps_small.py   # Dual GPS model architecture
├── ppo_utils.py             # Shared PPO utilities
├── config_train_pytorch2.py # Training configuration
├── run_ppo_small.sh         # SLURM job script for PPO (see below)
└── run_grpo_small.sh        # SLURM job script for GRPO
```

## Hardware Requirements

**An HPC cluster with GPU access is strongly recommended.**
The dual GPS architecture is memory- and compute-intensive; training on CPU
is prohibitively slow for dataset sizes of 1,000+ instances. The scripts
were developed and tested on a single NVIDIA GPU with 32 GB RAM and 16 CPU
cores.

## Running on HPC (SLURM)

Submit the PPO job using the provided bash script:

```bash
sbatch run_ppo_small.sh
```

The script (`run_ppo_small.sh`) handles environment activation, logging,
and SLURM resource allocation. Key parameters can be overridden at
submission time:

```bash
# Change dataset
DATASETS="seeds" sbatch run_ppo_small.sh

# Change number of training instances
NUM_FILES=500 sbatch run_ppo_small.sh

# Change random seed
SEED=123 sbatch run_ppo_small.sh
```

> **Note:** `BATCH_SIZE` is fixed at 1 for PPO. Each rollout samples
> `G=4` candidates for a single problem instance and normalises advantages
> within that group. Larger batch sizes would mix instances and break
> per-instance advantage normalisation.

Default SLURM configuration in `run_ppo_small.sh`:

| Resource | Value |
|---|---|
| Partition | `nopreempt` |
| GPUs | 1 |
| CPUs | 16 |
| Memory | 32 GB |
| Requeue on preemption | Yes |

## Key Hyperparameters

Both algorithms share the following defaults (edit in `main()` of each
training script):

| Parameter | Value | Description |
|---|---|---|
| `G` | 4 | Candidates sampled per instance |
| `temperature` | 1.8 | Sampling temperature |
| `clip_epsilon` | 0.4 | PPO/GRPO clip range |
| `w1` | 1.0 | Accuracy reward weight |
| `w2` | 0.5 | Feasibility penalty weight |

PPO-specific:

| Parameter | Value |
|---|---|
| `n_inner_epochs` | 4 |
| `kl_coef` | 0.01 |
| `kl_target` | 0.02 |

GRPO-specific:

| Parameter | Value |
|---|---|
| `alpha` (supervised anchor) | 0.3 |
| EMA momentum | 0.99 |

## Results

5-fold cross-validation test accuracy (100 instances per fold):

| Dataset | GPS (supervised) | PPO | GRPO | CART |
|---|---|---|---|---|
| `small_toy` | 0.629 | 0.692 | 0.692 | 0.395 |
| `glass` | 0.355 | 0.354 | 0.384 | 0.556 |
| `seeds` | 0.432 | 0.340 | **0.842** | 0.876 |

GRPO consistently matches or improves over the supervised GPS baseline.
The largest gain is on `seeds` (+41 pp over GPS).

## Dependencies

```bash
conda activate neural_diving_pytorch
```

Core dependencies: `PyTorch`, `torch-geometric`, `cplex` (for instance
generation), `numpy`, `scikit-learn`.
