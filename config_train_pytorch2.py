"""Configuration parameters for Neural LNS training."""

import ml_collections
import os
import logging
from typing import Optional

# List all files in the directories
base_path = os.path.join(os.getcwd(), "hpc_datasets2")

# Helper function to get all files from nested problem folders (e.g., seeds, glass)
def get_all_problem_files(split: str, subfolder: str, base_path: str, datasets_to_include=None):
    """
    Args:
        split (str): 'train' or 'valid'
        subfolder (str): 'training' or 'outputs'
        datasets_to_include (List[str], optional): e.g., ['seeds'] or ['seeds', 'glass']
    
    Returns:
        List[str]: List of file paths
    """
    full_paths = []
    split_path = os.path.join(base_path, split)

    for problem in sorted(os.listdir(split_path)):  # Sort for deterministic order
        if datasets_to_include and problem not in datasets_to_include:
            continue

        problem_path = os.path.join(split_path, problem)
        target_dir = os.path.join(problem_path, subfolder)

        if not os.path.isdir(target_dir):
            continue

        files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])  # Sort for deterministic order
        full_paths.extend(files)

    return sorted(full_paths)  # Final sort ensures consistent ordering

# Gather all file paths
# train_datasets_files = get_all_problem_files("train", "training")
# train_outputs_files = get_all_problem_files("train", "outputs")

# valid_datasets_files = get_all_problem_files("valid", "training")
# valid_outputs_files = get_all_problem_files("valid", "outputs")




def get_light_gnn_model_config():
    """Current best LightGNN config."""
    config = ml_collections.ConfigDict()

    # Tunable parameters
    config.params = ml_collections.ConfigDict()
    config.params.n_layers = 6  # Reduced from 25 for faster training
    config.params.node_model_hidden_sizes = [22, 256]  # 22 base features (LP/warm_start removed) + 4 pos + 7 x_agg + 5 var_type = 38
    config.params.output_model_hidden_sizes = [256, 128]  # Reduced from [512, 256]
    config.params.dropout = 0.0  # DISABLED for overfitting test
    config.params.gps_heads = 4  # Reduced from 8 heads

    # NOTE: Total input features = 25 (base) + 4 (Z_POS_DIM) + 7 (X_AGG_DIM) = 36
    # The model (two_stage_gps_skipped5) automatically adds Z_POS_DIM and X_AGG_DIM internally

    return config


def get_config(test: Optional[bool] = False):
    """Training configuration."""
    config = ml_collections.ConfigDict()

    # Detect datasets from env variable or default to all
    datasets_env = os.getenv("DATASETS")  # Passed from SLURM bash script
    datasets_to_use = datasets_env.split(',') if datasets_env else None

    # Paths
    if os.getenv("SLURM_JOB_ID"):  
        config.work_unit_dir = os.path.join(os.environ["HOME_REPO"], "saved_models")
    else:  
        #config.work_unit_dir = "/Users/navarrodelacruz/Documents/GitHub/two_stage_gcn_hpc/"
        config.work_unit_dir = "/Users/josen/OneDrive/Documentos/two_stage_gcn_hpc"


    # Training config - scale epochs inversely with dataset size
    # NOTE: With reduction='mean', we need higher LR than with reduction='sum'
    num_files = int(os.getenv('NUM_FILES', '1000'))
    
    if num_files <= 20:
        # 20 files or fewer: quick test run (2 batches, many epochs for convergence)
        config.num_train_steps = 400
        config.eval_every_steps = 20
    elif num_files == 100:
        # 100 files: longer training for thorough learning
        config.num_train_steps = 400
        config.eval_every_steps = 25
    elif num_files == 250:
        # 250 files: medium epochs (2.5× data, ~60% epochs)
        config.num_train_steps = 250
        config.eval_every_steps = 20
    elif num_files == 500:
        # 500 files: balanced epochs (5× data, ~55% epochs) - RECOMMENDED FOR GENERALIZATION
        config.num_train_steps = 220
        config.eval_every_steps = 18
    elif num_files == 1000:
        # 1000 files: fewer epochs (10× data, ~50% epochs)
        config.num_train_steps = 200
        config.eval_every_steps = 5
    else:
        # Custom: scale inversely with sqrt of data size
        config.num_train_steps = max(150, int(400 * (100 / num_files) ** 0.5))
        config.eval_every_steps = max(10, config.num_train_steps // 20)
        logging.info(f"[CUSTOM] NUM_FILES={num_files} not standard; using num_train_steps={config.num_train_steps}")
    
    logging.info(f"[NUM_FILES={num_files}] num_train_steps={config.num_train_steps}, eval_every={config.eval_every_steps}")
    
    # GRPO Stabilization: Reduced LR to make updates more stable + complement higher KL penalty
    # - Was: 0.001 (too aggressive with GRPO importance ratios)
    # - Now: 0.0005 (slower updates = more stable policy changes)
    # - Also: decoder/b optimizers scale from this (decoder: 0.5x, b: 0.1x)
    config.learning_rate = 1e-3
    config.decay_steps = None
    config.num_train_run_steps = 1
    config.eval_steps = 5
    config.grad_clip_norm = 80.0  # Increased from 15 - allow full gradient updates (gradients stable at 33-35)

    # Temperature annealing for routing decisions (replaces STE)
    config.routing_tau_start = 2.0   # High temp = soft routing, good gradients early
    config.routing_tau_end = 0.1     # Low temp = near-hard routing, tree-like at end

    # Dataset loading based on selected datasets
    if test == True:
        test_datasets_files = get_all_problem_files("", "training", base_path, datasets_to_use)
        test_outputs_files = get_all_problem_files("", "outputs", base_path, datasets_to_use)

        config.test_problems_datasets = [(test_datasets_files, 'test_datasets')]
        config.test_problems_outputs = [(test_outputs_files, 'test_outputs')]

    else:
        train_datasets_files = get_all_problem_files("train", "training", base_path, datasets_to_use)
        train_outputs_files = get_all_problem_files("train", "outputs", base_path, datasets_to_use)
        train_linear_feats_files = get_all_problem_files("train", "linear_feats", base_path, datasets_to_use)  # ADD

        
        valid_datasets_files = get_all_problem_files("valid", "training", base_path, datasets_to_use)
        valid_outputs_files = get_all_problem_files("valid", "outputs", base_path, datasets_to_use)
        valid_linear_feats_files = get_all_problem_files("valid", "linear_feats", base_path, datasets_to_use)  # ADD

        config.train_problems_datasets = [(train_datasets_files, 'train_datasets')]
        config.train_problems_outputs = [(train_outputs_files, 'train_outputs')]
        config.train_problems_linear_feats = [(train_linear_feats_files, 'train_linear_feats')]  # ADD

        config.valid_problems_datasets = [(valid_datasets_files, 'valid_datasets')]
        config.valid_problems_outputs = [(valid_outputs_files, 'valid_outputs')]
        config.valid_problems_linear_feats = [(valid_linear_feats_files, 'valid_linear_feats')]  # ADD

    logging.info(f"Using datasets: {datasets_to_use if datasets_to_use else 'ALL'}")


    config.model_config = get_light_gnn_model_config()
    return config


