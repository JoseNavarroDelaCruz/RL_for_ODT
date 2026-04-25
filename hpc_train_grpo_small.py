# ============================================================================
# HPC Train with GRPO - SIMPLIFIED MODEL (Explicit Tree Routing)
# ============================================================================
# This training script uses the simplified two_stage_gps_small model with:
# - Explicit tree routing in decoder_forward() (no learned decoder)
# - Z computed directly from A, B, X using differentiable path probabilities
# - L computed from Z-C-Y alignment (misclassification count)
#
# GRPO techniques:
# 1. Group-Based Comparison - Learn relative quality instead of absolute targets
# 2. Baseline Normalization - Normalize within dataset groups
# 3. Multiple Candidate Sampling - Generate G predictions per input
# ============================================================================

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# NOTE: Using StepLR scheduler (ExponentialLR removed as unused)
from torch_geometric.data import Batch

# ============================================================================
# TF32 Precision for Ampere+ GPUs (A100, RTX 6000 Ada, RTX 30/40 series)
# ============================================================================
# TF32 provides ~2x speedup with FP32-like numerical stability (no NaN issues)
# - Uses 19-bit mantissa (vs FP16's 10-bit) for accuracy
# - Same exponent range as FP32 (no overflow/underflow)
# - Silently falls back to FP32 on older GPUs (V100, RTX 20 series)
# ============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import two_stage_data_utils3
import two_stage_gps_small
from config_train_pytorch2 import get_config
from two_stage_train_utils import log_available_resources, save_ckpt, load_ckpt, signal_handler, flatten_selected_solution_data
from grpo_utils import GRPOConfig, create_grpo_trainer
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from absl import logging
from absl import flags
from absl import app
import os
import tempfile
import logging
import math
import signal
import random
import numpy as np
import subprocess
import json
from datetime import datetime

# os.environ["DATASETS"] = "small_toy"
logging.basicConfig(level=logging.INFO, format='%(message)s')
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'
tempfile.tempdir = '/tmp'

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'config_train_pytorch2.py'),
    'Training configuration.')


MIN_LEARNING_RATE = 1e-5

# Clamp learnable loss weights to prevent E2E/L collapse in supervised_only.
# Raised E2E floor so the optimizer cannot down-weight E2E signal too aggressively.
MIN_E2E_WEIGHT = 8.0
MIN_L_WEIGHT = 0.8
E2E_PROB_FLOOR = 1e-3

# Entropy regularization weights for sharp feature/class selection
# Low entropy = one-hot distribution (sharp), High entropy = spread distribution
ENTROPY_WEIGHT_A = 0.0  # Temporarily disabled - model over-constrained, can't learn with penalty
ENTROPY_WEIGHT_C = 0.0  # Temporarily disabled - will re-enable after model shows learning


def compute_routing_temperature(global_step, num_train_steps, tau_start=2.0, tau_end=0.1):
    """Linear annealing of routing temperature: tau_start at step 0, tau_end at final step.
    High temp = soft routing (good gradients). Low temp = near-hard routing (tree-like)."""
    progress = min(global_step / max(num_train_steps, 1), 1.0)
    return tau_start + (tau_end - tau_start) * progress


# ============================================================================
# Reproducibility and Logging Utilities
# ============================================================================

def get_git_info():
    """Get git commit hash and branch for reproducibility tracking."""
    git_info = {
        'commit_hash': 'unknown',
        'commit_hash_short': 'unknown',
        'branch': 'unknown',
        'is_dirty': False,
        'commit_date': 'unknown',
        'commit_message': 'unknown',
    }
    try:
        # Get full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
            git_info['commit_hash_short'] = result.stdout.strip()[:8]

        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()

        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info['is_dirty'] = len(result.stdout.strip()) > 0

        # Get commit date
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info['commit_date'] = result.stdout.strip()

        # Get commit message (first line)
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%s'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info['commit_message'] = result.stdout.strip()[:80]  # Truncate long messages

    except Exception as e:
        logging.warning(f"Could not get git info: {e}")

    return git_info


def set_reproducibility_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    # Make CuDNN deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"🎲 Random seed set to: {seed}")
    logging.info(f"   → Python random, NumPy, PyTorch (CPU+CUDA) all seeded")
    logging.info(f"   → CuDNN deterministic mode: ENABLED")

    return seed


def log_full_config(config, grpo_config, git_info, seed, device):
    """Log complete configuration for reproducibility."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'local')
    hostname = os.getenv('HOSTNAME', os.getenv('SLURM_NODELIST', 'unknown'))

    logging.info("\n" + "="*80)
    logging.info("📋 FULL CONFIGURATION FOR REPRODUCIBILITY")
    logging.info("="*80)

    # --- Section 1: Run Identification ---
    logging.info("\n[1] RUN IDENTIFICATION")
    logging.info(f"    Timestamp:        {timestamp}")
    logging.info(f"    SLURM Job ID:     {slurm_job_id}")
    logging.info(f"    Hostname:         {hostname}")
    logging.info(f"    Random Seed:      {seed}")

    # --- Section 2: Git Information ---
    logging.info("\n[2] CODE VERSION (Git)")
    logging.info(f"    Commit Hash:      {git_info['commit_hash']}")
    logging.info(f"    Branch:           {git_info['branch']}")
    logging.info(f"    Commit Date:      {git_info['commit_date']}")
    logging.info(f"    Commit Message:   {git_info['commit_message']}")
    logging.info(f"    Working Dir Dirty: {git_info['is_dirty']}")
    if git_info['is_dirty']:
        logging.warning("    ⚠️  Uncommitted changes detected! Results may not be reproducible.")

    # --- Section 3: Hardware ---
    logging.info("\n[3] HARDWARE")
    logging.info(f"    Device:           {device}")
    if device.type == 'cuda':
        logging.info(f"    GPU:              {torch.cuda.get_device_name(0)}")
        logging.info(f"    GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logging.info(f"    TF32 (matmul):    {torch.backends.cuda.matmul.allow_tf32}")
        logging.info(f"    TF32 (cudnn):     {torch.backends.cudnn.allow_tf32}")
        logging.info(f"    CuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    logging.info(f"    CPUs (SLURM):     {os.getenv('SLURM_CPUS_PER_TASK', 'N/A')}")

    # --- Section 4: Dataset Configuration ---
    datasets_env = os.getenv("DATASETS", "all")
    num_files = os.getenv('NUM_FILES', '1000')
    batch_size = os.getenv('BATCH_SIZE', '10')

    logging.info("\n[4] DATASET CONFIGURATION")
    logging.info(f"    DATASETS env:     {datasets_env}")
    logging.info(f"    NUM_FILES:        {num_files}")
    logging.info(f"    BATCH_SIZE:       {batch_size}")

    # --- Section 5: Training Hyperparameters ---
    logging.info("\n[5] TRAINING HYPERPARAMETERS")
    logging.info(f"    Learning Rate:    {config.learning_rate}")
    logging.info(f"    Decay Steps:      {config.decay_steps}")
    logging.info(f"    Num Train Steps:  {config.num_train_steps}")
    logging.info(f"    Eval Every Steps: {config.eval_every_steps}")
    logging.info(f"    Grad Clip Norm:   {config.grad_clip_norm}")

    # --- Section 6: GRPO Configuration ---
    logging.info("\n[6] GRPO CONFIGURATION")
    logging.info(f"    num_candidates:       {grpo_config.num_candidates}")
    logging.info(f"    temperature:          {grpo_config.temperature}")
    logging.info(f"    sampling_dropout:     {grpo_config.sampling_dropout}")
    logging.info(f"    accuracy_weight:      {grpo_config.accuracy_weight}")
    logging.info(f"    infeasibility_weight: {grpo_config.infeasibility_weight}")
    logging.info(f"    use_clipping:         {grpo_config.use_clipping}")
    logging.info(f"    clip_epsilon:         {grpo_config.clip_epsilon}")
    logging.info(f"    clip_range:           {grpo_config.clip_range}")
    logging.info(f"    loss_method:          {grpo_config.loss_method}")
    logging.info(f"    e2e_weight:           {grpo_config.e2e_weight}")
    logging.info(f"    kl_coef:              {grpo_config.kl_coef}")
    logging.info(f"    min_group_size:       {grpo_config.min_group_size}")
    logging.info(f"    use_advantage_weighting: {grpo_config.use_advantage_weighting}")
    logging.info(f"    use_value_ratios:     {grpo_config.use_value_ratios}")
    logging.info(f"    norm_eps:             {grpo_config.norm_eps}")

    # --- Section 7: Model Architecture ---
    logging.info("\n[7] MODEL ARCHITECTURE")
    if hasattr(config, 'model_config') and config.model_config:
        mc = config.model_config.params
        logging.info(f"    n_layers:             {mc.n_layers}")
        logging.info(f"    node_model_hidden:    {mc.node_model_hidden_sizes}")
        logging.info(f"    output_model_hidden:  {mc.output_model_hidden_sizes}")
        logging.info(f"    dropout:              {mc.dropout}")
        logging.info(f"    gps_heads:            {mc.gps_heads}")

    # --- Section 8: Full Config JSON (for copy-paste replication) ---
    logging.info("\n[8] REPLICATION COMMAND")
    logging.info(f"    To replicate this run:")
    logging.info(f"    export SEED={seed} DATASETS={datasets_env} NUM_FILES={num_files} BATCH_SIZE={batch_size}")
    logging.info(f"    git checkout {git_info['commit_hash_short']}")

    logging.info("\n" + "="*80)
    logging.info("📋 END CONFIGURATION")
    logging.info("="*80 + "\n")

    # Save config to JSON file in json_logs folder
    config_dict = {
        'timestamp': timestamp,
        'slurm_job_id': slurm_job_id,
        'hostname': hostname,
        'seed': seed,
        'git': git_info,
        'datasets': datasets_env,
        'num_files': num_files,
        'batch_size': batch_size,
        'training': {
            'learning_rate': config.learning_rate,
            'decay_steps': config.decay_steps,
            'num_train_steps': config.num_train_steps,
            'eval_every_steps': config.eval_every_steps,
            'grad_clip_norm': config.grad_clip_norm,
        },
        'grpo': {
            'num_candidates': grpo_config.num_candidates,
            'temperature': grpo_config.temperature,
            'sampling_dropout': grpo_config.sampling_dropout,
            'accuracy_weight': grpo_config.accuracy_weight,
            'infeasibility_weight': grpo_config.infeasibility_weight,
            'use_clipping': grpo_config.use_clipping,
            'clip_epsilon': grpo_config.clip_epsilon,
            'loss_method': grpo_config.loss_method,
            'e2e_weight': grpo_config.e2e_weight,
            'kl_coef': grpo_config.kl_coef,
        },
    }

    # Save to JSON file in json_logs folder
    json_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json_logs')
    config_path = os.path.join(json_logs_dir, f"config_{slurm_job_id}.json")
    try:
        os.makedirs(json_logs_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logging.info(f"💾 Config saved to: {config_path}")
    except Exception as e:
        logging.warning(f"Could not save config JSON: {e}")


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal = alpha * (1 - pt) ** gamma * bce_loss
    return focal.mean() if reduction == 'mean' else focal.sum()


signal.signal(signal.SIGUSR1, signal_handler)

stop_training = False


class GRPOCandidateSampler:
    """
    Generates multiple diverse candidate predictions for GRPO training.
    
    Technique 3: Multiple Candidate Sampling
    - Uses dropout for diversity (even during inference)
    - Temperature scaling for exploration
    - Multiple forward passes with different noise
    
    Full Policy Gradient: Now samples discrete actions and tracks them
    for log-probability computation.
    
    GRPO Enhancement: Stores log_prob_old at sampling time for importance ratio.
    """
    
    def __init__(
        self,
        num_candidates: int = 4,
        temperature: float = 1.2,
        sampling_dropout: float = 0.15,
        b_log_std: float = 0.0,  # For Gaussian log-prob of b (std=1.0 for stability)
    ):
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.sampling_dropout = sampling_dropout
        self.b_log_std = b_log_std  # log(std) = 0 → std = 1.0
    
    def _compute_log_prob_at_sampling(
        self,
        pred_a: List[torch.Tensor],   # [P, Td] per graph
        pred_b: torch.Tensor,          # [total_Td]
        pred_c: List[torch.Tensor],   # [K, Tl] per graph
        pred_d: torch.Tensor,          # [total_Td]
        z_logits: torch.Tensor,        # [N, num_leaves]
        actions_a: List[torch.Tensor], # [Td] per graph
        actions_b: torch.Tensor,       # [total_Td]
        actions_c: List[torch.Tensor], # [Tl] per graph
        actions_d: torch.Tensor,       # [total_Td]
        actions_z: torch.Tensor,       # [N]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute log probabilities at sampling time (for π_θ_old).
        
        Returns per-variable log probs for variable-level decomposition.
        """
        device = pred_d.device
        T = self.temperature
        
        # === SIZE-INVARIANT LOG-PROBABILITIES ===
        # Must match compute_log_probability() normalization (mean, not sum)
        # so that importance ratio exp(log_new - log_old) is well-behaved.
        
        # === Log prob for 'a' (Categorical per decision node) — MEAN ===
        log_prob_a_sum = torch.tensor(0.0, device=device)
        num_a_elements = 0
        for logits, actions in zip(pred_a, actions_a):
            log_softmax_a = F.log_softmax(logits / T, dim=0)  # [P, Td]
            for t in range(logits.shape[1]):
                log_prob_a_sum = log_prob_a_sum + log_softmax_a[actions[t], t]
                num_a_elements += 1
        log_prob_a = log_prob_a_sum / max(num_a_elements, 1)
        
        # === Log prob for 'b' (Gaussian) — MEAN ===
        b_std = math.exp(self.b_log_std)
        b_diff = actions_b - pred_b
        # Clamp the squared difference to prevent extreme log-probs
        b_diff_sq = torch.clamp(b_diff ** 2, max=100.0)  # Prevent extreme values
        log_prob_b_per_node = -0.5 * b_diff_sq / (b_std ** 2) - self.b_log_std - 0.5 * math.log(2 * math.pi)
        # Clamp individual log-probs to reasonable range
        log_prob_b_per_node = torch.clamp(log_prob_b_per_node, min=-50.0, max=0.0)
        log_prob_b = log_prob_b_per_node.mean()  # MEAN not sum — size-invariant
        
        # === Log prob for 'c' (Categorical per leaf node) — MEAN ===
        log_prob_c_sum = torch.tensor(0.0, device=device)
        num_c_elements = 0
        for logits, actions in zip(pred_c, actions_c):
            log_softmax_c = F.log_softmax(logits / T, dim=0)  # [K, Tl]
            for t in range(logits.shape[1]):
                log_prob_c_sum = log_prob_c_sum + log_softmax_c[actions[t], t]
                num_c_elements += 1
        log_prob_c = log_prob_c_sum / max(num_c_elements, 1)
        
        # === Log prob for 'd' (Bernoulli) — MEAN ===
        d_probs = torch.sigmoid(pred_d / T)
        d_probs = torch.clamp(d_probs, 1e-7, 1 - 1e-7)
        log_prob_d_per_node = actions_d * torch.log(d_probs) + (1 - actions_d) * torch.log(1 - d_probs)
        log_prob_d = log_prob_d_per_node.mean()  # MEAN not sum — size-invariant
        
        # === Log prob for 'z' (Categorical per sample) — MEAN over N ===
        log_softmax_z = F.log_softmax(z_logits / T, dim=-1)  # [N, num_leaves]
        log_prob_z_per_sample = log_softmax_z.gather(dim=-1, index=actions_z.unsqueeze(-1)).squeeze(-1)
        log_prob_z = log_prob_z_per_sample.mean()  # MEAN not sum — size-invariant
        
        # Clamp total log-prob (tighter bounds since terms are means)
        log_prob_total = log_prob_a + log_prob_b + log_prob_c + log_prob_d + log_prob_z
        log_prob_total = torch.clamp(log_prob_total, min=-250.0, max=0.0)
        
        return {
            'log_prob_a_old': torch.clamp(log_prob_a, min=-50.0, max=0.0).detach(),
            'log_prob_b_old': torch.clamp(log_prob_b, min=-50.0, max=0.0).detach(),
            'log_prob_c_old': torch.clamp(log_prob_c, min=-50.0, max=0.0).detach(),
            'log_prob_d_old': torch.clamp(log_prob_d, min=-50.0, max=0.0).detach(),
            'log_prob_z_old': torch.clamp(log_prob_z, min=-50.0, max=0.0).detach(),
            'log_prob_total_old': log_prob_total.detach(),
        }
    
    def sample_candidates(
        self,
        model: nn.Module,
        inputs: Tuple,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate G diverse candidate predictions WITH sampled actions AND log_prob_old.
        
        For GRPO, we need to:
        1. Get logits from the model
        2. Sample discrete actions from the distributions
        3. Compute log_prob_old at sampling time (for importance ratio later)
        4. Track which actions were sampled
        
        Returns list of dictionaries with:
            - pred_*: Raw logits
            - actions_*: Sampled discrete actions
            - log_prob_*_old: Log probabilities at sampling time (π_θ_old)
        """
        candidates = []
        
        # Store original training mode
        was_training = model.training
        
        # Enable dropout for diversity
        model.train()
        
        # Get original dropout rates
        target = model.module if hasattr(model, 'module') else model
        original_dropout = getattr(target, 'dropout', 0.0)
        
        # Set sampling dropout
        self._set_dropout(model, self.sampling_dropout)
        
        try:
            for i in range(self.num_candidates):
                with torch.no_grad():
                    # Forward pass with dropout for diversity in SAMPLING
                    fs_logits, z_logits, L_logits, all_L = model(*inputs)
                    pred_a, pred_b, pred_c, pred_d = fs_logits
                    
                    # Apply temperature scaling for sampling
                    pred_a_temp = [a / self.temperature for a in pred_a]
                    pred_c_temp = [c / self.temperature for c in pred_c]
                    z_logits_temp = z_logits / self.temperature
                    
                    # === Sample discrete actions from distributions ===
                    
                    # Sample actions for 'a' (categorical per decision node)
                    actions_a = []
                    for pa in pred_a_temp:
                        probs_a = F.softmax(pa, dim=0)  # [P, Td]
                        sampled_a = torch.multinomial(probs_a.t(), num_samples=1).squeeze(-1)  # [Td]
                        actions_a.append(sampled_a)
                    
                    # Sample actions for 'c' (categorical per leaf node)
                    actions_c = []
                    for pc in pred_c_temp:
                        probs_c = F.softmax(pc, dim=0)  # [K, Tl]
                        sampled_c = torch.multinomial(probs_c.t(), num_samples=1).squeeze(-1)  # [Tl]
                        actions_c.append(sampled_c)
                    
                    # Sample actions for 'd' (Bernoulli per decision node)
                    probs_d = torch.sigmoid(pred_d / self.temperature)
                    actions_d = torch.bernoulli(probs_d)  # [total_Td]
                    
                    # Sample actions for 'z' (categorical per sample)
                    probs_z = F.softmax(z_logits_temp, dim=-1)  # [N, num_leaves]
                    actions_z = torch.multinomial(probs_z, num_samples=1).squeeze(-1)  # [N]
                    
                    # Sample 'b' with Gaussian noise
                    noise_b = torch.randn_like(pred_b) * math.exp(self.b_log_std)
                    actions_b = pred_b + noise_b
                    
                    # === OPTIMIZATION: Reuse sampling logits for log_prob_old ===
                    # Instead of a second forward pass, use the training-mode logits
                    # This halves the number of forward passes (4 candidates = 4 passes instead of 8)
                    pred_a_eval, pred_b_eval, pred_c_eval, pred_d_eval = pred_a, pred_b, pred_c, pred_d
                    z_logits_eval = z_logits
                    
                    log_probs_old = self._compute_log_prob_at_sampling(
                        pred_a_eval, pred_b_eval, pred_c_eval, pred_d_eval, z_logits_eval,
                        actions_a, actions_b, actions_c, actions_d, actions_z
                    )
                    
                    # OPTIMIZED: Reuse already-computed temperature-scaled probabilities
                    # These were computed above for sampling, no need to recompute
                    old_a_probs = [F.softmax(a, dim=0).detach() for a in pred_a_temp]  # Already temp-scaled
                    old_b_values = pred_b_eval.detach().clone()
                    old_c_probs = [F.softmax(c, dim=0).detach() for c in pred_c_temp]  # Already temp-scaled
                    old_d_probs = probs_d.detach()  # Already computed for sampling
                    old_z_probs = probs_z.detach()  # Already computed for sampling
                    
                    candidates.append({
                        # Raw logits (unscaled) - from eval pass for consistency
                        'pred_a': pred_a_eval,
                        'pred_b': pred_b_eval,
                        'pred_c': pred_c_eval,
                        'pred_d': pred_d_eval,
                        'pred_z': z_logits_eval,
                        'pred_L': all_L,
                        # Temperature-scaled for reward computation
                        'pred_a_temp': pred_a_temp,
                        'pred_c_temp': pred_c_temp,
                        'pred_z_temp': z_logits_temp,
                        # Sampled discrete actions
                        'actions_a': actions_a,
                        'actions_b': actions_b,
                        'actions_c': actions_c,
                        'actions_d': actions_d,
                        'actions_z': actions_z,
                        # Log probabilities at sampling time (π_θ_old) - keep for comparison
                        **log_probs_old,
                        # NEW: Actual probability values for value-ratio
                        'old_a_probs': old_a_probs,   # List of [P, Td]
                        'old_b_values': old_b_values,  # [total_Td]
                        'old_c_probs': old_c_probs,    # List of [K, Tl]
                        'old_d_probs': old_d_probs,    # [total_Td]
                        'old_z_probs': old_z_probs,    # [N, num_leaves]
                    })
        finally:
            # Restore original state
            self._set_dropout(model, original_dropout)
            if not was_training:
                model.eval()
        
        return candidates
    
    def _set_dropout(self, model: nn.Module, rate: float):
        """Set dropout rate in model."""
        target = model.module if hasattr(model, 'module') else model
        if hasattr(target, 'dropout'):
            target.dropout = rate
        
        # Also update dropout layers
        for module in target.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate


class GRPORewardComputer:
    """
    Computes rewards for ODT predictions.
    
    Technique 1: Group-Based Comparison
    - Reward function: r(pred) = w1*accuracy - w2*infeasibility
    - Rewards normalized within group of candidates
    
    Uses SAMPLED ACTIONS for reward computation (not argmax of logits).
    """
    
    def __init__(
        self,
        accuracy_weight: float = 1.0,
        infeasibility_weight: float = 0.5,
        norm_eps: float = 1e-8,
    ):
        self.accuracy_weight = accuracy_weight
        self.infeasibility_weight = infeasibility_weight
        self.norm_eps = norm_eps
    
    def compute_reward(
        self,
        candidate: Dict[str, torch.Tensor],
        true_labels: Dict[str, torch.Tensor],
        variable_shapes: List,
        normalized_X: torch.Tensor = None,  # [N, P] normalized features for tree routing
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute reward for a single candidate based on SAMPLED ACTIONS.
        
        r(pred) = w1 * accuracy - w2 * infeasibility
        
        Key change: Uses actions_a, actions_c, actions_d, actions_z (sampled)
        instead of argmax of logits. This ensures the reward reflects
        the actual actions taken, not what the model was most confident about.
        """
        # Extract raw logits
        pred_a = candidate['pred_a']  # List of [P, Td]
        pred_b = candidate['pred_b']  # [total_Td]
        pred_c = candidate['pred_c']  # List of [K, Tl]
        pred_d = candidate['pred_d']  # [total_Td]
        pred_z = candidate['pred_z']  # [N, num_leaves]
        pred_L = candidate['pred_L']  # [batch] or scalar
        
        # Extract sampled actions
        actions_a = candidate['actions_a']  # List of [Td] - sampled feature indices
        actions_b = candidate['actions_b']  # [total_Td] - sampled thresholds
        actions_c = candidate['actions_c']  # List of [Tl] - sampled class indices
        actions_d = candidate['actions_d']  # [total_Td] - sampled binary activations
        actions_z = candidate['actions_z']  # [N] - sampled leaf indices
        
        # Extract true labels
        true_a_per_graph = true_labels['true_a_per_graph']  # List of [P, Td]
        true_b = true_labels['true_b']  # [total_Td]
        true_c_per_graph = true_labels['true_c_per_graph']  # List of [K, Tl]
        true_d = true_labels['true_d']  # [total_Td]
        z_labels_indices = true_labels['z_labels_indices']  # [N]
        L_labels_per_batch = true_labels['L_labels_per_batch']  # [batch]
        
        # 1. Compute accuracy reward using SAMPLED ACTIONS
        accuracy, accuracy_details = self._compute_accuracy_from_actions(
            actions_a, actions_b, actions_c, actions_d, actions_z,
            true_a_per_graph, true_b, true_c_per_graph, true_d, z_labels_indices
        )
        
        # 2. Compute infeasibility penalty (routing consistency)
        if normalized_X is not None:
            infeasibility, routing_details = self._compute_routing_infeasibility_from_actions(
                normalized_X, actions_a, actions_b, actions_d, actions_z, pred_z.shape[1]
            )
        else:
            # Fallback to constraint-based infeasibility
            infeasibility = self._compute_constraint_infeasibility(pred_a, pred_d)
            routing_details = {}
        
        # Total reward: r(pred) = w1 * accuracy - w2 * infeasibility
        reward = (
            self.accuracy_weight * accuracy
            - self.infeasibility_weight * infeasibility
        )
        
        metrics = {
            'accuracy': accuracy.item(),
            'infeasibility': infeasibility.item(),
            'total_reward': reward.item(),
            **accuracy_details,  # Include per-variable accuracy breakdown
            **routing_details,   # Include routing consistency details
        }
        
        return reward, metrics
    
    def _compute_accuracy_from_actions(
        self,
        actions_a: List[torch.Tensor],  # List of [Td] - sampled feature indices
        actions_b: torch.Tensor,         # [total_Td] - sampled thresholds
        actions_c: List[torch.Tensor],  # List of [Tl] - sampled class indices
        actions_d: torch.Tensor,         # [total_Td] - sampled binary
        actions_z: torch.Tensor,         # [N] - sampled leaf indices
        true_a: List[torch.Tensor],     # List of [P, Td] one-hot
        true_b: torch.Tensor,           # [total_Td]
        true_c: List[torch.Tensor],     # List of [K, Tl] one-hot
        true_d: torch.Tensor,           # [total_Td]
        true_z: torch.Tensor,           # [N] leaf indices (ground truth)
        b_tolerance: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute accuracy using SAMPLED ACTIONS vs ground truth.
        
        This is the key difference from _compute_accuracy:
        - Instead of using argmax(logits), we use the actually sampled actions
        - This ensures the reward reflects the quality of the sampled trajectory
        """
        device = actions_d.device
        accuracies = {}
        weights = {}
        
        # === Variable a: Compare sampled feature index vs true ===
        a_correct = 0
        a_total = 0
        for act_a, ta in zip(actions_a, true_a):
            true_idx = ta.argmax(dim=0)  # [Td] - ground truth feature indices
            a_correct += (act_a == true_idx).sum().item()
            a_total += act_a.numel()
        accuracies['acc_a'] = a_correct / max(a_total, 1)
        weights['acc_a'] = a_total
        
        # === Variable b: Compare sampled threshold vs true (tolerance-based) ===
        b_diff = (actions_b - true_b).abs()
        b_correct = (b_diff < b_tolerance).sum().item()
        b_total = actions_b.numel()
        accuracies['acc_b'] = b_correct / max(b_total, 1)
        weights['acc_b'] = b_total
        
        # === Variable c: Compare sampled class index vs true ===
        c_correct = 0
        c_total = 0
        for act_c, tc in zip(actions_c, true_c):
            true_idx = tc.argmax(dim=0)  # [Tl] - ground truth class indices
            c_correct += (act_c == true_idx).sum().item()
            c_total += act_c.numel()
        accuracies['acc_c'] = c_correct / max(c_total, 1)
        weights['acc_c'] = c_total
        
        # === Variable d: Compare sampled binary vs true ===
        true_d_binary = (true_d > 0.5).float()
        d_correct = (actions_d == true_d_binary).sum().item()
        d_total = actions_d.numel()
        accuracies['acc_d'] = d_correct / max(d_total, 1)
        weights['acc_d'] = d_total
        
        # === Variable z: Compare sampled leaf assignment vs true ===
        z_correct = (actions_z == true_z).sum().item()
        z_total = actions_z.numel()
        accuracies['acc_z'] = z_correct / max(z_total, 1)
        weights['acc_z'] = z_total
        
        # === UNIFORM average accuracy across all variables ===
        # Each variable contributes equally regardless of its element count.
        # This prevents z (N elements, varies by dataset) from dominating
        # the reward for large datasets (e.g., glass N=214 vs small_toy N=6).
        # Example without fix: glass z=79% of total weight, small_toy z=23%.
        uniform_accuracy = (
            accuracies['acc_a'] + accuracies['acc_b'] + accuracies['acc_c'] +
            accuracies['acc_d'] + accuracies['acc_z']
        ) / 5.0
        
        # === DIAGNOSTIC: Compare uniform vs weighted accuracy ===
        total_weight = sum(weights.values())
        weighted_accuracy = sum(accuracies[k] * weights[k] for k in accuracies) / max(total_weight, 1)
        z_weight_fraction = weights['acc_z'] / max(total_weight, 1)
        logging.info(f"[SIZE-NORM-TEST] Accuracy: uniform={uniform_accuracy:.4f}, weighted={weighted_accuracy:.4f} | "
                    f"Element counts: a={weights['acc_a']}, b={weights['acc_b']}, c={weights['acc_c']}, "
                    f"d={weights['acc_d']}, z={weights['acc_z']} | z fraction={z_weight_fraction:.2%}")
        
        return torch.tensor(uniform_accuracy, device=device), accuracies
    
    def _compute_end_to_end_accuracy(
        self,
        actions_c: List[torch.Tensor],  # List of [Tl] - sampled class indices per leaf
        actions_z: torch.Tensor,         # [N] - sampled leaf indices for each sample
        true_labels: torch.Tensor,       # [N] - ground truth class labels for each sample
        num_leaves: int = 4,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute end-to-end classification accuracy using C and Z predictions.
        
        Pipeline: sample → leaf assignment (Z) → class prediction (C at that leaf) → compare with ground truth
        
        For each sample s:
        1. Get the predicted leaf: leaf_s = Z[s]
        2. Get the class prediction at that leaf: pred_class_s = C[leaf_s]
        3. Compare with ground truth: correct if pred_class_s == true_labels[s]
        
        This measures the actual classification performance of the decision tree,
        which is what we ultimately care about.
        
        Args:
            actions_c: List of [Tl] tensors, each containing sampled class index for each leaf
                       For a tree with 4 leaves, actions_c[0] is a tensor of shape [4]
                       where actions_c[0][leaf_idx] = predicted class for that leaf
            actions_z: [N] tensor of leaf indices - which leaf each sample goes to
            true_labels: [N] tensor of ground truth class labels for each sample
            num_leaves: Number of leaves in the tree (default 4 for depth-2)
        
        Returns:
            accuracy: Scalar accuracy value
            details: Dictionary with per-leaf and overall statistics
        """
        device = actions_z.device
        N = actions_z.shape[0]
        
        # Get class predictions for each leaf from actions_c
        # actions_c is a list (one per graph in batch), we use the first one
        # actions_c[0] has shape [Tl] where Tl = num_leaves
        if len(actions_c) == 0:
            logging.warning("[E2E Accuracy] No actions_c provided")
            return torch.tensor(0.0, device=device), {'e2e_accuracy': 0.0}
        
        leaf_class_predictions = actions_c[0]  # [Tl] - class index for each leaf
        
        # For each sample, get the predicted class based on its leaf assignment
        # pred_class[s] = leaf_class_predictions[actions_z[s]]
        pred_classes = leaf_class_predictions[actions_z]  # [N] - predicted class for each sample (0-indexed)
        pred_classes = pred_classes + 1  # Align with 1-indexed dataset labels
        
        # Compare with ground truth
        correct = (pred_classes == true_labels).float()
        accuracy = correct.mean()
        
        # Per-leaf statistics
        leaf_stats = {}
        for leaf in range(num_leaves):
            leaf_mask = (actions_z == leaf)
            leaf_count = leaf_mask.sum().item()
            if leaf_count > 0:
                leaf_correct = correct[leaf_mask].sum().item()
                leaf_acc = leaf_correct / leaf_count
                leaf_stats[f'e2e_leaf_{leaf}_acc'] = leaf_acc
                leaf_stats[f'e2e_leaf_{leaf}_count'] = leaf_count
                leaf_stats[f'e2e_leaf_{leaf}_pred_class'] = leaf_class_predictions[leaf].item()
        
        details = {
            'e2e_accuracy': accuracy.item(),
            'e2e_total_samples': N,
            'e2e_total_correct': correct.sum().item(),
            **leaf_stats,
        }
        
        return accuracy, details
    
    def compute_reward_with_e2e_accuracy(
        self,
        candidate: Dict[str, torch.Tensor],
        true_labels: Dict[str, torch.Tensor],
        variable_shapes: List,
        sample_true_labels: torch.Tensor = None,  # [N] ground truth class labels per sample
        normalized_X: torch.Tensor = None,
        use_e2e_weight: float = 0.5,  # Weight for end-to-end accuracy vs variable accuracy
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute reward using end-to-end accuracy (C → Z → label prediction).
        
        This is an alternative to compute_reward that focuses on classification accuracy:
        - Variable accuracy: How well do individual predictions match ground truth?
        - E2E accuracy: Does the full pipeline produce correct class labels?
        
        Args:
            candidate: Dictionary with predictions and sampled actions
            true_labels: Dictionary with ground truth for all variables
            variable_shapes: Shapes of variables per graph
            sample_true_labels: [N] ground truth class labels for each sample
                               (this is the actual classification target, not z)
            normalized_X: [N, P] input features for routing computation
            use_e2e_weight: How much to weight E2E accuracy vs variable accuracy [0, 1]
        
        Returns:
            reward: Combined reward value
            metrics: Dictionary with detailed metrics
        """
        # Extract sampled actions
        actions_a = candidate['actions_a']
        actions_b = candidate['actions_b']
        actions_c = candidate['actions_c']
        actions_d = candidate['actions_d']
        actions_z = candidate['actions_z']
        
        # Extract true labels for variable-level accuracy
        true_a_per_graph = true_labels['true_a_per_graph']
        true_b = true_labels['true_b']
        true_c_per_graph = true_labels['true_c_per_graph']
        true_d = true_labels['true_d']
        z_labels_indices = true_labels['z_labels_indices']
        
        # 1. Compute variable-level accuracy (existing method)
        var_accuracy, var_details = self._compute_accuracy_from_actions(
            actions_a, actions_b, actions_c, actions_d, actions_z,
            true_a_per_graph, true_b, true_c_per_graph, true_d, z_labels_indices
        )
        
        # 2. Compute end-to-end accuracy if sample labels are provided
        if sample_true_labels is not None:
            e2e_accuracy, e2e_details = self._compute_end_to_end_accuracy(
                actions_c, actions_z, sample_true_labels, num_leaves=4
            )
        else:
            # Fallback: use z_labels_indices as a proxy for ground truth
            # This measures if the predicted leaf matches the "correct" leaf
            # Not ideal, but allows the method to work without explicit sample labels
            e2e_accuracy, e2e_details = self._compute_end_to_end_accuracy(
                actions_c, actions_z, z_labels_indices, num_leaves=4
            )
            e2e_details['e2e_using_z_proxy'] = True
        
        # 3. Combine accuracies
        combined_accuracy = (
            (1 - use_e2e_weight) * var_accuracy +
            use_e2e_weight * e2e_accuracy
        )
        
        # 4. Compute infeasibility penalty (routing consistency)
        if normalized_X is not None:
            infeasibility, routing_details = self._compute_routing_infeasibility_from_actions(
                normalized_X, actions_a, actions_b, actions_d, actions_z, num_leaves=4
            )
        else:
            infeasibility = self._compute_constraint_infeasibility(
                candidate['pred_a'], candidate['pred_d']
            )
            routing_details = {}
        
        # 5. Total reward
        reward = (
            self.accuracy_weight * combined_accuracy
            - self.infeasibility_weight * infeasibility
        )
        
        metrics = {
            'accuracy': combined_accuracy.item(),
            'var_accuracy': var_accuracy.item(),
            'e2e_accuracy': e2e_accuracy.item() if torch.is_tensor(e2e_accuracy) else e2e_accuracy,
            'infeasibility': infeasibility.item(),
            'total_reward': reward.item(),
            **var_details,
            **e2e_details,
            **routing_details,
        }
        
        return reward, metrics
    
    def _compute_routing_infeasibility_from_actions(
        self,
        X: torch.Tensor,                 # [N, P] normalized input features
        actions_a: List[torch.Tensor],   # List of [Td] - sampled feature indices
        actions_b: torch.Tensor,         # [total_Td] sampled split thresholds
        actions_d: torch.Tensor,         # [total_Td] sampled node activation
        actions_z: torch.Tensor,         # [N] sampled leaf indices from NN
        num_leaves: int = 4,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute routing infeasibility using SAMPLED ACTIONS.
        
        Routes samples through tree using sampled (a, b) and compares
        against sampled z from the NN. Uses hard (sampled) actions.
        """
        device = X.device
        N = X.shape[0]
        
        # Route samples through tree using SAMPLED feature indices and thresholds
        z_tree = self._route_samples_with_sampled_actions(X, actions_a, actions_b, num_leaves)
        
        # Get tree-based leaf assignments
        z_tree_hard = z_tree.argmax(dim=-1)  # [N]
        
        # Compare with sampled z from NN
        mismatches = (actions_z != z_tree_hard).float()
        infeasibility = mismatches.mean()
        
        # Per-leaf statistics
        leaf_match_rates = {}
        for leaf in range(num_leaves):
            tree_at_leaf = (z_tree_hard == leaf)
            nn_at_leaf = (actions_z == leaf)
            if tree_at_leaf.sum() > 0:
                agreement = (tree_at_leaf & nn_at_leaf).sum().float() / tree_at_leaf.sum().float()
                leaf_match_rates[f'leaf_{leaf}_agree'] = agreement.item()
        
        details = {
            'routing_mismatch': infeasibility.item(),
            'num_samples': N,
            **leaf_match_rates,
        }
        
        return infeasibility, details
    
    def _route_samples_with_sampled_actions(
        self,
        X: torch.Tensor,                 # [N, P]
        actions_a: List[torch.Tensor],   # List of [Td] - sampled feature indices
        actions_b: torch.Tensor,         # [total_Td] - sampled thresholds
        num_leaves: int = 4,
    ) -> torch.Tensor:
        """
        Route samples through tree using SAMPLED actions (not logits).
        
        For depth=2 tree with sampled feature indices and thresholds.
        """
        device = X.device
        N = X.shape[0]
        
        # Get sampled feature indices for each decision node
        feature_indices = actions_a[0]  # [Td] where Td=3 for depth=2
        Td = feature_indices.shape[0]
        
        # Get sampled thresholds
        b = actions_b[:Td]  # [Td]
        
        # Initialize leaf assignments
        z_tree = torch.zeros(N, num_leaves, device=device)
        
        # Route through tree using sampled actions
        feat_0 = feature_indices[0].item()
        go_left_0 = X[:, feat_0] < b[0]
        
        feat_1 = feature_indices[1].item()
        feat_2 = feature_indices[2].item()
        
        go_left_1 = X[:, feat_1] < b[1]
        go_left_2 = X[:, feat_2] < b[2]
        
        # Assign to leaves
        leaf_0_mask = go_left_0 & go_left_1
        leaf_1_mask = go_left_0 & ~go_left_1
        leaf_2_mask = ~go_left_0 & go_left_2
        leaf_3_mask = ~go_left_0 & ~go_left_2
        
        z_tree[leaf_0_mask, 0] = 1.0
        z_tree[leaf_1_mask, 1] = 1.0
        z_tree[leaf_2_mask, 2] = 1.0
        z_tree[leaf_3_mask, 3] = 1.0
        
        return z_tree

    def _compute_accuracy(
        self,
        pred_a: List[torch.Tensor],
        pred_b: torch.Tensor,
        pred_c: List[torch.Tensor],
        pred_d: torch.Tensor,
        true_a: List[torch.Tensor],
        true_b: torch.Tensor,
        true_c: List[torch.Tensor],
        true_d: torch.Tensor,
        b_tolerance: float = 0.1,  # Tolerance for continuous b values
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute classification/regression accuracy for variables a, b, c, d.
        (Legacy method using argmax of logits instead of sampled actions)
        
        Returns:
            accuracy: Weighted average accuracy across variables
            details: Per-variable accuracy breakdown
        """
        device = pred_d.device
        accuracies = {}
        weights = {}  # Weight by number of elements
        
        # === Variable a: Feature selection (categorical) ===
        # For each decision node, which feature is selected
        a_correct = 0
        a_total = 0
        for pa, ta in zip(pred_a, true_a):
            pred_idx = pa.argmax(dim=0)  # [Td] - which feature per node
            true_idx = ta.argmax(dim=0)  # [Td]
            a_correct += (pred_idx == true_idx).sum().item()
            a_total += pred_idx.numel()
        accuracies['acc_a'] = a_correct / max(a_total, 1)
        weights['acc_a'] = a_total
        
        # === Variable b: Split thresholds (continuous) ===
        # Use tolerance-based accuracy: |pred - true| < tolerance
        b_diff = (pred_b - true_b).abs()
        b_correct = (b_diff < b_tolerance).sum().item()
        b_total = pred_b.numel()
        accuracies['acc_b'] = b_correct / max(b_total, 1)
        weights['acc_b'] = b_total
        
        # === Variable c: Class assignment (categorical) ===
        # For each leaf, which class is assigned
        c_correct = 0
        c_total = 0
        for pc, tc in zip(pred_c, true_c):
            pred_idx = pc.argmax(dim=0)  # [Tl] - which class per leaf
            true_idx = tc.argmax(dim=0)  # [Tl]
            c_correct += (pred_idx == true_idx).sum().item()
            c_total += pred_idx.numel()
        accuracies['acc_c'] = c_correct / max(c_total, 1)
        weights['acc_c'] = c_total
        
        # === Variable d: Node activation (binary) ===
        # Threshold at 0.5 for binary classification
        pred_d_binary = (torch.sigmoid(pred_d) > 0.5).float()
        true_d_binary = (true_d > 0.5).float()
        d_correct = (pred_d_binary == true_d_binary).sum().item()
        d_total = pred_d.numel()
        accuracies['acc_d'] = d_correct / max(d_total, 1)
        weights['acc_d'] = d_total
        
        # === UNIFORM average accuracy across all variables ===
        # Each variable contributes equally regardless of element count,
        # ensuring fair representation across different-sized problems.
        uniform_accuracy = (
            accuracies['acc_a'] + accuracies['acc_b'] + accuracies['acc_c'] +
            accuracies['acc_d']
        ) / 4.0
        
        return torch.tensor(uniform_accuracy, device=device), accuracies
    
    def _route_samples_through_tree(
        self,
        X: torch.Tensor,           # [N, P] - normalized features for all samples
        pred_a: List[torch.Tensor], # List of [P, Td] per graph
        pred_b: torch.Tensor,      # [total_Td] - split thresholds
        pred_d: torch.Tensor,      # [total_Td] - node activation
        num_leaves: int = 4,
        tree_depth: int = 2,
    ) -> torch.Tensor:
        """
        Route samples through decision tree built from (a, b, d) using HARD routing.
        
        For depth=2 tree:
        - Decision nodes: 0, 1, 2 (T_D = 3)
        - Leaf nodes: 0, 1, 2, 3 (T_L = 4)
        
        Tree structure:
                  Node 0 (root)
                 /            \\
              Node 1         Node 2
              /    \\         /    \\
           Leaf 0  Leaf 1  Leaf 2  Leaf 3
        
        Routing logic: At each decision node t, compare X[s, feature_t] with b[t].
        If X[s, feature_t] < b[t], go left; otherwise go right.
        
        Returns:
            z_tree: [N, T_L] one-hot encoding of leaf assignments
        """
        device = X.device
        N, P = X.shape
        
        # For single graph case (batch processing)
        # pred_a[0] has shape [P, Td] where Td=3 for depth=2
        a = pred_a[0]  # [P, Td]
        Td = a.shape[1]  # Should be 3 for depth=2
        
        # Get hard feature selection: which feature at each decision node
        # a[:, t] contains logits over features, take argmax for hard routing
        a_probs = F.softmax(a, dim=0)  # [P, Td]
        feature_indices = a_probs.argmax(dim=0)  # [Td] - which feature for each node
        
        # Get split thresholds (sigmoid to [0, 1] since X is normalized)
        b = pred_b[:Td]  # [Td]
        
        # Initialize leaf assignments
        z_tree = torch.zeros(N, num_leaves, device=device)
        
        # Route each sample through the tree (vectorized per level)
        # Level 0: root decision
        feat_0 = feature_indices[0].item()
        go_left_0 = X[:, feat_0] < b[0]  # [N] boolean
        
        # Level 1: second level decisions
        feat_1 = feature_indices[1].item()
        feat_2 = feature_indices[2].item()
        
        go_left_1 = X[:, feat_1] < b[1]  # [N] boolean - for samples going to node 1
        go_left_2 = X[:, feat_2] < b[2]  # [N] boolean - for samples going to node 2
        
        # Assign to leaves based on path
        # Samples that went left at root (node 0) → node 1
        #   Then left at node 1 → leaf 0
        #   Then right at node 1 → leaf 1
        # Samples that went right at root (node 0) → node 2
        #   Then left at node 2 → leaf 2
        #   Then right at node 2 → leaf 3
        
        leaf_0_mask = go_left_0 & go_left_1      # Left at root, left at node 1
        leaf_1_mask = go_left_0 & ~go_left_1     # Left at root, right at node 1
        leaf_2_mask = ~go_left_0 & go_left_2     # Right at root, left at node 2
        leaf_3_mask = ~go_left_0 & ~go_left_2    # Right at root, right at node 2
        
        z_tree[leaf_0_mask, 0] = 1.0
        z_tree[leaf_1_mask, 1] = 1.0
        z_tree[leaf_2_mask, 2] = 1.0
        z_tree[leaf_3_mask, 3] = 1.0
        
        return z_tree
    
    def _compute_routing_infeasibility(
        self,
        X: torch.Tensor,              # [N, P] normalized input features
        pred_a: List[torch.Tensor],   # List of [P, Td] per graph
        pred_b: torch.Tensor,         # [total_Td] split thresholds
        pred_d: torch.Tensor,         # [total_Td] node activation
        pred_z: torch.Tensor,         # [N, num_leaves] NN's leaf assignment logits
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute routing infeasibility by comparing z_nn (from NN) vs z_tree (from tree routing).
        
        This implements constraints (2e) and (2f) from the ODT formulation:
        - If z_{st} = 1, then sample s must satisfy all routing conditions to reach leaf t
        
        The intuition: If the NN predicts z but the actual tree routing (using a, b, d)
        would assign samples differently, the solution is INFEASIBLE.
        
        Returns:
            infeasibility: Mismatch rate between z_nn and z_tree [0, 1]
            details: Dictionary with routing statistics
        """
        device = X.device
        N = X.shape[0]
        num_leaves = pred_z.shape[1]
        
        # Route samples through tree using predicted (a, b, d)
        z_tree = self._route_samples_through_tree(X, pred_a, pred_b, pred_d, num_leaves=num_leaves)
        
        # Get hard leaf assignments from NN
        z_nn_probs = F.softmax(pred_z, dim=-1)  # [N, num_leaves]
        z_nn_hard = z_nn_probs.argmax(dim=-1)   # [N] - leaf index per sample
        
        # Get tree-based leaf assignments
        z_tree_hard = z_tree.argmax(dim=-1)     # [N] - leaf index per sample
        
        # Compute mismatch rate (infeasibility)
        mismatches = (z_nn_hard != z_tree_hard).float()
        infeasibility = mismatches.mean()
        
        # Compute per-leaf statistics
        leaf_match_rates = {}
        for leaf in range(num_leaves):
            tree_at_leaf = (z_tree_hard == leaf)
            nn_at_leaf = (z_nn_hard == leaf)
            if tree_at_leaf.sum() > 0:
                # Of samples tree routes to this leaf, how many does NN agree?
                agreement = (tree_at_leaf & nn_at_leaf).sum().float() / tree_at_leaf.sum().float()
                leaf_match_rates[f'leaf_{leaf}_agree'] = agreement.item()
        
        details = {
            'routing_mismatch': infeasibility.item(),
            'num_samples': N,
            **leaf_match_rates,
        }
        
        return infeasibility, details
    
    def _compute_constraint_infeasibility(
        self,
        pred_a: List[torch.Tensor],
        pred_d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fallback: Compute constraint violation penalty for constraint 2g.
        Used when X is not available.
        """
        violations = []
        
        # Constraint 2g: sum_j(a_jt) = d_t
        d_probs = torch.sigmoid(pred_d)
        d_offset = 0
        
        for pa in pred_a:
            a_probs = F.softmax(pa, dim=0)  # [P, Td]
            Td = pa.shape[1]
            
            # Each column should sum to ~1 if d=1, ~0 if d=0
            a_sum = a_probs.sum(dim=0)  # [Td]
            d_segment = d_probs[d_offset:d_offset + Td]
            
            # Violation: |sum(a) - d|
            violation = (a_sum - d_segment).abs().mean()
            violations.append(violation)
            
            d_offset += Td
        
        if violations:
            total_violation = torch.stack(violations).mean()
        else:
            total_violation = torch.tensor(0.0, device=pred_d.device)
        
        return total_violation.clamp(0, 1)
    
    def normalize_rewards(self, rewards: List[torch.Tensor]) -> torch.Tensor:
        """
        Normalize rewards within group.
        
        r_norm = (r - mean(r)) / std(r)
        
        This is the core of GRPO: we learn "A is better than B"
        rather than "A should have value X".
        """
        rewards_tensor = torch.stack(rewards)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        
        normalized = (rewards_tensor - mean_r) / (std_r + self.norm_eps)
        return normalized


class GRPOLossComputer:
    """
    Computes FULL GRPO POLICY GRADIENT loss with importance ratio.
    
    Implements the GRPO objective with variable-level decomposition:
    
    L_GRPO = -1/G * Σ_g [ f(ratio_g) * A_g ]
    
    Where:
    - ratio_g = π_θ(actions_g) / π_θ_old(actions_g) = exp(log_prob_new - log_prob_old)
    - f(ratio) = ratio  (no clipping) or clip(ratio, 1-ε, 1+ε) (with clipping)
    - A_g is the normalized advantage for candidate g
    
    Variable-level decomposition:
    - Computes log-prob separately for a, b, c, d, z (autoregressive sequence)
    - log π = log π_a + log π_b + log π_c + log π_d + log π_z
    
    log_prob_old is stored at sampling time (no frozen model needed).
    """
    
    def __init__(
        self,
        device: torch.device,
        b_log_std: float = -1.0,
        use_clipping: bool = False,  # Optional PPO-style clipping
        clip_epsilon: float = 0.2,   # Clipping range [1-ε, 1+ε]
        temperature: float = 1.2,    # MUST match sampler temperature for correct ratio computation
    ):
        self.device = device
        # For continuous variable b, we use a fixed log std deviation
        self.b_log_std = b_log_std  # log(std) ≈ -1 → std ≈ 0.37
        self.temperature = temperature  # Must match GRPOCandidateSampler.temperature!
        self.use_clipping = use_clipping
        self.clip_epsilon = clip_epsilon
    
    def compute_log_probability(
        self,
        logits_a: List[torch.Tensor],  # [P, Td] per graph
        logits_b: torch.Tensor,         # [total_Td] 
        logits_c: List[torch.Tensor],  # [K, Tl] per graph
        logits_d: torch.Tensor,         # [total_Td]
        logits_z: torch.Tensor,         # [N, num_leaves]
        actions_a: List[torch.Tensor],  # [Td] sampled indices per graph
        actions_b: torch.Tensor,         # [total_Td] sampled values
        actions_c: List[torch.Tensor],  # [Tl] sampled indices per graph
        actions_d: torch.Tensor,         # [total_Td] sampled binary
        actions_z: torch.Tensor,         # [N] sampled leaf indices
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute log probability of sampled actions under CURRENT policy (π_θ_new).
        
        log π(actions | input) = log π_a + log π_b + log π_c + log π_d + log π_z
        
        Returns:
            total_log_prob: Scalar sum of all log probabilities
            log_probs: Dict with per-variable log probabilities
        """
        T = self.temperature
        log_probs = {}
        
        # === SIZE-INVARIANT LOG-PROBABILITIES ===
        # All log-probs are AVERAGED (not summed) over their elements.
        # This ensures the importance ratio exp(log_new - log_old) is comparable
        # across problems of different sizes (e.g., glass N=214 vs small_toy N=6).
        
        # === Log prob for 'a' (Categorical per decision node) — MEAN over elements ===
        log_prob_a_sum = torch.tensor(0.0, device=self.device)
        num_a_elements = 0
        for logits, actions in zip(logits_a, actions_a):
            # logits: [P, Td], actions: [Td]
            log_softmax_a = F.log_softmax(logits / T, dim=0)  # [P, Td]
            for t in range(logits.shape[1]):
                log_prob_a_sum = log_prob_a_sum + log_softmax_a[actions[t], t]
                num_a_elements += 1
        log_prob_a = log_prob_a_sum / max(num_a_elements, 1)
        log_probs['log_prob_a'] = torch.clamp(log_prob_a, min=-50.0, max=0.0)
        
        # === Log prob for 'b' (Gaussian with fixed std) — MEAN over elements ===
        b_std = math.exp(self.b_log_std)
        b_diff = actions_b - logits_b
        # Clamp squared difference to prevent extreme log-probs
        b_diff_sq = torch.clamp(b_diff ** 2, max=100.0)
        log_prob_b_per = -0.5 * b_diff_sq / (b_std ** 2) - self.b_log_std - 0.5 * math.log(2 * math.pi)
        # Clamp individual log-probs
        log_prob_b_per = torch.clamp(log_prob_b_per, min=-50.0, max=0.0)
        log_prob_b = log_prob_b_per.mean()  # MEAN not sum — size-invariant
        log_probs['log_prob_b'] = torch.clamp(log_prob_b, min=-50.0, max=0.0)
        
        # === Log prob for 'c' (Categorical per leaf node) — MEAN over elements ===
        log_prob_c_sum = torch.tensor(0.0, device=self.device)
        num_c_elements = 0
        for logits, actions in zip(logits_c, actions_c):
            log_softmax_c = F.log_softmax(logits / T, dim=0)  # [K, Tl]
            for t in range(logits.shape[1]):
                log_prob_c_sum = log_prob_c_sum + log_softmax_c[actions[t], t]
                num_c_elements += 1
        log_prob_c = log_prob_c_sum / max(num_c_elements, 1)
        log_probs['log_prob_c'] = torch.clamp(log_prob_c, min=-50.0, max=0.0)
        
        # === Log prob for 'd' (Bernoulli per decision node) — MEAN over elements ===
        d_probs = torch.sigmoid(logits_d / T)
        d_probs = torch.clamp(d_probs, 1e-7, 1 - 1e-7)
        log_prob_d_per = actions_d * torch.log(d_probs) + (1 - actions_d) * torch.log(1 - d_probs)
        log_prob_d = log_prob_d_per.mean()  # MEAN not sum — size-invariant
        log_probs['log_prob_d'] = torch.clamp(log_prob_d, min=-50.0, max=0.0)
        
        # === Log prob for 'z' (Categorical per sample) — MEAN over N samples ===
        # CRITICAL: z has N elements where N varies across datasets (6 vs 214).
        # Using mean ensures log_prob_z is comparable regardless of dataset size.
        log_softmax_z = F.log_softmax(logits_z / T, dim=-1)  # [N, num_leaves]
        log_prob_z_per = log_softmax_z.gather(dim=-1, index=actions_z.unsqueeze(-1)).squeeze(-1)
        log_prob_z = log_prob_z_per.mean()  # MEAN not sum — size-invariant
        log_probs['log_prob_z'] = torch.clamp(log_prob_z, min=-50.0, max=0.0)
        
        # === Total log probability (mean of per-variable means) ===
        # Since each variable's log-prob is already a mean, the total is their sum.
        # Clamp bounds are tighter since individual terms are means (not sums).
        total_log_prob = (log_probs['log_prob_a'] + log_probs['log_prob_b'] + 
                         log_probs['log_prob_c'] + log_probs['log_prob_d'] + log_probs['log_prob_z'])
        total_log_prob = torch.clamp(total_log_prob, min=-250.0, max=0.0)
        
        # === DIAGNOSTIC: Show mean-based log-probs (size-invariant) ===
        logging.info(f"[SIZE-NORM-TEST] Log-probs (MEAN-normalized): a={log_probs['log_prob_a'].item():.4f}, "
                    f"b={log_probs['log_prob_b'].item():.4f}, c={log_probs['log_prob_c'].item():.4f}, "
                    f"d={log_probs['log_prob_d'].item():.4f}, z={log_probs['log_prob_z'].item():.4f} | "
                    f"total={total_log_prob.item():.4f}")
        
        return total_log_prob, log_probs
    
    def compute_policy_gradient_loss(
        self,
        model: nn.Module,
        inputs: Tuple,
        candidates: List[Dict],
        advantages: torch.Tensor,  # [G] normalized advantages
        standard_loss_fn,
        true_labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss with importance ratio and optional PPO clipping.
        
        L_GRPO = -1/G * Σ_g [ f(ratio_g) * A_g ]
        
        Where:
        - ratio_g = exp(log_prob_new - log_prob_old)  (importance ratio)
        - f(ratio) = ratio  (no clipping)
                   = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)  (PPO clipping)
        
        We do G forward passes WITH gradients to compute log_prob_new,
        then use stored log_prob_old from sampling time.
        
        IMPORTANT: We use model.eval() during log-prob computation to ensure
        consistent outputs (no dropout randomness) between sampling and gradient pass.
        Gradients still flow because we don't use torch.no_grad().
        """
        # CRITICAL: Use eval mode to disable dropout for consistent log-prob computation
        # This ensures log_prob_new matches log_prob_old (same deterministic forward pass)
        # Gradients still flow - eval() only affects dropout/batchnorm behavior
        was_training = model.training
        model.eval()
        
        G = len(candidates)
        
        policy_terms = []  # Each term: f(ratio) * A
        metrics_per_candidate = []
        
        # Maximum allowed log-ratio to prevent numerical explosion
        MAX_LOG_RATIO = 10.0  # exp(10) ≈ 22000, exp(-10) ≈ 0.000045
        
        # OPTIMIZATION: Single forward pass for all candidates
        # In log-prob mode, we need gradients, so we do one forward pass
        # and reuse it for all candidates (they share the same current model output)
        fs_logits, z_logits, L_logits, all_L = model(*inputs)
        pred_a, pred_b, pred_c, pred_d = fs_logits
        
        for g, candidate in enumerate(candidates):
            # Extract sampled actions from this candidate
            actions_a = candidate['actions_a']
            actions_b = candidate['actions_b']
            actions_c = candidate['actions_c']
            actions_d = candidate['actions_d']
            actions_z = candidate['actions_z']
            
            # Compute log probability under CURRENT policy (π_θ_new)
            # REUSES cached forward pass
            log_prob_new, log_prob_components = self.compute_log_probability(
                pred_a, pred_b, pred_c, pred_d, z_logits,
                actions_a, actions_b, actions_c, actions_d, actions_z
            )
            
            # Get stored log probability from sampling time (π_θ_old)
            log_prob_old = candidate['log_prob_total_old']
            
            # Compute importance ratio: π_θ_new / π_θ_old
            # CRITICAL: Clamp log-ratio to prevent numerical explosion
            log_ratio_raw = log_prob_new - log_prob_old
            log_ratio = torch.clamp(log_ratio_raw, min=-MAX_LOG_RATIO, max=MAX_LOG_RATIO)
            ratio = torch.exp(log_ratio)
            
            # DEBUG: Check if log probs are changing
            if g == 0:  # Only log for first candidate to avoid spam
                logging.debug(f"[RATIO-DEBUG] Candidate {g}: log_prob_new={log_prob_new.item():.6f}, log_prob_old={log_prob_old.item():.6f}, log_ratio={log_ratio.item():.6f}, ratio={ratio.item():.4f}")
            
            # Compute per-variable ratios for logging (clamped)
            log_ratio_a = torch.clamp(log_prob_components['log_prob_a'] - candidate['log_prob_a_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_b = torch.clamp(log_prob_components['log_prob_b'] - candidate['log_prob_b_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_c = torch.clamp(log_prob_components['log_prob_c'] - candidate['log_prob_c_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_d = torch.clamp(log_prob_components['log_prob_d'] - candidate['log_prob_d_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_z = torch.clamp(log_prob_components['log_prob_z'] - candidate['log_prob_z_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            
            ratio_a = torch.exp(log_ratio_a)
            ratio_b = torch.exp(log_ratio_b)
            ratio_c = torch.exp(log_ratio_c)
            ratio_d = torch.exp(log_ratio_d)
            ratio_z = torch.exp(log_ratio_z)
            
            # DEBUG: Check per-variable ratios, especially B
            if g == 0:  # Only log for first candidate
                logging.debug(f"[VAR-RATIO-DEBUG] ratio_a={ratio_a.item():.4f}, ratio_b={ratio_b.item():.4f}, ratio_c={ratio_c.item():.4f}, ratio_d={ratio_d.item():.4f}, ratio_z={ratio_z.item():.4f}")
                logging.debug(f"[B-DEBUG] log_prob_b_new={log_prob_components['log_prob_b'].item():.6f}, log_prob_b_old={candidate['log_prob_b_old'].item():.6f}, log_ratio_b={log_ratio_b.item():.6f}")
            
            # Apply policy gradient with advantage
            A_g = advantages[g]
            
            if self.use_clipping:
                # PPO-style clipping: min(ratio * A, clip(ratio) * A)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                if A_g >= 0:
                    # For positive advantage, we want to increase probability
                    # clip prevents ratio from going too high
                    term = torch.min(ratio * A_g, clipped_ratio * A_g)
                else:
                    # For negative advantage, we want to decrease probability
                    # clip prevents ratio from going too low
                    term = torch.max(ratio * A_g, clipped_ratio * A_g)
                was_clipped = (ratio != clipped_ratio).float().item()
            else:
                # Standard GRPO without clipping
                term = ratio * A_g
                clipped_ratio = ratio
                was_clipped = 0.0
            
            policy_terms.append(term)
            
            # Track if log-ratio was clamped (indicates potential numerical issues)
            log_ratio_clamped = (log_ratio_raw.abs() > MAX_LOG_RATIO).float().item()
            
            metrics_per_candidate.append({
                'ratio': ratio.item(),
                'clipped_ratio': clipped_ratio.item() if isinstance(clipped_ratio, torch.Tensor) else clipped_ratio,
                'log_prob_new': log_prob_new.item(),
                'log_prob_old': log_prob_old.item(),
                'log_ratio': log_ratio.item(),
                'log_ratio_raw': log_ratio_raw.item(),  # Raw value before clamping
                'log_ratio_clamped': log_ratio_clamped,
                'advantage': A_g.item(),
                'term': term.item(),
                'was_clipped': was_clipped,
                # Per-variable ratios
                'ratio_a': ratio_a.item(),
                'ratio_b': ratio_b.item(),
                'ratio_c': ratio_c.item(),
                'ratio_d': ratio_d.item(),
                'ratio_z': ratio_z.item(),
            })
        
        # GRPO loss: -1/G * Σ_g [ f(ratio_g) * A_g ]
        # Negative because we want to MAXIMIZE the objective
        policy_gradient_loss = -torch.stack(policy_terms).mean()
        
        # OPTIMIZATION: Reuse cached forward pass for supervised loss
        supervised_loss, loss_components = standard_loss_fn(
            fs_logits, z_logits, L_logits, all_L, true_labels
        )
        
        # Restore model to original training mode
        if was_training:
            model.train()
        
        # === Combine losses ===
        # Policy gradient encourages good actions, supervised loss provides anchor
        alpha = 0.3  # Policy gradient weight (1-α goes to supervised)
        total_loss = alpha * policy_gradient_loss + (1 - alpha) * supervised_loss
        
        # Aggregate metrics
        best_idx = advantages.argmax().item()
        avg_ratio = sum(m['ratio'] for m in metrics_per_candidate) / G
        avg_log_ratio = sum(m['log_ratio'] for m in metrics_per_candidate) / G
        clip_fraction = sum(m['was_clipped'] for m in metrics_per_candidate) / G
        log_ratio_clamp_fraction = sum(m['log_ratio_clamped'] for m in metrics_per_candidate) / G
        
        metrics = {
            'policy_gradient_loss': policy_gradient_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'total_loss': total_loss.item(),
            'alpha': alpha,
            # Importance ratio statistics
            'avg_ratio': avg_ratio,
            'avg_log_ratio': avg_log_ratio,
            'best_ratio': metrics_per_candidate[best_idx]['ratio'],
            'best_log_ratio': metrics_per_candidate[best_idx]['log_ratio'],
            'best_log_ratio_raw': metrics_per_candidate[best_idx]['log_ratio_raw'],
            # Clipping statistics (even if clipping disabled, shows what would happen)
            'use_clipping': self.use_clipping,
            'clip_epsilon': self.clip_epsilon,
            'clip_fraction': clip_fraction,
            'log_ratio_clamp_fraction': log_ratio_clamp_fraction,  # Track numerical issues
            # Per-variable ratio stats for best candidate
            'best_ratio_a': metrics_per_candidate[best_idx]['ratio_a'],
            'best_ratio_b': metrics_per_candidate[best_idx]['ratio_b'],
            'best_ratio_c': metrics_per_candidate[best_idx]['ratio_c'],
            'best_ratio_d': metrics_per_candidate[best_idx]['ratio_d'],
            'best_ratio_z': metrics_per_candidate[best_idx]['ratio_z'],
            # Best candidate info
            'best_log_prob_new': metrics_per_candidate[best_idx]['log_prob_new'],
            'best_log_prob_old': metrics_per_candidate[best_idx]['log_prob_old'],
            'best_advantage': advantages[best_idx].item(),
            'best_candidate_idx': best_idx,
            'num_candidates': G,
            'metrics_per_candidate': metrics_per_candidate,
            # Supervised loss components
            **loss_components,
        }
        
        return total_loss, metrics

    def compute_advantage_weighted_loss(
        self,
        model: nn.Module,
        inputs: Tuple,
        true_labels: Dict[str, torch.Tensor],
        advantages: torch.Tensor,  # [G] normalized advantages
        candidates: List[Dict],
        standard_loss_fn,
        loss_method: str = 'supervised_only',  # 'log_prob', 'value_ratio', or 'e2e_accuracy'
        sample_true_labels: torch.Tensor = None,  # [N] ground truth class labels (for e2e_accuracy)
        normalized_X: torch.Tensor = None,  # [N, P] input features (for e2e_accuracy)
        e2e_weight: float = 0.5,  # Weight for E2E accuracy vs variable accuracy
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Entry point for GRPO loss computation.
        
        Args:
            loss_method: Which method to use for policy gradient computation:
                - 'log_prob': Original log-probability ratio approach
                - 'value_ratio': Advisor's element-wise value ratio approach (default)
                - 'e2e_accuracy': End-to-end accuracy-based approach (C → Z → label)
            sample_true_labels: [N] ground truth class labels for each sample
                               (required for 'e2e_accuracy' method)
            normalized_X: [N, P] input features for tree routing
            e2e_weight: How much to weight E2E accuracy vs variable accuracy [0, 1]
        """
        if loss_method == 'value_ratio':
            return self.compute_value_ratio_policy_loss(
                model, inputs, candidates, advantages, standard_loss_fn, true_labels
            )
        elif loss_method == 'e2e_accuracy':
            return self.compute_e2e_accuracy_policy_loss(
                model, inputs, candidates, advantages, standard_loss_fn, true_labels,
                sample_true_labels=sample_true_labels,
                normalized_X=normalized_X,
                e2e_weight=e2e_weight,
            )
        elif loss_method == 'supervised_only':
            # Pure supervised learning - no policy gradient, just minimize loss
            return self.compute_supervised_only_loss(
                model, inputs, standard_loss_fn, true_labels,
                sample_true_labels=sample_true_labels,
            )
        else:  # 'log_prob' or fallback
            return self.compute_policy_gradient_loss(
                model, inputs, candidates, advantages, standard_loss_fn, true_labels
            )

    def compute_supervised_only_loss(
        self,
        model: nn.Module,
        inputs: Tuple,
        standard_loss_fn,
        true_labels: Dict,
        sample_true_labels: torch.Tensor = None,  # [N] ground truth class labels (for E2E accuracy)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Pure supervised learning - no GRPO policy gradient.
        
        This is useful for:
        1. Pre-training first-stage variables before GRPO fine-tuning
        2. Debugging to ensure the model can learn with supervised loss
        3. When you want the model to overfit to the training data
        
        No candidate sampling, no advantage weighting - just direct supervision.
        """
        was_training = model.training
        model.train()  # Keep in training mode for dropout etc.
        
        # Forward pass
        fs_logits, z_logits, L_logits, all_L = model(*inputs)
        
        # === AMP STABILITY: Cast logits to float32 for loss computation ===
        # This prevents numerical issues when loss is computed in float16
        pred_a, pred_b, pred_c, pred_d = fs_logits
        pred_a = [a.float() for a in pred_a]
        pred_b = pred_b.float() if pred_b is not None else None
        pred_c = [c.float() for c in pred_c]
        pred_d = pred_d.float() if pred_d is not None else None
        fs_logits = [pred_a, pred_b, pred_c, pred_d]
        
        if z_logits is not None:
            z_logits = z_logits.float()
        if L_logits is not None:
            L_logits = L_logits.float()
        
        # Compute supervised loss (now in float32)
        total_loss, loss_components = standard_loss_fn(
            fs_logits, z_logits, L_logits, all_L, true_labels
        )
        
        # === ARGMAX ACCURACY on training data (deterministic, no sampling noise) ===
        # This measures the model's ACTUAL learned quality using greedy predictions.
        # Unlike "Best candidate accuracy" which uses temperature=1.8 + dropout=0.25 + noise,
        # this uses argmax of the raw logits - the model's best guess.
        with torch.no_grad():
            argmax_acc = {}
            
            # Variable a: argmax of logits vs true feature index
            a_correct, a_total = 0, 0
            for pa, ta in zip(pred_a, true_labels['true_a_per_graph']):
                pred_idx = pa.argmax(dim=0)  # [Td]
                true_idx = ta.argmax(dim=0)  # [Td]
                a_correct += (pred_idx == true_idx).sum().item()
                a_total += pred_idx.numel()
            argmax_acc['argmax_acc_a'] = a_correct / max(a_total, 1)
            
            # Variable b: within tolerance of true threshold
            b_diff = (pred_b - true_labels['true_b']).abs()
            b_correct = (b_diff < 0.1).sum().item()
            b_total = pred_b.numel()
            argmax_acc['argmax_acc_b'] = b_correct / max(b_total, 1)
            
            # Variable c: argmax of logits vs true class index
            c_correct, c_total = 0, 0
            for pc, tc in zip(pred_c, true_labels['true_c_per_graph']):
                pred_idx = pc.argmax(dim=0)  # [Tl]
                true_idx = tc.argmax(dim=0)  # [Tl]
                c_correct += (pred_idx == true_idx).sum().item()
                c_total += pred_idx.numel()
            argmax_acc['argmax_acc_c'] = c_correct / max(c_total, 1)
            
            # Variable d: sigmoid > 0.5 vs true binary
            pred_d_binary = (torch.sigmoid(pred_d) > 0.5).float()
            true_d_binary = (true_labels['true_d'] > 0.5).float()
            d_correct = (pred_d_binary == true_d_binary).sum().item()
            d_total = pred_d.numel()
            argmax_acc['argmax_acc_d'] = d_correct / max(d_total, 1)
            
            # Variable z: argmax of logits vs true leaf index
            if z_logits is not None:
                z_pred = z_logits.argmax(dim=-1)
                z_true = true_labels['z_labels_indices']
                z_correct = (z_pred == z_true).sum().item()
                z_total = z_true.numel()
                argmax_acc['argmax_acc_z'] = z_correct / max(z_total, 1)
            else:
                argmax_acc['argmax_acc_z'] = 0.0

            # === END-TO-END CLASSIFICATION ACCURACY (logging only, not used for loss) ===
            # This measures actual classification performance: route sample → get class from C → compare to true label
            # Unlike element-wise accuracy, this measures what matters at inference time.
            e2e_accuracy = 0.0
            e2e_correct = 0
            e2e_total = 0
            if sample_true_labels is not None and z_logits is not None:
                # Get predicted leaf for each sample
                z_pred_idx = z_logits.argmax(dim=-1)  # [N] leaf indices (0, 1, 2, 3)

                # Get predicted class for each sample based on predicted leaf and predicted C
                # pred_c is a list of [K, Tl] tensors, one per graph in batch
                # We need to map each sample to its graph's C matrix
                z_labels_list = true_labels.get('z_labels_list', None)
                if z_labels_list is not None:
                    pred_classes_list = []
                    sample_offset = 0
                    for graph_idx, (pc, z_lbl) in enumerate(zip(pred_c, z_labels_list)):
                        n_samples_in_graph = z_lbl.numel() // 4
                        if n_samples_in_graph > 0:
                            # Get leaf indices for samples in this graph
                            graph_z_pred = z_pred_idx[sample_offset:sample_offset + n_samples_in_graph]
                            # Get class predictions: for each sample, pred_class = argmax(C[:, leaf])
                            # pc is [K, Tl], we need C[:, leaf].argmax() for each sample
                            for leaf_idx in graph_z_pred:
                                leaf_idx_clamped = min(leaf_idx.item(), pc.shape[1] - 1)
                                pred_class = pc[:, leaf_idx_clamped].argmax().item()
                                pred_classes_list.append(pred_class)
                            sample_offset += n_samples_in_graph

                    if pred_classes_list:
                        pred_classes_tensor = torch.tensor(pred_classes_list, device=z_logits.device)
                        pred_classes_tensor = pred_classes_tensor + 1
                        true_labels_adjusted = sample_true_labels[:len(pred_classes_list)].long()
                        e2e_correct = (pred_classes_tensor == true_labels_adjusted).sum().item()
                        e2e_total = len(pred_classes_list)
                        e2e_accuracy = e2e_correct / max(e2e_total, 1)

            argmax_acc['e2e_accuracy'] = e2e_accuracy
            argmax_acc['e2e_correct'] = e2e_correct
            argmax_acc['e2e_total'] = e2e_total

        if was_training:
            model.train()
        else:
            model.eval()
        
        metrics = {
            'policy_gradient_loss': 0.0,  # No PG loss in supervised mode
            'supervised_loss': total_loss.item(),
            'total_loss': total_loss.item(),
            'alpha': 0.0,  # All supervised
            'loss_method': 'supervised_only',
            **argmax_acc,  # Include argmax accuracy metrics
            # Compatibility keys
            'avg_ratio': 1.0,
            'best_ratio': 1.0,
            'best_ratio_a': 1.0,
            'best_ratio_b': 1.0,
            'best_ratio_c': 1.0,
            'best_ratio_d': 1.0,
            'best_ratio_z': 1.0,
            'avg_log_ratio': 0.0,
            'best_log_ratio': 0.0,
            'best_log_ratio_raw': 0.0,
            'best_log_prob_new': 0.0,
            'best_log_prob_old': 0.0,
            'best_advantage': 0.0,
            'log_ratio_clamp_fraction': 0.0,
            'clip_fraction': 0.0,
            'best_candidate_idx': 0,
            'num_candidates': 1,
            'use_clipping': False,
            'clip_epsilon': 0.0,
            **loss_components,
        }
        
        return total_loss, metrics

    def compute_elementwise_value_ratios(
        self,
        new_a_logits: List[torch.Tensor],
        new_b: torch.Tensor,
        new_c_logits: List[torch.Tensor],
        new_d_logits: torch.Tensor,
        new_z_logits: torch.Tensor,
        old_a_probs: List[torch.Tensor],
        old_b_values: torch.Tensor,
        old_c_probs: List[torch.Tensor],
        old_d_probs: torch.Tensor,
        old_z_probs: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        '''
        Compute element-wise probability ratios: ratio[i,j] = new[i,j] / old[i,j]
        
        For a [7, 3] matrix, computes all 21 individual ratios.
        '''
        T = self.temperature
        
        # Variable a: Element-wise for each [P, Td] matrix
        ratio_a_list = []
        for new_a, old_a in zip(new_a_logits, old_a_probs):
            new_a_probs = F.softmax(new_a / T, dim=0)  # [P, Td]
            # Clamp old_a to avoid division by very small values
            old_a_safe = torch.clamp(old_a, min=eps)
            ratio_a = new_a_probs / old_a_safe  # [P, Td]
            # Clamp ratio to prevent extreme values
            ratio_a = torch.clamp(ratio_a, min=0.01, max=100.0)
            ratio_a_list.append(ratio_a)
        
        # Variable b: Use ratio of absolute values + sign, or just use difference-based approach
        # The old approach of dividing raw values is numerically unstable
        # Instead, use a bounded similarity measure
        b_diff = (new_b - old_b_values).abs()
        b_scale = torch.clamp(old_b_values.abs(), min=0.1)  # Avoid division by tiny values
        ratio_b = 1.0 / (1.0 + b_diff / b_scale)  # Bounded in (0, 1], closer values → ratio ≈ 1
        ratio_b = torch.clamp(ratio_b, min=0.01, max=100.0)
        
        # Variable c: Element-wise for each [K, Tl] matrix
        ratio_c_list = []
        for new_c, old_c in zip(new_c_logits, old_c_probs):
            new_c_probs = F.softmax(new_c / T, dim=0)
            old_c_safe = torch.clamp(old_c, min=eps)
            ratio_c = new_c_probs / old_c_safe  # [K, Tl]
            ratio_c = torch.clamp(ratio_c, min=0.01, max=100.0)
            ratio_c_list.append(ratio_c)
        
        # Variable d: Element-wise
        new_d_probs = torch.sigmoid(new_d_logits / T)
        old_d_safe = torch.clamp(old_d_probs, min=eps, max=1-eps)
        new_d_safe = torch.clamp(new_d_probs, min=eps, max=1-eps)
        ratio_d = new_d_safe / old_d_safe  # [total_Td]
        ratio_d = torch.clamp(ratio_d, min=0.01, max=100.0)
        
        # Variable z: Element-wise
        new_z_probs = F.softmax(new_z_logits / T, dim=-1)
        old_z_safe = torch.clamp(old_z_probs, min=eps)
        ratio_z = new_z_probs / old_z_safe  # [N, num_leaves]
        ratio_z = torch.clamp(ratio_z, min=0.01, max=100.0)
        
        # Aggregate to scalars (mean of all elements)
        mean_ratio_a = torch.cat([r.flatten() for r in ratio_a_list]).mean()
        mean_ratio_b = ratio_b.mean()
        mean_ratio_c = torch.cat([r.flatten() for r in ratio_c_list]).mean()
        mean_ratio_d = ratio_d.mean()
        mean_ratio_z = ratio_z.mean()
        
        # Clamp individual means before combining
        mean_ratio_a = torch.clamp(mean_ratio_a, min=0.1, max=10.0)
        mean_ratio_b = torch.clamp(mean_ratio_b, min=0.1, max=10.0)
        mean_ratio_c = torch.clamp(mean_ratio_c, min=0.1, max=10.0)
        mean_ratio_d = torch.clamp(mean_ratio_d, min=0.1, max=10.0)
        mean_ratio_z = torch.clamp(mean_ratio_z, min=0.1, max=10.0)
        
        # Combined: geometric mean (safe with clamped values)
        combined_ratio = (mean_ratio_a * mean_ratio_b * mean_ratio_c * mean_ratio_d * mean_ratio_z) ** 0.2
        
        # Final safety clamp
        combined_ratio = torch.clamp(combined_ratio, min=0.1, max=10.0)
        
        return {
            'ratio_a_list': ratio_a_list,
            'ratio_b': ratio_b,
            'ratio_c_list': ratio_c_list,
            'ratio_d': ratio_d,
            'ratio_z': ratio_z,
            'mean_ratio_a': mean_ratio_a,
            'mean_ratio_b': mean_ratio_b,
            'mean_ratio_c': mean_ratio_c,
            'mean_ratio_d': mean_ratio_d,
            'mean_ratio_z': mean_ratio_z,
            'combined_ratio': combined_ratio,
        }
    
    def compute_value_ratio_policy_loss(
        self,
        model: nn.Module,
        inputs: Tuple,
        candidates: List[Dict],
        advantages: torch.Tensor,
        standard_loss_fn,
        true_labels: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        '''
        GRPO loss using direct value ratios instead of log-probability ratios.
        
        OPTIMIZED: Single forward pass for all computations.
        - One forward pass computes ratios for ALL candidates
        - Same forward pass reused for supervised loss
        - Reduces forward passes from (G + 1) to 1
        '''
        was_training = model.training
        model.eval()
        
        G = len(candidates)
        policy_terms = []
        metrics_per_candidate = []
        
        # OPTIMIZATION: Single forward pass for all candidates
        # All candidates use the same current model outputs (they differ only in old_* values)
        fs_logits, z_logits, L_logits, all_L = model(*inputs)
        pred_a, pred_b, pred_c, pred_d = fs_logits
        
        for g, candidate in enumerate(candidates):
            # Compute element-wise value ratios using CACHED forward pass
            ratios = self.compute_elementwise_value_ratios(
                pred_a, pred_b, pred_c, pred_d, z_logits,
                candidate['old_a_probs'],
                candidate['old_b_values'],
                candidate['old_c_probs'],
                candidate['old_d_probs'],
                candidate['old_z_probs'],
            )
            
            ratio = ratios['combined_ratio']
            A_g = advantages[g]
            
            if self.use_clipping:
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                if A_g >= 0:
                    term = torch.min(ratio * A_g, clipped_ratio * A_g)
                else:
                    term = torch.max(ratio * A_g, clipped_ratio * A_g)
                was_clipped = (ratio != clipped_ratio).float().item()
            else:
                term = ratio * A_g
                clipped_ratio = ratio
                was_clipped = 0.0
            
            # Safety check for NaN in policy term
            if torch.isnan(term) or torch.isinf(term):
                logging.warning(f"[NaN in policy term] candidate {g}: ratio={ratio.item()}, A_g={A_g.item()}")
                term = torch.tensor(0.0, device=term.device)
            
            policy_terms.append(term)
            
            metrics_per_candidate.append({
                'combined_ratio': ratio.item() if not torch.isnan(ratio) else 0.0,
                'mean_ratio_a': ratios['mean_ratio_a'].item() if not torch.isnan(ratios['mean_ratio_a']) else 0.0,
                'mean_ratio_b': ratios['mean_ratio_b'].item() if not torch.isnan(ratios['mean_ratio_b']) else 0.0,
                'mean_ratio_c': ratios['mean_ratio_c'].item() if not torch.isnan(ratios['mean_ratio_c']) else 0.0,
                'mean_ratio_d': ratios['mean_ratio_d'].item() if not torch.isnan(ratios['mean_ratio_d']) else 0.0,
                'mean_ratio_z': ratios['mean_ratio_z'].item() if not torch.isnan(ratios['mean_ratio_z']) else 0.0,
                'advantage': A_g.item(),
                'was_clipped': was_clipped,
            })
        
        policy_gradient_loss = -torch.stack(policy_terms).mean()
        
        # Safety check for policy gradient loss
        if torch.isnan(policy_gradient_loss) or torch.isinf(policy_gradient_loss):
            logging.warning(f"[NaN in policy_gradient_loss] Setting to 0.0")
            policy_gradient_loss = torch.tensor(0.0, device=policy_gradient_loss.device, requires_grad=True)
        
        # OPTIMIZATION: Reuse forward pass from above for supervised loss
        # No need to call model(*inputs) again!
        supervised_loss, loss_components = standard_loss_fn(
            fs_logits, z_logits, L_logits, all_L, true_labels
        )
        
        # Safety check for supervised loss
        if torch.isnan(supervised_loss) or torch.isinf(supervised_loss):
            logging.warning(f"[NaN in supervised_loss] loss_components: {loss_components}")
            supervised_loss = torch.tensor(0.0, device=supervised_loss.device, requires_grad=True)
        
        if was_training:
            model.train()
        
        alpha = 0.3
        total_loss = alpha * policy_gradient_loss + (1 - alpha) * supervised_loss
        
        best_idx = advantages.argmax().item()
        
        metrics = {
            'policy_gradient_loss': policy_gradient_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'total_loss': total_loss.item(),
            'alpha': alpha,
            'avg_combined_ratio': sum(m['combined_ratio'] for m in metrics_per_candidate) / G,
            'clip_fraction': sum(m['was_clipped'] for m in metrics_per_candidate) / G,
            'best_ratio_a': metrics_per_candidate[best_idx]['mean_ratio_a'],
            'best_ratio_b': metrics_per_candidate[best_idx]['mean_ratio_b'],
            'best_ratio_c': metrics_per_candidate[best_idx]['mean_ratio_c'],
            'best_ratio_d': metrics_per_candidate[best_idx]['mean_ratio_d'],
            'best_ratio_z': metrics_per_candidate[best_idx]['mean_ratio_z'],
            'best_combined_ratio': metrics_per_candidate[best_idx]['combined_ratio'],
            'best_candidate_idx': best_idx,
            'num_candidates': G,
            'use_clipping': self.use_clipping,
            'clip_epsilon': self.clip_epsilon,
            'metrics_per_candidate': metrics_per_candidate,
            # Add compatibility keys for logging
            'avg_ratio': metrics_per_candidate[best_idx]['combined_ratio'],
            'best_ratio': metrics_per_candidate[best_idx]['combined_ratio'],
            'avg_log_ratio': 0.0,  # Not computed in value ratio mode
            'best_log_ratio': 0.0,  # Not computed in value ratio mode
            'best_log_ratio_raw': 0.0,  # Not computed in value ratio mode
            'best_log_prob_new': 0.0,  # Not computed in value ratio mode
            'best_log_prob_old': 0.0,  # Not computed in value ratio mode
            'best_advantage': advantages[best_idx].item(),
            'log_ratio_clamp_fraction': 0.0,  # Not applicable in value ratio mode
            **loss_components,
        }
        
        return total_loss, metrics

    def compute_e2e_accuracy_policy_loss(
        self,
        model: nn.Module,
        inputs: Tuple,
        candidates: List[Dict],
        advantages: torch.Tensor,
        standard_loss_fn,
        true_labels: Dict,
        sample_true_labels: torch.Tensor = None,
        normalized_X: torch.Tensor = None,
        e2e_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        GRPO loss combining E2E accuracy signal WITH importance ratio correction.
        
        This is the full GRPO/PPO approach:
        1. E2E accuracy-based rewards → normalized advantages (task-aware signal)
        2. Importance ratio π_θ_new/π_θ_old (proper off-policy correction)
        3. PPO-style clipping on the ratio (stability)
        
        Policy term: clipped_ratio × A_g
        
        Where:
        - A_g is the normalized advantage (derived from e2e-based rewards in reward_computer)
        - ratio = exp(log_prob_new - log_prob_old) provides importance sampling correction
        
        The E2E accuracy metrics are computed for logging but the policy gradient
        uses the proper importance ratio formulation.
        """
        was_training = model.training
        model.eval()  # Deterministic forward pass for consistent log-prob computation
        
        G = len(candidates)
        policy_terms = []
        metrics_per_candidate = []
        
        # Maximum allowed log-ratio to prevent numerical explosion
        MAX_LOG_RATIO = 10.0  # exp(10) ≈ 22000, exp(-10) ≈ 0.000045
        
        # OPTIMIZATION: Single forward pass for all candidates
        # All candidates share the same current model output (differ only in sampled actions)
        fs_logits, z_logits, L_logits, all_L = model(*inputs)
        pred_a, pred_b, pred_c, pred_d = fs_logits
        
        # Extract ground truth
        z_labels_indices = true_labels['z_labels_indices']
        
        for g, candidate in enumerate(candidates):
            # Get sampled actions from candidate
            actions_a = candidate['actions_a']
            actions_b = candidate['actions_b']
            actions_c = candidate['actions_c']
            actions_d = candidate['actions_d']
            actions_z = candidate['actions_z']
            
            # =================================================================
            # IMPORTANCE RATIO COMPUTATION (GRPO core)
            # =================================================================
            # Compute log probability under CURRENT policy (π_θ_new)
            log_prob_new, log_prob_components = self.compute_log_probability(
                pred_a, pred_b, pred_c, pred_d, z_logits,
                actions_a, actions_b, actions_c, actions_d, actions_z
            )
            
            # Get stored log probability from sampling time (π_θ_old)
            log_prob_old = candidate['log_prob_total_old']
            
            # Compute importance ratio with clamping for numerical stability
            log_ratio_raw = log_prob_new - log_prob_old
            log_ratio = torch.clamp(log_ratio_raw, min=-MAX_LOG_RATIO, max=MAX_LOG_RATIO)
            ratio = torch.exp(log_ratio)
            
            # Per-variable ratios for debugging/logging
            log_ratio_a = torch.clamp(log_prob_components['log_prob_a'] - candidate['log_prob_a_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_b = torch.clamp(log_prob_components['log_prob_b'] - candidate['log_prob_b_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_c = torch.clamp(log_prob_components['log_prob_c'] - candidate['log_prob_c_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_d = torch.clamp(log_prob_components['log_prob_d'] - candidate['log_prob_d_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            log_ratio_z = torch.clamp(log_prob_components['log_prob_z'] - candidate['log_prob_z_old'], -MAX_LOG_RATIO, MAX_LOG_RATIO)
            
            ratio_a = torch.exp(log_ratio_a)
            ratio_b = torch.exp(log_ratio_b)
            ratio_c = torch.exp(log_ratio_c)
            ratio_d = torch.exp(log_ratio_d)
            ratio_z = torch.exp(log_ratio_z)
            
            # DEBUG: Log ratio info for first candidate
            if g == 0:
                logging.debug(f"[E2E-GRPO] Candidate {g}: log_prob_new={log_prob_new.item():.4f}, "
                             f"log_prob_old={log_prob_old.item():.4f}, ratio={ratio.item():.4f}")
                logging.debug(f"[E2E-GRPO] Per-var ratios: a={ratio_a.item():.4f}, b={ratio_b.item():.4f}, "
                             f"c={ratio_c.item():.4f}, d={ratio_d.item():.4f}, z={ratio_z.item():.4f}")
            
            # =================================================================
            # E2E ACCURACY COMPUTATION (for metrics/logging)
            # =================================================================
            # Variable-level accuracy
            correct_a = 0
            total_a = 0
            for act_a, true_a in zip(actions_a, true_labels['true_a_per_graph']):
                true_idx = true_a.argmax(dim=0)
                correct_a += (act_a == true_idx).float().sum().item()
                total_a += true_idx.numel()
            acc_a = correct_a / max(total_a, 1)
            
            # B tolerance: use 0.1 for accuracy metric (more lenient than reward)
            acc_b = (torch.abs(actions_b - true_labels['true_b']) < 0.1).float().mean().item()
            
            correct_c = 0
            total_c = 0
            for act_c, true_c in zip(actions_c, true_labels['true_c_per_graph']):
                true_idx = true_c.argmax(dim=0)
                correct_c += (act_c == true_idx).float().sum().item()
                total_c += true_idx.numel()
            acc_c = correct_c / max(total_c, 1)
            
            acc_d = ((actions_d > 0.5).float() == (true_labels['true_d'] > 0.5).float()).float().mean().item()
            acc_z = (actions_z == z_labels_indices).float().mean().item()
            
            var_accuracy = (acc_a + acc_b + acc_c + acc_d + acc_z) / 5.0
            
            # End-to-end accuracy
            if len(actions_c) > 0:
                leaf_class_preds = actions_c[0]
            else:
                leaf_class_preds = torch.zeros(4, device=actions_z.device, dtype=torch.long)
            
            N = actions_z.shape[0]
            num_leaves = min(leaf_class_preds.shape[0], 4)
            leaf_indices = torch.clamp(actions_z, 0, num_leaves - 1)
            pred_classes = leaf_class_preds[leaf_indices]
            
            if sample_true_labels is not None:
                pred_classes_cmp = pred_classes + 1
                true_classes = sample_true_labels
            else:
                pred_classes_cmp = pred_classes
                true_classes = z_labels_indices

            e2e_correct = (pred_classes_cmp == true_classes).float()
            e2e_accuracy = e2e_correct.mean().item()
            
            combined_accuracy = (1 - e2e_weight) * var_accuracy + e2e_weight * e2e_accuracy
            
            # =================================================================
            # POLICY GRADIENT TERM with importance ratio
            # =================================================================
            A_g = advantages[g]
            
            # Apply PPO-style clipping to importance ratio
            if self.use_clipping:
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                if A_g >= 0:
                    # Positive advantage: want to increase probability, clip prevents too high ratio
                    term = torch.min(ratio * A_g, clipped_ratio * A_g)
                else:
                    # Negative advantage: want to decrease probability, clip prevents too low ratio
                    term = torch.max(ratio * A_g, clipped_ratio * A_g)
                was_clipped = (ratio != clipped_ratio).float().item()
            else:
                term = ratio * A_g
                clipped_ratio = ratio
                was_clipped = 0.0
            
            # Safety check for NaN/Inf
            if torch.isnan(term) or torch.isinf(term):
                logging.warning(f"[E2E-GRPO] NaN/Inf in policy term for candidate {g}, using 0.0")
                term = torch.tensor(0.0, device=A_g.device, requires_grad=True)
            
            policy_terms.append(term)
            
            # Track if log-ratio was clamped (indicates potential numerical issues)
            log_ratio_clamped = (log_ratio_raw.abs() > MAX_LOG_RATIO).float().item()
            
            metrics_per_candidate.append({
                # Importance ratio metrics
                'ratio': ratio.item(),
                'clipped_ratio': clipped_ratio.item() if isinstance(clipped_ratio, torch.Tensor) else clipped_ratio,
                'log_prob_new': log_prob_new.item(),
                'log_prob_old': log_prob_old.item(),
                'log_ratio': log_ratio.item(),
                'log_ratio_raw': log_ratio_raw.item(),
                'log_ratio_clamped': log_ratio_clamped,
                'was_clipped': was_clipped,
                # Per-variable ratios
                'ratio_a': ratio_a.item(),
                'ratio_b': ratio_b.item(),
                'ratio_c': ratio_c.item(),
                'ratio_d': ratio_d.item(),
                'ratio_z': ratio_z.item(),
                # E2E accuracy metrics
                'combined_accuracy': combined_accuracy,
                'var_accuracy': var_accuracy,
                'e2e_accuracy': e2e_accuracy,
                'acc_a': acc_a,
                'acc_b': acc_b,
                'acc_c': acc_c,
                'acc_d': acc_d,
                'acc_z': acc_z,
                'advantage': A_g.item(),
            })
        
        # GRPO loss: -1/G * Σ_g [ clipped_ratio_g * A_g ]
        policy_gradient_loss = -torch.stack(policy_terms).mean()
        
        if torch.isnan(policy_gradient_loss) or torch.isinf(policy_gradient_loss):
            logging.warning(f"[E2E-GRPO] NaN in policy_gradient_loss, setting to 0.0")
            policy_gradient_loss = torch.tensor(0.0, device=policy_gradient_loss.device, requires_grad=True)
        
        # Supervised loss for stability (reuse cached forward pass)
        supervised_loss, loss_components = standard_loss_fn(
            fs_logits, z_logits, L_logits, all_L, true_labels
        )
        
        if torch.isnan(supervised_loss) or torch.isinf(supervised_loss):
            logging.warning(f"[E2E-GRPO] NaN in supervised_loss")
            supervised_loss = torch.tensor(0.0, device=supervised_loss.device, requires_grad=True)
        
        if was_training:
            model.train()
        
        # Combine PG loss and supervised loss
        # alpha = 0.5 means 50% policy gradient, 50% supervised
        alpha = 0.5
        total_loss = alpha * policy_gradient_loss + (1 - alpha) * supervised_loss
        
        # Aggregate metrics
        best_idx = advantages.argmax().item()
        avg_ratio = sum(m['ratio'] for m in metrics_per_candidate) / G
        avg_log_ratio = sum(m['log_ratio'] for m in metrics_per_candidate) / G
        clip_fraction = sum(m['was_clipped'] for m in metrics_per_candidate) / G
        log_ratio_clamp_fraction = sum(m['log_ratio_clamped'] for m in metrics_per_candidate) / G
        
        metrics = {
            'policy_gradient_loss': policy_gradient_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'total_loss': total_loss.item(),
            'alpha': alpha,
            'loss_method': 'e2e_accuracy',
            'e2e_weight': e2e_weight,
            # Importance ratio statistics
            'avg_ratio': avg_ratio,
            'avg_log_ratio': avg_log_ratio,
            'best_ratio': metrics_per_candidate[best_idx]['ratio'],
            'best_log_ratio': metrics_per_candidate[best_idx]['log_ratio'],
            'best_log_ratio_raw': metrics_per_candidate[best_idx]['log_ratio_raw'],
            'best_log_prob_new': metrics_per_candidate[best_idx]['log_prob_new'],
            'best_log_prob_old': metrics_per_candidate[best_idx]['log_prob_old'],
            # Per-variable ratios
            'best_ratio_a': metrics_per_candidate[best_idx]['ratio_a'],
            'best_ratio_b': metrics_per_candidate[best_idx]['ratio_b'],
            'best_ratio_c': metrics_per_candidate[best_idx]['ratio_c'],
            'best_ratio_d': metrics_per_candidate[best_idx]['ratio_d'],
            'best_ratio_z': metrics_per_candidate[best_idx]['ratio_z'],
            # Clipping statistics
            'use_clipping': self.use_clipping,
            'clip_epsilon': self.clip_epsilon,
            'clip_fraction': clip_fraction,
            'log_ratio_clamp_fraction': log_ratio_clamp_fraction,
            # E2E accuracy statistics
            'avg_combined_accuracy': sum(m['combined_accuracy'] for m in metrics_per_candidate) / G,
            'avg_var_accuracy': sum(m['var_accuracy'] for m in metrics_per_candidate) / G,
            'avg_e2e_accuracy': sum(m['e2e_accuracy'] for m in metrics_per_candidate) / G,
            'best_combined_accuracy': metrics_per_candidate[best_idx]['combined_accuracy'],
            'best_e2e_accuracy': metrics_per_candidate[best_idx]['e2e_accuracy'],
            'best_acc_a': metrics_per_candidate[best_idx]['acc_a'],
            'best_acc_b': metrics_per_candidate[best_idx]['acc_b'],
            'best_acc_c': metrics_per_candidate[best_idx]['acc_c'],
            'best_acc_d': metrics_per_candidate[best_idx]['acc_d'],
            'best_acc_z': metrics_per_candidate[best_idx]['acc_z'],
            'best_advantage': advantages[best_idx].item(),
            'best_candidate_idx': best_idx,
            'num_candidates': G,
            'metrics_per_candidate': metrics_per_candidate,
            **loss_components,
        }
        
        return total_loss, metrics


class DatasetGroupNormalizer:
    """
    Normalizes rewards/losses within dataset groups for GRPO.
    
    GRPO Technique 2: Baseline Normalization
    - Different datasets have different difficulty levels and sizes
    - Without normalization, larger/easier datasets dominate training
    - We normalize rewards using running statistics per dataset
    
    Key insight: Each dataset should contribute equally to policy updates,
    regardless of:
    1. Dataset size (number of samples)
    2. Dataset difficulty (baseline reward level)
    3. Dataset variance (spread of rewards)
    """
    
    def __init__(
        self,
        norm_eps: float = 1e-8,
        momentum: float = 0.99,  # EMA momentum for running stats
        warmup_samples: int = 10,  # Minimum samples before using running stats
    ):
        self.norm_eps = norm_eps
        self.momentum = momentum
        self.warmup_samples = warmup_samples
        self.dataset_stats = {}  # Running statistics per dataset
    
    @staticmethod
    def extract_dataset_name(dataset_path: str) -> str:
        """
        Extract a meaningful dataset identifier from the file path.
        
        Examples:
            '/path/to/iris_train.npz' -> 'iris'
            '/path/to/wine_quality_0.npz' -> 'wine_quality'
            '/hpc_datasets/train/breast_cancer_5.npz' -> 'breast_cancer'
        """
        import os
        import re
        
        # Get filename without extension
        filename = os.path.basename(dataset_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Remove trailing numbers/indices (e.g., 'iris_0' -> 'iris', 'wine_5' -> 'wine')
        # Also handles 'dataset_train_0' -> 'dataset_train'
        name_clean = re.sub(r'_\d+$', '', name_without_ext)
        
        return name_clean
    
    def get_dataset_names_from_batch(self, states: List[Dict]) -> List[str]:
        """
        Extract dataset names from batch of states.
        
        Each state has meta_info with dataset_path.
        """
        dataset_names = []
        for s in states:
            meta_info = s.get('meta_info', {})
            dataset_path = meta_info.get('dataset_path', 'unknown')
            dataset_name = self.extract_dataset_name(dataset_path)
            dataset_names.append(dataset_name)
        return dataset_names
    
    def update_running_stats(self, dataset_name: str, reward: float):
        """
        Update running statistics for a dataset using exponential moving average.
        
        Uses Welford's algorithm for online variance computation,
        combined with EMA for smooth updates.
        """
        if dataset_name not in self.dataset_stats:
            self.dataset_stats[dataset_name] = {
                'mean': reward,
                'var': 0.0,
                'count': 1,
                'ema_mean': reward,
                'ema_var': 1.0,
            }
            return
        
        stats = self.dataset_stats[dataset_name]
        n = stats['count']
        old_mean = stats['mean']
        
        # Welford's online algorithm for exact statistics
        n += 1
        delta = reward - old_mean
        new_mean = old_mean + delta / n
        delta2 = reward - new_mean
        new_var = ((n - 1) * stats['var'] + delta * delta2) / n
        
        stats['count'] = n
        stats['mean'] = new_mean
        stats['var'] = new_var
        
        # EMA for smooth, recent-biased statistics
        m = self.momentum
        stats['ema_mean'] = m * stats['ema_mean'] + (1 - m) * reward
        stats['ema_var'] = m * stats['ema_var'] + (1 - m) * (reward - stats['ema_mean']) ** 2
    
    def get_dataset_baseline(self, dataset_name: str) -> Tuple[float, float]:
        """
        Get normalization parameters (mean, std) for a dataset.
        
        Uses EMA statistics if enough samples have been seen,
        otherwise falls back to exact statistics or defaults.
        """
        if dataset_name not in self.dataset_stats:
            return 0.0, 1.0
        
        stats = self.dataset_stats[dataset_name]
        
        if stats['count'] < self.warmup_samples:
            # Not enough samples, use exact statistics
            if stats['count'] < 2:
                return stats['mean'], 1.0
            std = math.sqrt(stats['var'] + self.norm_eps)
            return stats['mean'], max(std, self.norm_eps)
        
        # Use EMA statistics for stable, recent-biased normalization
        ema_std = math.sqrt(stats['ema_var'] + self.norm_eps)
        return stats['ema_mean'], max(ema_std, self.norm_eps)
    
    def normalize_rewards_by_dataset(
        self,
        rewards: List[torch.Tensor],  # [G] rewards per candidate
        dataset_name: str,
    ) -> torch.Tensor:
        """
        Normalize rewards for a single sample using dataset-specific baseline.
        
        This is the PER-SAMPLE normalization using the dataset's running stats.
        
        normalized_reward = (reward - dataset_mean) / dataset_std
        
        This ensures rewards from different datasets are on the same scale.
        """
        rewards_tensor = torch.stack(rewards)  # [G]
        
        # Get dataset-specific baseline
        baseline_mean, baseline_std = self.get_dataset_baseline(dataset_name)
        
        # Normalize using dataset baseline
        # This puts rewards on a common scale across datasets
        normalized = (rewards_tensor - baseline_mean) / baseline_std
        
        return normalized
    
    def normalize_advantages_within_group(
        self,
        rewards: List[torch.Tensor],  # [G] rewards per candidate
    ) -> torch.Tensor:
        """
        Compute advantages by normalizing within the candidate group.
        
        This is the standard GRPO advantage computation:
        A_i = (r_i - mean(r)) / std(r)
        
        This creates relative comparisons within the group.
        """
        rewards_tensor = torch.stack(rewards)  # [G]
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        
        advantages = (rewards_tensor - mean_r) / (std_r + self.norm_eps)
        return advantages
    
    def compute_normalized_advantages(
        self,
        rewards: List[torch.Tensor],  # [G] rewards per candidate
        dataset_name: str,
        update_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full GRPO advantage computation with dataset normalization.
        
        Two-level normalization:
        1. Dataset-level: Scale rewards to common range using running stats
        2. Group-level: Compute relative advantages within candidate group
        
        This ensures:
        - Different datasets contribute equally (dataset normalization)
        - Learning signal is relative quality (group normalization)
        """
        rewards_tensor = torch.stack(rewards)  # [G]
        
        # Update running statistics with mean reward from this group
        if update_stats:
            mean_reward = rewards_tensor.mean().item()
            self.update_running_stats(dataset_name, mean_reward)
        
        # Get dataset baseline for normalization
        baseline_mean, baseline_std = self.get_dataset_baseline(dataset_name)
        
        # === TWO-LEVEL NORMALIZATION ===
        # Level 1: Dataset-level normalization — scale rewards to a common range
        # This ensures rewards from different datasets (e.g., glass vs small_toy)
        # are on a comparable scale before computing group-relative advantages.
        normalized_rewards = (rewards_tensor - baseline_mean) / max(baseline_std, self.norm_eps)
        
        # Level 2: Group-level normalization — compute relative advantages within candidate group
        # Standard GRPO: A_i = (r_i - mean(r)) / std(r) applied to dataset-normalized rewards
        mean_r = normalized_rewards.mean()
        std_r = normalized_rewards.std()
        advantages = (normalized_rewards - mean_r) / (std_r + self.norm_eps)
        
        # For logging/debugging
        details = {
            'dataset_name': dataset_name,
            'dataset_baseline_mean': baseline_mean,
            'dataset_baseline_std': baseline_std,
            'dataset_sample_count': self.dataset_stats.get(dataset_name, {}).get('count', 0),
            'group_mean_reward': mean_r.item(),
            'group_std_reward': std_r.item(),
        }
        
        return advantages, details
    
    def get_all_dataset_stats(self) -> Dict[str, Dict]:
        """Return statistics for all datasets (for logging)."""
        return {
            name: {
                'count': stats['count'],
                'mean': stats['mean'],
                'std': math.sqrt(stats['var'] + self.norm_eps),
                'ema_mean': stats['ema_mean'],
                'ema_std': math.sqrt(stats['ema_var'] + self.norm_eps),
            }
            for name, stats in self.dataset_stats.items()
        }
    
    def normalize_by_group(
        self,
        losses: torch.Tensor,  # [B] per-sample losses
        dataset_names: List[str],  # [B] dataset identifiers
    ) -> torch.Tensor:
        """
        Legacy method: Normalize losses within each dataset group in a batch.
        
        loss_norm = (loss - mean_group) / std_group
        """
        device = losses.device
        normalized = torch.zeros_like(losses)
        
        # Group by dataset
        unique_datasets = list(set(dataset_names))
        
        for dataset in unique_datasets:
            mask = torch.tensor([n == dataset for n in dataset_names], device=device)
            group_losses = losses[mask]
            
            if group_losses.numel() >= 2:
                mean_loss = group_losses.mean()
                std_loss = group_losses.std()
                normalized[mask] = (group_losses - mean_loss) / (std_loss + self.norm_eps)
            else:
                # Single sample, use global normalization
                normalized[mask] = group_losses - losses.mean()
        
        return normalized


def train_and_evaluate_grpo(
    train_problems_datasets: List[str],
    train_problems_outputs: List[str],
    train_problems_linear_feats: List[str], 
    valid_problems_datasets: List[str],
    valid_problems_outputs: List[str],
    valid_problems_linear_feats: List[str], 
    device: torch.device,
    learning_rate: float,
    model_dir: str,
    decay_steps: int,
    num_train_steps: int,
    num_train_run_steps: int,
    eval_every_steps: int,
    eval_steps: int,
    grad_clip_norm: float,
    model_config: ConfigDict,
    grpo_config: GRPOConfig,
    resume_ckpt: str = '',
    resume_training: bool = False,
    use_staged_training: bool = False,
    phase1_epochs: int = 100,
    phase2_freeze_encoder: bool = True,
    phase2_freeze_heads: list = None,
    phase2_loss_weights: dict = None,
    routing_tau_start: float = 2.0,
    routing_tau_end: float = 0.1,
):
    """
    Training function with GRPO techniques.
    
    Staged Training Mode (use_staged_training=True):
        Phase 1 (0-phase1_epochs): Normal training with all variables
        Phase 2 (phase1_epochs-num_train_steps): Freeze encoder + selected heads, train only remaining
        
    Args:
        use_staged_training: Enable two-phase training
        phase1_epochs: Number of epochs for Phase 1 (normal training)
        phase2_freeze_encoder: If True, freeze encoder in Phase 2
        phase2_freeze_heads: List of heads to freeze in Phase 2 (e.g., ['a', 'c', 'd'])
        phase2_loss_weights: Dict of loss weights for Phase 2 (e.g., {'a': 0.0, 'b': 1.0, 'c': 0.0, 'd': 0.0})
    """
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "none") 
    
    logging.info(f'Loading data...')
    logging.info(f'GRPO Config: {grpo_config}')

    # Determine dataset size from NUM_FILES env variable
    num_files = int(os.getenv('NUM_FILES', '100'))  # Default to 100

    # Check if BATCH_SIZE is explicitly set via environment variable
    batch_size_env = os.getenv('BATCH_SIZE')
    if batch_size_env is not None:
        batch_size = int(batch_size_env)
        logging.info(f"[BATCH_SIZE] Using BATCH_SIZE={batch_size} from environment variable")
    else:
        # Auto-calculate batch_size based on NUM_FILES
        if num_files == 100:
            batch_size = 10  # 10 batches
        elif num_files == 250:
            batch_size = 25  # 10 batches
        elif num_files == 500:
            batch_size = 50  # 10 batches
        elif num_files == 1000:
            batch_size = 50  # 20 batches
        else:
            # Custom: aim for ~10-20 batches
            batch_size = max(10, num_files // 15)
            logging.info(f"[CUSTOM] NUM_FILES={num_files} not standard; using batch_size={batch_size}")

    logging.info(f"[NUM_FILES={num_files}] Loading {num_files} files with batch_size={batch_size}")

    # =========================================================================
    # Balanced per-dataset sampling: pick files_per_dataset from EACH dataset
    # so that both glass (N=214) and small_toy (N=6) appear in training.
    # =========================================================================
    datasets_env = os.getenv("DATASETS", "glass,small_toy")
    dataset_names = [d.strip() for d in datasets_env.split(',')]
    n_datasets = len(dataset_names)
    files_per_dataset = num_files // n_datasets  # e.g., 20 // 2 = 10

    def select_balanced_files(file_list, dataset_names, files_per_dataset):
        """Select files_per_dataset files from EACH dataset in file_list."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for f in file_list:
            for ds in dataset_names:
                if f'/{ds}/' in f:
                    buckets[ds].append(f)
                    break
        selected = []
        for ds in dataset_names:
            selected.extend(buckets[ds][:files_per_dataset])
        logging.info(f"[BALANCED] Selected {len(selected)} files: " +
                     ", ".join(f"{ds}={len(buckets[ds][:files_per_dataset])}" for ds in dataset_names))
        return selected

    train_datasets_sel = select_balanced_files(train_problems_datasets[0][0], dataset_names, files_per_dataset)
    train_outputs_sel  = select_balanced_files(train_problems_outputs[0][0],  dataset_names, files_per_dataset)
    train_linear_sel   = select_balanced_files(train_problems_linear_feats[0][0], dataset_names, files_per_dataset)

    train_data_loaders = two_stage_data_utils3.get_dataset(
        train_datasets_sel,
        train_outputs_sel,
        train_linear_sel,
        batch_size=10
    )
    logging.info(f'Train datasets loaded: {len(train_data_loaders)} batches '
                 f'({len(train_datasets_sel)} files, batch_size=10)')

    # Validation: 5 files per dataset (10 total) for balanced eval
    valid_per_dataset = max(1, 10 // n_datasets)
    valid_datasets_sel = select_balanced_files(valid_problems_datasets[0][0], dataset_names, valid_per_dataset)
    valid_outputs_sel  = select_balanced_files(valid_problems_outputs[0][0],  dataset_names, valid_per_dataset)
    valid_linear_sel   = select_balanced_files(valid_problems_linear_feats[0][0], dataset_names, valid_per_dataset)

    valid_data_loaders = two_stage_data_utils3.get_dataset(
        valid_datasets_sel,
        valid_outputs_sel,
        valid_linear_sel,
        batch_size=1
    )
    logging.info(f'Valid datasets loaded: {len(valid_data_loaders)}')

    model = two_stage_gps_small.get_model(**model_config.params).to(device)
    torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        logging.info(f"{num_gpus} GPUs detected: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    # === Setup optimizers with SEPARATE OPTIMIZER for B (Option 2) ===
    # Option 2: Separate optimizer for B with 10x lower learning rate
    # Rationale: B (thresholds) is a continuous variable that needs slower convergence
    #            Separating it prevents gradient conflicts with A/C/D (categorical variables)
    target = model.module if isinstance(model, nn.DataParallel) else model
    
    # Disable weight decay to allow memorization/overfitting on the toy set
    # ENCODER: All parameters EXCEPT b_head
    # NOTE: Using simplified model with explicit tree routing (no learned decoder)
    encoder_params = [
        {'params': target.pos_encoding.parameters(), 'weight_decay': 0.0},
        {'params': target.feature_attention.parameters(), 'weight_decay': 0.0},
        {'params': target.first_linear.parameters(), 'weight_decay': 0.0},
        {'params': target.constraint_linear.parameters(), 'weight_decay': 0.0},
        {'params': target.first_layers.parameters(), 'weight_decay': 0.0},
        {'params': target.a_head.parameters(), 'weight_decay': 0.0},
        # NOTE: b_head REMOVED - will use separate optimizer
        {'params': target.c_head.parameters(), 'weight_decay': 0.0},
        {'params': target.d_head.parameters(), 'weight_decay': 0.0},
    ]

    # B-specific parameters: only b_head
    b_params = [
        {'params': target.b_head.parameters(), 'weight_decay': 0.0},
    ]

    # NOTE: decoder_params removed - simplified model uses explicit tree routing
    # All learning happens in encoder heads (a_head, b_head, c_head, d_head)
    # decoder_forward() just computes Z from A, B, X using differentiable path probabilities

    # Encoder optimizer: all parameters except b_head
    encoder_optimizer = optim.AdamW(encoder_params, lr=learning_rate)

    # SEPARATE optimizer for B with lower learning rate
    # B (thresholds) needs slower learning than categorical variables (A, C, D)
    # Gradient explosion fixed by clamping z_probs before log in decoder_forward
    # (max gradient now ~100 instead of 1e8)
    B_LR_SCALE = 0.01  # 100x lower than base LR (was 0.001 before z_probs clamp fix)
    b_optimizer = optim.AdamW(b_params, lr=learning_rate * B_LR_SCALE)
    logging.info(f"[B-HEAD] Using separate LR: {learning_rate * B_LR_SCALE:.6f} (base_lr * {B_LR_SCALE})")

    # Loss weight optimizer
    loss_weight_optimizer = optim.AdamW([target.log_loss_weights], lr=min(1e-4, learning_rate * 0.05))

    # NOTE: decoder_optimizer removed - simplified model uses explicit tree routing
    # All learning happens in encoder (GPS layers + output heads a, b, c, d)

    # LR schedulers: step decay at epoch 300 for 400-epoch run
    # Reduces LR from 0.001 → 0.0005 (2x reduction) to stabilize late training
    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=300, gamma=0.5)
    b_scheduler = optim.lr_scheduler.StepLR(b_optimizer, step_size=300, gamma=0.5)
    
    # Initialize loss weights - prioritize first-stage, include E2E and L losses
    with torch.no_grad():
        # First-stage (A+B+C+D combined) = 9.0, E2E = 5.0, L = 1.0
        fixed_values = [math.log(9.0), math.log(5.0), math.log(1.0)]
        target.log_loss_weights.data = torch.tensor(fixed_values, device=target.log_loss_weights.device)
        # ENABLE gradients for learnable weights
        target.log_loss_weights.requires_grad = True
    
    # Keep first-stage sum constant to stabilize contributions
    FS_WEIGHT_SUM = float(math.exp(fixed_values[0]))
    prior_log_weights = torch.tensor([fixed_values[0]], device=device)
    
    # === Helper function for staged training ===
    def freeze_parameters(parameters):
        """Freeze a list of parameters."""
        for param in parameters:
            param.requires_grad = False
    
    def freeze_model_components_phase2(freeze_encoder=False, freeze_heads=None):
        """Freeze specific model components for Phase 2."""
        if freeze_encoder:
            freeze_parameters(target.pos_encoding.parameters())
            freeze_parameters(target.first_linear.parameters())
            freeze_parameters(target.first_layers.parameters())
            # NOTE: second_linear and second_layers removed in simplified model
            logging.info("🔒 [PHASE 2] Encoder frozen")
        
        if freeze_heads:
            head_map = {
                'a': target.a_head,
                'b': target.b_head,
                'c': target.c_head,
                'd': target.d_head,
            }
            for head_name in freeze_heads:
                if head_name in head_map:
                    freeze_parameters(head_map[head_name].parameters())
                    logging.info(f"🔒 [PHASE 2] {head_name}_head frozen")

    # === Initialize GRPO components ===
    # Shared log_std for Gaussian (b variable)
    b_log_std = -1.0  # log(std) ≈ -1 → std ≈ 0.37
    
    candidate_sampler = GRPOCandidateSampler(
        num_candidates=grpo_config.num_candidates,
        temperature=grpo_config.temperature,
        sampling_dropout=grpo_config.sampling_dropout,
        b_log_std=b_log_std,
    )
    
    reward_computer = GRPORewardComputer(
        accuracy_weight=grpo_config.accuracy_weight,
        infeasibility_weight=grpo_config.infeasibility_weight,
    )
    
    # GRPO Loss Computer with optional clipping
    # Set use_clipping=True to enable PPO-style clipping, False for vanilla GRPO
    use_clipping = getattr(grpo_config, 'use_clipping', False)
    clip_epsilon = getattr(grpo_config, 'clip_epsilon', 0.2)
    
    loss_computer = GRPOLossComputer(
        device=device,
        b_log_std=b_log_std,
        use_clipping=use_clipping,
        clip_epsilon=clip_epsilon,
        temperature=grpo_config.temperature,  # CRITICAL: Must match sampler temperature
    )
    dataset_normalizer = DatasetGroupNormalizer()

    logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0):.2f}")
    logging.info(f"GPU memory reserved: {torch.cuda.memory_reserved(0):.2f}")

    def compute_standard_loss(fs_logits, z_logits, L_logits, all_L, true_labels):
        """Standard supervised loss computation."""
        pred_a, pred_b, pred_c, pred_d = fs_logits
        true_a_per_graph = true_labels['true_a_per_graph']
        true_c_per_graph = true_labels['true_c_per_graph']
        true_b = true_labels['true_b']
        true_d = true_labels['true_d']
        z_labels_indices = true_labels['z_labels_indices']
        L_labels_per_batch = true_labels['L_labels_per_batch']
        
        # Loss a: cross-entropy per decision node
        loss_a = 0
        num_a_cols = 0
        for pred_a_graph, true_a_graph in zip(pred_a, true_a_per_graph):
            for col in range(pred_a_graph.shape[1]):
                loss_a += F.cross_entropy(
                    pred_a_graph[:, col].unsqueeze(0),
                    true_a_graph[:, col].argmax().unsqueeze(0)
                )
                num_a_cols += 1
        loss_a = loss_a / max(num_a_cols, 1)

        # Entropy regularization on A to encourage sharp one-hot feature selection
        entropy_a = 0
        for pred_a_graph in pred_a:
            a_probs = F.softmax(pred_a_graph, dim=0)  # [P, Td]
            # Entropy per column (decision node), averaged
            col_entropy = -(a_probs * torch.log(a_probs + 1e-8)).sum(dim=0)  # [Td]
            entropy_a += col_entropy.mean()
        entropy_a = entropy_a / max(len(pred_a), 1)
        # Keep ENTROPY_WEIGHT_A = 0.0 (temperature annealing handles entropy reduction)
        # Apply 10x amplifier to cross-entropy to compensate for 40x weight disparity with E2E loss
        # B already has 5x amplifier; A's task (selecting 1 of P features) is harder
        loss_a = loss_a * 10.0

        # Loss b: MSE with 5× amplification to compete with discrete variable gradients
        # B is split thresholds: continuous variable that's harder to learn than categorical A, C, D
        # MSE provides strong signal for threshold learning (gradient = 2*error)
        # Amplified from 0.1× to 5.0× because:
        #   - Discrete vars (A/C/D) achieve 60-87% accuracy with cross-entropy losses
        #   - B stuck at 20% accuracy with weak MSE×0.1 signal
        #   - Phase 2 analysis shows other vars improve while B stagnates → B needs stronger signal
        loss_b = F.mse_loss(pred_b, true_b, reduction='mean')

        # Loss c: cross-entropy per leaf node
        loss_c = 0
        num_c_cols = 0
        for pred_c_graph, true_c_graph in zip(pred_c, true_c_per_graph):
            for col in range(pred_c_graph.shape[1]):
                loss_c += F.cross_entropy(
                    pred_c_graph[:, col].unsqueeze(0),
                    true_c_graph[:, col].argmax().unsqueeze(0)
                )
                num_c_cols += 1
        loss_c = loss_c / max(num_c_cols, 1)

        # Entropy regularization on C to encourage sharp one-hot class assignment
        entropy_c = 0
        for pred_c_graph in pred_c:
            c_probs = F.softmax(pred_c_graph, dim=0)  # [K, Tl]
            # Entropy per column (leaf node), averaged
            col_entropy = -(c_probs * torch.log(c_probs + 1e-8)).sum(dim=0)  # [Tl]
            entropy_c += col_entropy.mean()
        entropy_c = entropy_c / max(len(pred_c), 1)
        loss_c = loss_c + ENTROPY_WEIGHT_C * entropy_c

        # Loss d: BCE - SCALED DOWN by 0.1 to prevent instability
        # D is binary activation: BCE can explode on hard examples
        # Scaling down allows other losses (categorical A, C) to drive learning
        loss_d = F.binary_cross_entropy_with_logits(pred_d, true_d, reduction='mean') * 0.1
        
        # Unify tree-structure losses into first_stage_loss
        first_stage_loss = loss_a + loss_b + loss_c + loss_d
        
        # === E2E Loss (Replaces Z Loss) ===
        # Match inference: use hard argmax leaf + hard class assignment in forward pass,
        # but keep gradients via straight-through estimators.
        # z_logits: [N, Tl] unnormalized log probs
        # pred_c: List of [K, Tl] logits
        e2e_loss = torch.tensor(0.0, device=z_logits.device)
        sample_true_labels = true_labels.get('sample_true_labels')
        e2e_diag = {
            'total_samples': 0,
            'correct_samples': 0,
            'floor_hits': 0,
            'route_correct': 0,
            'route_total': 0,
            'leaf_correct': 0,
            'leaf_total': 0,
            'route_only_errors': 0,
            'leaf_only_errors': 0,
            'both_errors': 0,
            'other_errors': 0,
            'true_prob_sum': 0.0,
            'z_margin_sum': 0.0,
            'z_margin_count': 0,
            'c_margin_sum': 0.0,
            'c_margin_count': 0,
        }
        
        if sample_true_labels is not None:
            true_labels_idx = sample_true_labels.long() - 1
            
            # Soft leaf assignment for E2E loss — use continuous probabilities
            # REMOVED second STE (z_hard + z_soft - z_soft.detach()) which was blocking
            # gradient flow to B routing parameters. Z margin was frozen at 0.9612.
            # z_soft provides differentiable gradient path: E2E loss → softmax → log → clamp → STE → B
            z_soft = F.softmax(z_logits, dim=-1)  # [total_N, Tl] — continuous, differentiable
            z_hard_idx = z_soft.argmax(dim=-1)    # kept for diagnostics only

            if z_soft.shape[1] >= 2:
                z_top2 = torch.topk(z_soft, k=2, dim=-1).values
                z_margins = z_top2[:, 0] - z_top2[:, 1]
                e2e_diag['z_margin_sum'] += z_margins.sum().item()
                e2e_diag['z_margin_count'] += z_margins.numel()
            
            # To handle multiple graphs, we process per graph
            z_labels_list = true_labels.get('z_labels_list', [])
            if len(z_labels_list) > 0 and len(pred_c) == len(z_labels_list):
                expected_class_probs_list = []
                z_offset = 0
                for graph_idx, (pred_c_graph, z_lbl) in enumerate(zip(pred_c, z_labels_list)):
                    N_i = z_lbl.numel() // 4
                    if N_i > 0:
                        true_labels_idx_i = true_labels_idx[z_offset:z_offset + N_i]
                        z_pred_i = z_hard_idx[z_offset:z_offset + N_i]
                        z_probs_i = z_soft[z_offset:z_offset + N_i]  # [N_i, Tl] — soft routing for gradient flow
                        c_soft_i = F.softmax(pred_c_graph, dim=0)    # [K, Tl]
                        if c_soft_i.shape[0] >= 2:
                            c_top2 = torch.topk(c_soft_i, k=2, dim=0).values
                            c_margins = c_top2[0] - c_top2[1]
                            e2e_diag['c_margin_sum'] += c_margins.sum().item()
                            e2e_diag['c_margin_count'] += c_margins.numel()

                        c_hard_idx = c_soft_i.argmax(dim=0)
                        c_hard = F.one_hot(c_hard_idx, num_classes=c_soft_i.shape[0]).float().transpose(0, 1)
                        c_st = c_hard + (c_soft_i - c_soft_i.detach())

                        if graph_idx < len(true_c_per_graph):
                            true_c_idx = true_c_per_graph[graph_idx].argmax(dim=0).long()
                            leaf_dim = min(c_hard_idx.numel(), true_c_idx.numel())
                            if leaf_dim > 0:
                                e2e_diag['leaf_correct'] += (c_hard_idx[:leaf_dim] == true_c_idx[:leaf_dim]).sum().item()
                                e2e_diag['leaf_total'] += leaf_dim

                        # For each sample, prob of class k is sum_leaf(z * c)
                        exp_class_probs_raw_i = torch.einsum('nl,kl->nk', z_probs_i, c_st)
                        true_class_probs_raw_i = exp_class_probs_raw_i.gather(
                            1, true_labels_idx_i.unsqueeze(1)
                        ).squeeze(1)
                        e2e_diag['floor_hits'] += (true_class_probs_raw_i < E2E_PROB_FLOOR).sum().item()
                        e2e_diag['true_prob_sum'] += true_class_probs_raw_i.sum().item()

                        exp_class_probs_i = exp_class_probs_raw_i.clamp_min(E2E_PROB_FLOOR)

                        if c_hard_idx.numel() > 0:
                            z_pred_clamped_i = z_pred_i.clamp(0, c_hard_idx.numel() - 1)
                            pred_classes_i = c_hard_idx[z_pred_clamped_i]
                            e2e_correct_i = pred_classes_i == true_labels_idx_i
                            e2e_diag['correct_samples'] += e2e_correct_i.sum().item()
                            e2e_diag['total_samples'] += e2e_correct_i.numel()

                            if z_labels_indices is not None and z_labels_indices.numel() >= z_offset + N_i:
                                z_true_i = z_labels_indices[z_offset:z_offset + N_i].long()
                                route_correct_i = z_pred_i == z_true_i
                                e2e_diag['route_correct'] += route_correct_i.sum().item()
                                e2e_diag['route_total'] += route_correct_i.numel()

                                z_true_clamped_i = z_true_i.clamp(0, c_hard_idx.numel() - 1)
                                c_true_route_i = c_hard_idx[z_true_clamped_i]
                                c_true_route_correct_i = c_true_route_i == true_labels_idx_i

                                incorrect_i = ~e2e_correct_i
                                route_only_i = incorrect_i & (~route_correct_i) & c_true_route_correct_i
                                leaf_only_i = incorrect_i & route_correct_i & (~c_true_route_correct_i)
                                both_i = incorrect_i & (~route_correct_i) & (~c_true_route_correct_i)
                                explained_i = route_only_i | leaf_only_i | both_i
                                other_i = incorrect_i & (~explained_i)

                                e2e_diag['route_only_errors'] += route_only_i.sum().item()
                                e2e_diag['leaf_only_errors'] += leaf_only_i.sum().item()
                                e2e_diag['both_errors'] += both_i.sum().item()
                                e2e_diag['other_errors'] += other_i.sum().item()
                        
                        # Apply log to get log_probs with clamped floor
                        log_exp_class_probs_i = torch.log(exp_class_probs_i)
                        expected_class_probs_list.append(log_exp_class_probs_i)
                        
                        z_offset += N_i
                
                if expected_class_probs_list:
                    log_exp_class_probs = torch.cat(expected_class_probs_list, dim=0)  # [total_N, K]
                    # cross entropy using NLLLoss since we already did log expected probs
                    e2e_loss = F.nll_loss(log_exp_class_probs, true_labels_idx)
            else:
                logging.warning("[E2E Loss] Could not compute E2E loss: graph count mismatch")
        else:
            logging.debug("[E2E Loss] sample_true_labels not available - E2E loss set to 0")

        # Loss L
        L_loss = F.smooth_l1_loss(all_L, L_labels_per_batch, reduction='mean')
        
        # Note: entropy_a and entropy_c are computed earlier at lines 2861-2868 and 2892-2899
        # and already added to loss_a and loss_c respectively

        # Get model weights with normalization
        raw_weights = torch.exp(target.log_loss_weights)
        # target.log_loss_weights is now 3 elements: [FS, E2E, L]
        # Keep first_stage weight at FS_WEIGHT_SUM, to ensure stability
        fs_weight = raw_weights[0] / (raw_weights[0] + 1e-8) * FS_WEIGHT_SUM
        weights = torch.cat([fs_weight.unsqueeze(0), raw_weights[1:3]], dim=0)
        
        # Prior regularization to prevent degenerate weight drift
        lambda_w = 5e-4
        weight_reg = F.mse_loss(target.log_loss_weights[0:1], prior_log_weights)
        
        total_loss = (
            weights[0] * first_stage_loss +
            weights[1] * e2e_loss +
            weights[2] * L_loss
        ) + lambda_w * weight_reg

        e2e_total = max(e2e_diag['total_samples'], 1)
        e2e_error_total = max(e2e_diag['total_samples'] - e2e_diag['correct_samples'], 0)
        e2e_error_denom = max(e2e_error_total, 1)
        
        loss_components = {
            'loss_a': loss_a.item() if torch.is_tensor(loss_a) else loss_a,
            'loss_b': loss_b.item() if torch.is_tensor(loss_b) else loss_b,
            'loss_c': loss_c.item() if torch.is_tensor(loss_c) else loss_c,
            'loss_d': loss_d.item() if torch.is_tensor(loss_d) else loss_d,
            'first_stage_loss': first_stage_loss.item() if torch.is_tensor(first_stage_loss) else first_stage_loss,
            'e2e_loss': e2e_loss.item() if torch.is_tensor(e2e_loss) else e2e_loss,
            'L_loss': L_loss.item(),
            'entropy_a': entropy_a.item() if torch.is_tensor(entropy_a) else float(entropy_a),
            'entropy_c': entropy_c.item() if torch.is_tensor(entropy_c) else float(entropy_c),
            'z_loss': 0.0,
            'e2e_diag_total': int(e2e_diag['total_samples']),
            'e2e_diag_errors': int(e2e_error_total),
            'e2e_true_prob_mean': e2e_diag['true_prob_sum'] / e2e_total,
            'e2e_floor_frac': e2e_diag['floor_hits'] / e2e_total,
            'e2e_route_acc': e2e_diag['route_correct'] / max(e2e_diag['route_total'], 1),
            'e2e_leaf_acc': e2e_diag['leaf_correct'] / max(e2e_diag['leaf_total'], 1),
            'e2e_err_route_only_frac': e2e_diag['route_only_errors'] / e2e_error_denom,
            'e2e_err_leaf_only_frac': e2e_diag['leaf_only_errors'] / e2e_error_denom,
            'e2e_err_both_frac': e2e_diag['both_errors'] / e2e_error_denom,
            'e2e_err_other_frac': e2e_diag['other_errors'] / e2e_error_denom,
            'e2e_z_margin_mean': e2e_diag['z_margin_sum'] / max(e2e_diag['z_margin_count'], 1),
            'e2e_c_margin_mean': e2e_diag['c_margin_sum'] / max(e2e_diag['c_margin_count'], 1),
        }

        # Log feature group attention weights (if available in model)
        with torch.no_grad():
            fa_target = model.module if isinstance(model, nn.DataParallel) else model
            if hasattr(fa_target, 'feature_attention') and hasattr(fa_target.feature_attention, 'get_group_weights'):
                group_weights = fa_target.feature_attention.get_group_weights()
                logging.info(f"Feature group weights: {group_weights}")

        
        return total_loss, loss_components

    def train_step_grpo(train_loader, epoch_idx, global_step):
        """Training step with GRPO techniques."""
        global stop_training
        model.train()
        # train_loader = next(iter(train_data_loaders))
        first_stage_graph = train_loader[1].to(device)
        states = train_loader[0]
        second_stage_states = train_loader[4]

        # === Extract and prepare labels (same as original) ===
        first_stage_variable_indices = [s['first_stage_variable_indices'].to(device) for s in states]
        variable_shapes = [s['variable_shapes'] for s in states]
        # NOTE: first_stage_constraint_shapes removed - not used by simplified model
        variable_labels = [flatten_selected_solution_data(s['solution_data']).to(device) for s in states]

        var_sizes = [
            [torch.prod(torch.tensor(shape, device=device)).item() for shape in sample[:4]]
            for sample in variable_shapes
        ]
        split_points_per_sample = [
            torch.cumsum(torch.tensor([0] + sample_sizes, device=device), dim=0)
            for sample_sizes in var_sizes
        ]
        
        true_a = torch.cat([vl[sp[0]:sp[1]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)
        true_b = torch.cat([vl[sp[1]:sp[2]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)
        true_c = torch.cat([vl[sp[2]:sp[3]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)
        true_d = torch.cat([vl[sp[3]:sp[4]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)

        # Reshape a and c per graph
        true_a_per_graph = []
        true_c_per_graph = []
        a_offset, c_offset = 0, 0

        for shapes in variable_shapes:
            P, Td = shapes[0]
            K, Tl = shapes[2]
            a_size, c_size = P * Td, K * Tl
            true_a_per_graph.append(true_a[a_offset:a_offset + a_size].view(P, Td))
            true_c_per_graph.append(true_c[c_offset:c_offset + c_size].view(K, Tl))
            a_offset += a_size
            c_offset += c_size

        # Build z labels
        z_labels_list = []
        for s, ss, sp in zip(states, second_stage_states, split_points_per_sample):
            z_gt = s['solution_data']['variable_z'].to(device).view(-1)
            idx = ss['second_stage_variable_indices'].to(device).view(-1).long()
            z_len = z_gt.numel()
            s0, s1, s2, s3, s4 = [int(x) for x in sp.tolist()]
            z_off = s4
            is_abs = bool(((idx >= z_off) & (idx < z_off + z_len)).all().item()) if z_len > 0 else False
            rel_idx = idx - z_off if is_abs else idx
            z_labels_list.append(z_gt[rel_idx])

        flat_z = torch.cat(z_labels_list, dim=0).view(-1)
        z_labels = flat_z.view(-1, 4)
        z_labels_indices = z_labels.argmax(dim=-1)

        L_labels = [s['solution_data']['variable_L'].to(device) for s in states]
        L_labels_per_batch = torch.stack([L.sum() for L in L_labels], dim=0)

        # NOTE: second_stage features (constraint_features, variable_features, edge_indices)
        # removed - not used by simplified model with explicit tree routing

        inputs = (
            first_stage_graph, True, first_stage_variable_indices, variable_shapes,
            states,  # first_stage_states containing original X/Y data for explicit tree routing
            device
        )

        true_labels = {
            'true_a_per_graph': true_a_per_graph,
            'true_b': true_b,
            'true_c_per_graph': true_c_per_graph,
            'true_d': true_d,
            'z_labels_indices': z_labels_indices,
            'L_labels_per_batch': L_labels_per_batch,
            'z_labels_list': z_labels_list,  # Per-problem z labels for size-fair z_loss
        }

        # === Extract normalized_X for tree routing ===
        # normalized_X is [N, P] - the input features used for decision tree routing
        normalized_X_list = []
        for s in states:
            if 'normalized_X' in s:
                normalized_X_list.append(s['normalized_X'].to(device))
        
        # Concatenate all samples from all graphs in batch
        if normalized_X_list:
            # Some datasets can have different feature counts; pad to common width.
            max_p = max(x.shape[1] for x in normalized_X_list)
            padded_list = []
            for x in normalized_X_list:
                if x.shape[1] < max_p:
                    pad_width = max_p - x.shape[1]
                    x = torch.nn.functional.pad(x, (0, pad_width), mode='constant', value=0.0)
                padded_list.append(x)
            normalized_X = torch.cat(padded_list, dim=0)  # [total_N, P]
        else:
            normalized_X = None
            logging.warning("normalized_X not found in states - using fallback infeasibility")

        # === Extract sample_true_labels for end-to-end accuracy ===
        # sample_true_labels is [N] - the ground truth class label for each sample
        sample_true_labels_list = []
        for s in states:
            if 'sample_true_labels' in s:
                sample_true_labels_list.append(s['sample_true_labels'].to(device))
        
        if sample_true_labels_list:
            sample_true_labels = torch.cat(sample_true_labels_list, dim=0)  # [total_N]
        else:
            sample_true_labels = None
            logging.debug("sample_true_labels not found in states - using z_labels as proxy for E2E accuracy")

        # Add sample_true_labels to true_labels dict for compute_standard_loss
        true_labels['sample_true_labels'] = sample_true_labels

        # === Extract dataset name for normalization ===
        # For single-sample batches, get the dataset name from meta_info
        dataset_names = dataset_normalizer.get_dataset_names_from_batch(states)
        current_dataset = dataset_names[0] if dataset_names else 'unknown'

        # === Determine loss method ===
        loss_method = getattr(grpo_config, 'loss_method', 'value_ratio')
        e2e_weight = getattr(grpo_config, 'e2e_weight', 0.5)
        
        # === GRPO candidate sampling (SKIP in supervised_only mode) ===
        # In supervised_only mode, candidates add noise (temperature, dropout, Gaussian)
        # and waste 4 forward passes that aren't used for the loss.
        if loss_method == 'supervised_only':
            # No candidate sampling — pure supervised learning
            candidates = []
            rewards = []
            reward_details = []
            advantages = None
            norm_details = {
                'dataset_baseline_mean': 0.0,
                'dataset_baseline_std': 1.0,
                'dataset_sample_count': 0,
            }
        else:
            # GRPO Technique 3: Sample multiple candidates
            candidates = candidate_sampler.sample_candidates(model, inputs, device)
            
            # GRPO Technique 1: Compute rewards and normalize within group
            rewards = []
            reward_details = []
            for cand in candidates:
                reward, details = reward_computer.compute_reward(
                    cand, true_labels, variable_shapes, normalized_X=normalized_X
                )
                rewards.append(reward)
                reward_details.append(details)
            
            # GRPO Technique 2: Dataset-aware advantage normalization
            advantages, norm_details = dataset_normalizer.compute_normalized_advantages(
                rewards, current_dataset, update_stats=True
            )

        # Zero gradients before forward pass
        encoder_optimizer.zero_grad()
        b_optimizer.zero_grad()
        loss_weight_optimizer.zero_grad()

        # === Compute loss ===
        total_loss, loss_metrics = loss_computer.compute_advantage_weighted_loss(
            model, inputs, true_labels, advantages, candidates,
            compute_standard_loss,
            loss_method=loss_method,
            sample_true_labels=sample_true_labels,
            normalized_X=normalized_X,
            e2e_weight=e2e_weight,
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Detailed diagnostics for NaN detection
            logging.warning(f"[NaN DETECTED] Skipping batch - Diagnosing source:")
            logging.warning(f"  - Policy gradient loss: {loss_metrics.get('policy_gradient_loss', 'N/A')}")
            logging.warning(f"  - Supervised loss: {loss_metrics.get('supervised_loss', 'N/A')}")
            logging.warning(f"  - avg_combined_ratio: {loss_metrics.get('avg_combined_ratio', 'N/A')}")
            logging.warning(f"  - best_ratio_a: {loss_metrics.get('best_ratio_a', 'N/A')}")
            logging.warning(f"  - best_ratio_b: {loss_metrics.get('best_ratio_b', 'N/A')}")
            logging.warning(f"  - best_ratio_c: {loss_metrics.get('best_ratio_c', 'N/A')}")
            logging.warning(f"  - best_ratio_d: {loss_metrics.get('best_ratio_d', 'N/A')}")
            logging.warning(f"  - best_ratio_z: {loss_metrics.get('best_ratio_z', 'N/A')}")
            logging.warning(f"  - advantages: {advantages.tolist() if advantages is not None else 'N/A'}")
            logging.warning(f"  - rewards: {[r.item() for r in rewards]}")
            # Zero gradients and skip this batch
            encoder_optimizer.zero_grad()
            b_optimizer.zero_grad()
            loss_weight_optimizer.zero_grad()
            return None

        # Backward pass
        total_loss.backward()

        # === Feature attention gradient diagnostic (after backward) ===
        if hasattr(target, 'feature_attention') and hasattr(target.feature_attention, 'group_logits'):
            if target.feature_attention.group_logits.grad is not None:
                grad_norm = target.feature_attention.group_logits.grad.norm().item()
                logging.info(f"Feature attention grad norm: {grad_norm:.6f}")
            else:
                logging.info("Feature attention grad is None (check if optimizer includes these params)")

        # === Per-head gradient norm logging ===
        head_grad_norms = {}
        for head_name, head_module in [('a_head', target.a_head),
                                        ('b_head', target.b_head),
                                        ('c_head', target.c_head),
                                        ('d_head', target.d_head),
                                        ('encoder', target.first_layers)]:
            head_norm_sq = sum(
                p.grad.norm() ** 2 for p in head_module.parameters() if p.grad is not None
            )
            head_grad_norms[head_name] = head_norm_sq ** 0.5

        logging.info(
            f"[GRAD-NORM] "
            f"a_head={head_grad_norms['a_head']:.4f}  "
            f"b_head={head_grad_norms['b_head']:.4f}  "
            f"c_head={head_grad_norms['c_head']:.4f}  "
            f"d_head={head_grad_norms['d_head']:.4f}  "
            f"encoder={head_grad_norms['encoder']:.4f}"
        )

        # Per-head gradient clipping for b_head
        # Increased to 80 to allow full gradient updates - gradients stable at 33-35
        B_GRAD_CLIP = 80.0  # Increased from 15 - model learning slowly, needs larger steps
        b_grad_norm_before = head_grad_norms['b_head']
        b_grad_norm = torch.nn.utils.clip_grad_norm_(target.b_head.parameters(), B_GRAD_CLIP)
        if b_grad_norm_before > B_GRAD_CLIP * 10:
            logging.info(f"[B-HEAD] Gradient clipped: {b_grad_norm_before:.2f} -> {B_GRAD_CLIP}")

        # Per-head gradient clipping for a_head
        # Set to 100 to allow strong gradient updates from 10x amplified cross-entropy
        # Higher than B (80) because A has P outputs vs B's scalar threshold
        A_GRAD_CLIP = 100.0
        a_grad_norm_before = head_grad_norms['a_head']
        a_grad_norm = torch.nn.utils.clip_grad_norm_(target.a_head.parameters(), A_GRAD_CLIP)
        if a_grad_norm_before > A_GRAD_CLIP * 10:
            logging.info(f"[A-HEAD] Gradient clipped: {a_grad_norm_before:.2f} -> {A_GRAD_CLIP}")

        # Global gradient clipping (for all other parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # Warn if gradients are unusually large (sign of instability)
        if grad_norm > grad_clip_norm * 0.8:
            logging.warning(f"[EPOCH {epoch_idx}] Large gradient detected: {grad_norm:.4f} (clipped to {grad_clip_norm})")

        # Optimizer steps
        encoder_optimizer.step()
        b_optimizer.step()
        loss_weight_optimizer.step()

        # Clamp E2E/L weights in supervised_only to prevent collapse
        if loss_method == 'supervised_only':
            with torch.no_grad():
                min_e2e_log = math.log(MIN_E2E_WEIGHT)
                min_L_log = math.log(MIN_L_WEIGHT)
                target.log_loss_weights.data[1] = torch.clamp(
                    target.log_loss_weights.data[1], min=min_e2e_log
                )
                target.log_loss_weights.data[2] = torch.clamp(
                    target.log_loss_weights.data[2], min=min_L_log
                )

        # NOTE: scheduler.step() called OUTSIDE this function (once per EPOCH)
        # Previously was here causing LR to decay every BATCH (10x too fast!)
        # This was the root cause of accuracy collapse after epoch 200

        # Logging - every step (full detailed logging for debugging)
        if rewards:
            avg_reward = sum(r.item() for r in rewards) / len(rewards)
            best_reward = max(r.item() for r in rewards)
        else:
            avg_reward = 0.0
            best_reward = 0.0
        
        # DEBUG: Check why importance ratios might not be changing (GRPO only)
        if loss_method != 'supervised_only':
            metrics_per_candidate = loss_metrics.get('metrics_per_candidate')
            if metrics_per_candidate and all('log_prob_new' in m and 'log_prob_old' in m for m in metrics_per_candidate):
                log_probs_new = [m['log_prob_new'] for m in metrics_per_candidate]
                log_probs_old = [m['log_prob_old'] for m in metrics_per_candidate]
                log_prob_variance = sum((ln - lo)**2 for ln, lo in zip(log_probs_new, log_probs_old)) / len(log_probs_new)
                logging.debug(f"[LOG-PROB-VARIANCE] Variance between new and old log probs: {log_prob_variance:.8f} (should be > 0 if policy is updating)")
            else:
                logging.debug("[LOG-PROB-VARIANCE] Skipping variance check (log_prob metrics unavailable)")
        
        # Get loss components from metrics
        loss_a = loss_metrics.get('loss_a', 0)
        loss_b = loss_metrics.get('loss_b', 0)
        loss_c = loss_metrics.get('loss_c', 0)
        loss_d = loss_metrics.get('loss_d', 0)
        e2e_loss = loss_metrics.get('e2e_loss', 0)
        L_loss = loss_metrics.get('L_loss', 0)
        
        # Compute aggregate losses
        first_stage_loss = loss_a + loss_b + loss_c + loss_d
        
        # Get learned weights (normalized)
        raw_weights = torch.exp(target.log_loss_weights)
        fs_weight = raw_weights[0] / (raw_weights[0] + 1e-8) * FS_WEIGHT_SUM
        weights = torch.cat([fs_weight.unsqueeze(0), raw_weights[1:3]], dim=0)

        # === Log raw losses (unweighted) ===
        logging.info(f"[SIZE-NORM-TEST] [TRAIN] Raw fs_loss: {first_stage_loss:.4f} (a={loss_a:.4f}, b={loss_b:.4f}, c={loss_c:.4f}, d={loss_d:.4f}), e2e: {e2e_loss:.4f}, L: {L_loss:.4f}")
        
        # === Log weighted losses (contribution to total loss) ===
        weighted_fs = weights[0] * first_stage_loss if isinstance(first_stage_loss, (int, float)) else (weights[0] * torch.tensor(first_stage_loss)).item()
        weighted_e2e = weights[1] * e2e_loss if isinstance(e2e_loss, (int, float)) else (weights[1] * torch.tensor(e2e_loss)).item()
        weighted_L = weights[2] * L_loss if isinstance(L_loss, (int, float)) else (weights[2] * torch.tensor(L_loss)).item()

        logging.info(f"[SIZE-NORM-TEST] [TRAIN] Weighted loss: fs: {weighted_fs:.4f}, e2e: {weighted_e2e:.4f}, L: {weighted_L:.4f}")

        e2e_diag_total = int(loss_metrics.get('e2e_diag_total', 0))
        if e2e_diag_total > 0:
            e2e_diag_errors = int(loss_metrics.get('e2e_diag_errors', 0))
            logging.info(
                "[E2E-DIAG] "
                f"p_true_mean={loss_metrics.get('e2e_true_prob_mean', 0.0):.4f} | "
                f"floor_frac={loss_metrics.get('e2e_floor_frac', 0.0):.2%} | "
                f"route_acc={loss_metrics.get('e2e_route_acc', 0.0):.4f} | "
                f"leaf_acc={loss_metrics.get('e2e_leaf_acc', 0.0):.4f} | "
                f"error_split(route_only={loss_metrics.get('e2e_err_route_only_frac', 0.0):.2%}, "
                f"leaf_only={loss_metrics.get('e2e_err_leaf_only_frac', 0.0):.2%}, "
                f"both={loss_metrics.get('e2e_err_both_frac', 0.0):.2%}, "
                f"other={loss_metrics.get('e2e_err_other_frac', 0.0):.2%}) | "
                f"margins(z={loss_metrics.get('e2e_z_margin_mean', 0.0):.4f}, "
                f"c={loss_metrics.get('e2e_c_margin_mean', 0.0):.4f}) | "
                f"samples={e2e_diag_total}, errors={e2e_diag_errors}"
            )

        # DEBUG: Check B prediction statistics to understand why it's not learning
        if 'pred_b' in loss_metrics:
            pred_b_values = loss_metrics['pred_b']
            if pred_b_values is not None and len(pred_b_values) > 0:
                b_mean = sum(pred_b_values) / len(pred_b_values)
                b_std = (sum((b - b_mean)**2 for b in pred_b_values) / len(pred_b_values)) ** 0.5
                b_min = min(pred_b_values)
                b_max = max(pred_b_values)
                logging.debug(f"[B-PRED-DEBUG] B predictions: mean={b_mean:.4f}, std={b_std:.4f}, min={b_min:.4f}, max={b_max:.4f}")
        
        # Check for loss explosion (sign of instability)
        if first_stage_loss > 10.0:
            logging.error(f"[EPOCH {epoch_idx}] LOSS EXPLOSION DETECTED! First-stage loss: {first_stage_loss:.4f}")
            logging.error(f"  - loss_a: {loss_a:.4f}, loss_b: {loss_b:.4f}, loss_c: {loss_c:.4f}, loss_d: {loss_d:.4f}")
            logging.error(f"  - Consider reducing learning rate or adding warmup")
            # Additional debug for B when it explodes
            if loss_b > 10.0:
                logging.error(f"[B-EXPLOSION] Variable B loss is extremely high! Check B head gradients and optimizer LR.")
        
        # Log entropy values and routing temperature
        entropy_a = loss_metrics.get('entropy_a', 0)
        entropy_c = loss_metrics.get('entropy_c', 0)
        target_temp = model.module if isinstance(model, nn.DataParallel) else model
        routing_temp = target_temp.routing_temperature
        logging.info(f"Entropy: entropy_a={entropy_a:.4f}, entropy_c={entropy_c:.4f} | Routing tau={routing_temp:.4f}")
        
        # Log aggregate losses
        second_stage_loss = e2e_loss + L_loss
        logging.info(f"First-stage loss (a+b+c+d): {first_stage_loss:.4f} | Second-stage loss (e2e+L): {second_stage_loss:.4f}")
        
        # Log learned weights with FS sum
        logging.info(f"Learned weights: FS={weights[0]:.4f}, E2E={weights[1]:.4f}, L={weights[2]:.4f} | sum={weights.sum().item():.4f}")
        
        # NOTE: linear_layer logging removed - simplified model uses explicit tree routing

        # GroupedFeatureAttention weights - shows which feature groups the model prioritizes
        if hasattr(target, 'feature_attention'):
            group_weights = target.feature_attention.get_group_weights()
            logging.info(f"[TRAIN] GroupedFeatureAttention weights:")
            for group_name, weight in group_weights.items():
                logging.info(f"        {group_name:20s}: {weight:.6f}")
            # Highlight x_aggregated weight for easier tracking
            if 'x_aggregated' in group_weights:
                logging.info(f"[TRAIN] >>> x_aggregated weight: {group_weights['x_aggregated']:.6f} <<<")

        # Log GRPO Policy Gradient metrics (only relevant for GRPO modes)
        if loss_method != 'supervised_only':
            pg_loss = loss_metrics.get('policy_gradient_loss', 0)
            sup_loss = loss_metrics.get('supervised_loss', 0)
            alpha = loss_metrics.get('alpha', 0.3)
            
            logging.info(
                f"[TRAIN] Epoch {epoch_idx} | Step {global_step} | "
                f"Total Loss: {total_loss.item():.4f} | "
                f"PG Loss: {pg_loss:.4f} | "
                f"Supervised Loss: {sup_loss:.4f} | "
                f"α={alpha:.2f}"
            )
        else:
            logging.info(
                f"[TRAIN] Epoch {epoch_idx} | Step {global_step} | "
                f"Total Loss: {total_loss.item():.4f} (supervised_only)"
            )
        
        # === GRPO-specific logging (skip in supervised_only mode) ===
        if loss_method != 'supervised_only':
            # Log importance ratio statistics (GRPO key metrics)
            avg_ratio = loss_metrics.get('avg_ratio', 1.0)
            avg_log_ratio = loss_metrics.get('avg_log_ratio', 0.0)
            best_ratio = loss_metrics.get('best_ratio', 1.0)
            best_log_ratio = loss_metrics.get('best_log_ratio', 0.0)
            best_log_ratio_raw = loss_metrics.get('best_log_ratio_raw', 0.0)
            log_ratio_clamp_frac = loss_metrics.get('log_ratio_clamp_fraction', 0.0)
            use_clipping = loss_metrics.get('use_clipping', False)
            clip_fraction = loss_metrics.get('clip_fraction', 0.0)
            
            logging.info(
                f"[GRPO] Importance Ratio: avg={avg_ratio:.4f}, best={best_ratio:.4f} | "
                f"Log-Ratio: avg={avg_log_ratio:.4f}, best={best_log_ratio:.4f} (raw={best_log_ratio_raw:.2f})"
            )
            
            # Log clamping statistics (important for detecting numerical instability)
            if log_ratio_clamp_frac > 0:
                logging.warning(f"[GRPO] Log-ratio CLAMPED for {log_ratio_clamp_frac:.0%} of candidates (numerical instability detected)")
            
            if use_clipping:
                clip_eps = loss_metrics.get('clip_epsilon', 0.2)
                logging.info(f"[GRPO] Clipping: ENABLED (ε={clip_eps:.2f}), Clip Fraction: {clip_fraction:.2%}")
            else:
                logging.info(f"[GRPO] Clipping: DISABLED (vanilla GRPO)")
            
            # Log per-variable ratios for best candidate (variable-level decomposition)
            logging.info(
                f"[GRPO] Best candidate per-variable ratios: "
                f"a={loss_metrics.get('best_ratio_a', 1):.4f}, "
                f"b={loss_metrics.get('best_ratio_b', 1):.4f}, "
                f"c={loss_metrics.get('best_ratio_c', 1):.4f}, "
                f"d={loss_metrics.get('best_ratio_d', 1):.4f}, "
                f"z={loss_metrics.get('best_ratio_z', 1):.4f}"
            )
            
            # Log log-probs
            best_log_prob_new = loss_metrics.get('best_log_prob_new', 0)
            best_log_prob_old = loss_metrics.get('best_log_prob_old', 0)
            logging.info(
                f"[GRPO] Log-Probs: new={best_log_prob_new:.2f}, old={best_log_prob_old:.2f} | "
                f"Avg Reward: {avg_reward:.4f} | Best Reward: {best_reward:.4f}"
            )
            
            # Log dataset normalization info
            logging.info(
                f"[DATASET-NORM] Dataset: '{current_dataset}' | "
                f"Baseline: mean={norm_details['dataset_baseline_mean']:.4f}, std={norm_details['dataset_baseline_std']:.4f} | "
                f"Samples seen: {norm_details['dataset_sample_count']}"
            )
            
            # Log accuracy breakdown from best candidate (now includes z accuracy)
            if reward_details:
                best_details = reward_details[loss_metrics['best_candidate_idx']]
                logging.info(f"Best candidate accuracy: acc_a={best_details.get('acc_a', 0):.3f}, "
                            f"acc_b={best_details.get('acc_b', 0):.3f}, acc_c={best_details.get('acc_c', 0):.3f}, "
                            f"acc_d={best_details.get('acc_d', 0):.3f}, acc_z={best_details.get('acc_z', 0):.3f}")
        
        # === ARGMAX ACCURACY: Deterministic model quality metric (no sampling noise) ===
        # This is the TRUE measure of whether the model has learned the training data.
        if 'argmax_acc_a' in loss_metrics:
            logging.info(
                f"[TRAIN-ARGMAX] Deterministic accuracy: "
                f"a={loss_metrics['argmax_acc_a']:.3f}, "
                f"b={loss_metrics['argmax_acc_b']:.3f}, "
                f"c={loss_metrics['argmax_acc_c']:.3f}, "
                f"d={loss_metrics['argmax_acc_d']:.3f}, "
                f"z={loss_metrics['argmax_acc_z']:.3f}"
            )

        # === END-TO-END CLASSIFICATION ACCURACY (logging only, not used for optimizer) ===
        # This is the actual inference metric: predicted class vs true class label
        if 'e2e_accuracy' in loss_metrics:
            e2e_acc = loss_metrics['e2e_accuracy']
            e2e_correct = loss_metrics.get('e2e_correct', 0)
            e2e_total = loss_metrics.get('e2e_total', 0)
            logging.info(
                f"[TRAIN] End-to-End Classification Accuracy: {e2e_acc:.4f} ({e2e_correct}/{e2e_total})"
            )

        # GRPO-specific: routing infeasibility and E2E accuracy
        if loss_method != 'supervised_only' and reward_details:
            best_details = reward_details[loss_metrics['best_candidate_idx']]
            routing_mismatch = best_details.get('routing_mismatch', None)
            if routing_mismatch is not None:
                logging.info(f"Routing infeasibility (z_sampled vs z_tree mismatch): {routing_mismatch:.3f}")
                # Log per-leaf agreement rates if available
                leaf_agrees = [f"leaf_{i}={best_details.get(f'leaf_{i}_agree', 0):.2f}" 
                              for i in range(4) if f'leaf_{i}_agree' in best_details]
                if leaf_agrees:
                    logging.info(f"Per-leaf agreement rates: {', '.join(leaf_agrees)}")

        # Log E2E accuracy if using that loss method
        if loss_method == 'e2e_accuracy':
            logging.info(
                f"[E2E] Loss Method: e2e_accuracy (weight={e2e_weight:.2f}) | "
                f"Avg E2E Accuracy: {loss_metrics.get('avg_e2e_accuracy', 0):.4f} | "
                f"Best E2E Accuracy: {loss_metrics.get('best_e2e_accuracy', 0):.4f}"
            )
            logging.info(
                f"[E2E] Best candidate accuracies: "
                f"var={loss_metrics.get('best_combined_accuracy', 0):.3f}, "
                f"a={loss_metrics.get('best_acc_a', 0):.3f}, "
                f"b={loss_metrics.get('best_acc_b', 0):.3f}, "
                f"c={loss_metrics.get('best_acc_c', 0):.3f}, "
                f"d={loss_metrics.get('best_acc_d', 0):.3f}, "
                f"z={loss_metrics.get('best_acc_z', 0):.3f}"
            )

        return total_loss.item()

    def eval_step_grpo(valid_loader, device) -> Tuple[float, Optional[Dict[str, Any]], float]:
        """Validation step - standard forward pass without GRPO candidate sampling.

        Returns:
            Tuple of (loss, predictions_dict, e2e_accuracy)
        """
        model.eval()
        torch.cuda.empty_cache()

        try:
            # valid_loader = next(iter(valid_data_loaders))
            states = valid_loader[0]
            first_stage_graph = valid_loader[1].to(device)
            second_stage_states = valid_loader[4]
        except Exception as e:
            logging.warning(f"[VALID] Invalid batch format, skipping. Error: {e}")
            return float('nan'), None, 0.0

        # === Extract and prepare labels (same as training) ===
        first_stage_variable_indices = [s['first_stage_variable_indices'].to(device) for s in states]
        variable_shapes = [s['variable_shapes'] for s in states]
        # NOTE: first_stage_constraint_shapes removed - not used by simplified model
        variable_labels = [flatten_selected_solution_data(s['solution_data']).to(device) for s in states]

        # === Extract sample_true_labels for end-to-end accuracy ===
        sample_true_labels_list = []
        for s in states:
            if 'sample_true_labels' in s:
                sample_true_labels_list.append(s['sample_true_labels'].to(device))

        if sample_true_labels_list:
            sample_true_labels = torch.cat(sample_true_labels_list, dim=0)  # [total_N]
        else:
            sample_true_labels = None

        var_sizes = [
            [torch.prod(torch.tensor(shape, device=device)).item() for shape in sample[:4]]
            for sample in variable_shapes
        ]
        split_points_per_sample = [
            torch.cumsum(torch.tensor([0] + sample_sizes, device=device), dim=0)
            for sample_sizes in var_sizes
        ]
        
        true_a = torch.cat([vl[sp[0]:sp[1]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)
        true_b = torch.cat([vl[sp[1]:sp[2]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)
        true_c = torch.cat([vl[sp[2]:sp[3]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)
        true_d = torch.cat([vl[sp[3]:sp[4]] for vl, sp in zip(variable_labels, split_points_per_sample)]).squeeze(-1)

        # Reshape a and c per graph
        true_a_per_graph = []
        true_c_per_graph = []
        a_offset, c_offset = 0, 0

        for shapes in variable_shapes:
            P, Td = shapes[0]
            K, Tl = shapes[2]
            a_size, c_size = P * Td, K * Tl
            true_a_per_graph.append(true_a[a_offset:a_offset + a_size].view(P, Td))
            true_c_per_graph.append(true_c[c_offset:c_offset + c_size].view(K, Tl))
            a_offset += a_size
            c_offset += c_size

        # Build z labels
        z_labels_list = []
        for s, ss, sp in zip(states, second_stage_states, split_points_per_sample):
            z_gt = s['solution_data']['variable_z'].to(device).view(-1)
            idx = ss['second_stage_variable_indices'].to(device).view(-1).long()
            z_len = z_gt.numel()
            s0, s1, s2, s3, s4 = [int(x) for x in sp.tolist()]
            z_off = s4
            is_abs = bool(((idx >= z_off) & (idx < z_off + z_len)).all().item()) if z_len > 0 else False
            rel_idx = idx - z_off if is_abs else idx
            z_labels_list.append(z_gt[rel_idx])

        flat_z = torch.cat(z_labels_list, dim=0).view(-1)
        z_labels = flat_z.view(-1, 4)
        z_labels_indices = z_labels.argmax(dim=-1)

        L_labels = [s['solution_data']['variable_L'].to(device) for s in states]
        L_labels_per_batch = torch.stack([L.sum() for L in L_labels], dim=0)

        # NOTE: second_stage features (constraint_features, variable_features, edge_indices)
        # removed - not used by simplified model with explicit tree routing

        # === Model forward pass (no training, no dropout) ===
        # Use low temperature for validation to approximate hard decisions (tree-like)
        val_target = model.module if isinstance(model, nn.DataParallel) else model
        saved_routing_temp = val_target.routing_temperature
        val_target.routing_temperature = 0.1  # Near-hard for validation

        inputs = (
            first_stage_graph, False, first_stage_variable_indices, variable_shapes,
            states,  # first_stage_states containing original X/Y data for explicit tree routing
            device
        )
        
        predictions = None
        with torch.no_grad():
            fs_logits, z_logits, L_logits, all_L = model(*inputs)
            pred_a, pred_b, pred_c, pred_d = fs_logits

            # === Compute losses ===
            # Loss a: cross-entropy per decision node
            loss_a = 0
            num_a_cols = 0
            for pred_a_graph, true_a_graph in zip(pred_a, true_a_per_graph):
                for col in range(pred_a_graph.shape[1]):
                    loss_a += F.cross_entropy(
                        pred_a_graph[:, col].unsqueeze(0),
                        true_a_graph[:, col].argmax().unsqueeze(0)
                    )
                    num_a_cols += 1
            loss_a = loss_a / max(num_a_cols, 1)

            # Entropy regularization on A to encourage sharp one-hot feature selection
            entropy_a = 0
            for pred_a_graph in pred_a:
                a_probs = F.softmax(pred_a_graph, dim=0)  # [P, Td]
                col_entropy = -(a_probs * torch.log(a_probs + 1e-8)).sum(dim=0)  # [Td]
                entropy_a += col_entropy.mean()
            entropy_a = entropy_a / max(len(pred_a), 1)
            loss_a = loss_a + ENTROPY_WEIGHT_A * entropy_a

            # Loss b: smooth L1
            loss_b = F.mse_loss(pred_b, true_b, reduction='mean')

            # Loss c: cross-entropy per leaf node (matching train - no clamping)
            loss_c = 0
            num_c_cols = 0
            for pred_c_graph, true_c_graph in zip(pred_c, true_c_per_graph):
                for col in range(pred_c_graph.shape[1]):
                    loss_c += F.cross_entropy(
                        pred_c_graph[:, col].unsqueeze(0),
                        true_c_graph[:, col].argmax().unsqueeze(0)
                    )
                    num_c_cols += 1
            loss_c = loss_c / max(num_c_cols, 1)

            # Entropy regularization on C to encourage sharp one-hot class assignment
            entropy_c = 0
            for pred_c_graph in pred_c:
                c_probs = F.softmax(pred_c_graph, dim=0)  # [K, Tl]
                col_entropy = -(c_probs * torch.log(c_probs + 1e-8)).sum(dim=0)  # [Tl]
                entropy_c += col_entropy.mean()
            entropy_c = entropy_c / max(len(pred_c), 1)
            loss_c = loss_c + ENTROPY_WEIGHT_C * entropy_c

            # Loss d: BCE - SCALED DOWN by 0.1 to match training
            loss_d = F.binary_cross_entropy_with_logits(pred_d, true_d, reduction='mean') * 0.1

            # ==== REPLACE Z LOSS WITH E2E LOSS ====
            e2e_loss = torch.tensor(0.0, device=z_logits.device)
            e2e_diag = {
                'total_samples': 0,
                'correct_samples': 0,
                'floor_hits': 0,
                'route_correct': 0,
                'route_total': 0,
                'leaf_correct': 0,
                'leaf_total': 0,
                'route_only_errors': 0,
                'leaf_only_errors': 0,
                'both_errors': 0,
                'other_errors': 0,
                'true_prob_sum': 0.0,
                'z_margin_sum': 0.0,
                'z_margin_count': 0,
                'c_margin_sum': 0.0,
                'c_margin_count': 0,
            }
            
            if sample_true_labels is not None:
                true_labels_idx = sample_true_labels.long() - 1
                # Soft routing for validation (matching training — no second STE)
                z_soft = F.softmax(z_logits, dim=-1)  # [total_N, Tl]
                z_hard_idx = z_soft.argmax(dim=-1)    # kept for diagnostics only

                if z_soft.shape[1] >= 2:
                    z_top2 = torch.topk(z_soft, k=2, dim=-1).values
                    z_margins = z_top2[:, 0] - z_top2[:, 1]
                    e2e_diag['z_margin_sum'] += z_margins.sum().item()
                    e2e_diag['z_margin_count'] += z_margins.numel()
                
                if len(z_labels_list) > 0 and len(pred_c) == len(z_labels_list):
                    expected_class_probs_list = []
                    z_offset = 0
                    for graph_idx, (pred_c_graph, z_lbl) in enumerate(zip(pred_c, z_labels_list)):
                        N_i = z_lbl.numel() // 4
                        if N_i > 0:
                            true_labels_idx_i = true_labels_idx[z_offset:z_offset + N_i]
                            z_pred_i = z_hard_idx[z_offset:z_offset + N_i]
                            z_probs_i = z_soft[z_offset:z_offset + N_i]  # [N_i, Tl] — soft routing
                            c_soft = F.softmax(pred_c_graph, dim=0)    # [K, Tl]
                            if c_soft.shape[0] >= 2:
                                c_top2 = torch.topk(c_soft, k=2, dim=0).values
                                c_margins = c_top2[0] - c_top2[1]
                                e2e_diag['c_margin_sum'] += c_margins.sum().item()
                                e2e_diag['c_margin_count'] += c_margins.numel()

                            c_hard_idx = c_soft.argmax(dim=0)
                            c_hard = F.one_hot(c_hard_idx, num_classes=c_soft.shape[0]).float().transpose(0, 1)
                            c_st = c_hard + (c_soft - c_soft.detach())

                            if graph_idx < len(true_c_per_graph):
                                true_c_idx = true_c_per_graph[graph_idx].argmax(dim=0).long()
                                leaf_dim = min(c_hard_idx.numel(), true_c_idx.numel())
                                if leaf_dim > 0:
                                    e2e_diag['leaf_correct'] += (c_hard_idx[:leaf_dim] == true_c_idx[:leaf_dim]).sum().item()
                                    e2e_diag['leaf_total'] += leaf_dim

                            # N_i x Tl, K x Tl -> N_i x K
                            exp_class_probs_raw_i = torch.einsum('nl,kl->nk', z_probs_i, c_st)
                            true_class_probs_raw_i = exp_class_probs_raw_i.gather(
                                1, true_labels_idx_i.unsqueeze(1)
                            ).squeeze(1)
                            e2e_diag['floor_hits'] += (true_class_probs_raw_i < E2E_PROB_FLOOR).sum().item()
                            e2e_diag['true_prob_sum'] += true_class_probs_raw_i.sum().item()

                            exp_class_probs_i = exp_class_probs_raw_i.clamp_min(E2E_PROB_FLOOR)

                            if c_hard_idx.numel() > 0:
                                z_pred_clamped_i = z_pred_i.clamp(0, c_hard_idx.numel() - 1)
                                pred_classes_i = c_hard_idx[z_pred_clamped_i]
                                e2e_correct_i = pred_classes_i == true_labels_idx_i
                                e2e_diag['correct_samples'] += e2e_correct_i.sum().item()
                                e2e_diag['total_samples'] += e2e_correct_i.numel()

                                if z_labels_indices.numel() >= z_offset + N_i:
                                    z_true_i = z_labels_indices[z_offset:z_offset + N_i].long()
                                    route_correct_i = z_pred_i == z_true_i
                                    e2e_diag['route_correct'] += route_correct_i.sum().item()
                                    e2e_diag['route_total'] += route_correct_i.numel()

                                    z_true_clamped_i = z_true_i.clamp(0, c_hard_idx.numel() - 1)
                                    c_true_route_i = c_hard_idx[z_true_clamped_i]
                                    c_true_route_correct_i = c_true_route_i == true_labels_idx_i

                                    incorrect_i = ~e2e_correct_i
                                    route_only_i = incorrect_i & (~route_correct_i) & c_true_route_correct_i
                                    leaf_only_i = incorrect_i & route_correct_i & (~c_true_route_correct_i)
                                    both_i = incorrect_i & (~route_correct_i) & (~c_true_route_correct_i)
                                    explained_i = route_only_i | leaf_only_i | both_i
                                    other_i = incorrect_i & (~explained_i)

                                    e2e_diag['route_only_errors'] += route_only_i.sum().item()
                                    e2e_diag['leaf_only_errors'] += leaf_only_i.sum().item()
                                    e2e_diag['both_errors'] += both_i.sum().item()
                                    e2e_diag['other_errors'] += other_i.sum().item()
                            
                            log_exp_class_probs_i = torch.log(exp_class_probs_i)
                            expected_class_probs_list.append(log_exp_class_probs_i)
                            z_offset += N_i
                    
                    if expected_class_probs_list:
                        log_exp_class_probs = torch.cat(expected_class_probs_list, dim=0)
                        e2e_loss = F.nll_loss(log_exp_class_probs, true_labels_idx)
            
            # Loss L
            L_loss = F.smooth_l1_loss(all_L, L_labels_per_batch, reduction='mean')

            # === COMPUTE ACCURACY METRICS (UNIFORM AND WEIGHTED) ===
            accuracies = {}
            weights_acc = {}
            
            # Variable a: Compare predicted feature index vs true
            pred_a_indices = [pa.argmax(dim=0) for pa in pred_a]  # List of [Td] tensors
            a_correct = 0
            a_total = 0
            for pred_idx, true_a_graph in zip(pred_a_indices, true_a_per_graph):
                true_idx = true_a_graph.argmax(dim=0)  # [Td]
                a_correct += (pred_idx == true_idx).sum().item()
                a_total += pred_idx.numel()
            accuracies['acc_a'] = a_correct / max(a_total, 1)
            weights_acc['acc_a'] = a_total
            
            # Variable b: Check if prediction is within tolerance
            b_tolerance = 0.1
            b_diff = (pred_b - true_b).abs()
            b_correct = (b_diff < b_tolerance).sum().item()
            b_total = pred_b.numel()
            accuracies['acc_b'] = b_correct / max(b_total, 1)
            weights_acc['acc_b'] = b_total
            
            # Variable c: Compare predicted class index vs true
            pred_c_indices = [pc.argmax(dim=0) for pc in pred_c]  # List of [Tl] tensors
            c_correct = 0
            c_total = 0
            for pred_idx, true_c_graph in zip(pred_c_indices, true_c_per_graph):
                true_idx = true_c_graph.argmax(dim=0)  # [Tl]
                c_correct += (pred_idx == true_idx).sum().item()
                c_total += pred_idx.numel()
            accuracies['acc_c'] = c_correct / max(c_total, 1)
            weights_acc['acc_c'] = c_total
            
            # Variable d: Compare predicted binary vs true
            pred_d_binary = (torch.sigmoid(pred_d) > 0.5).float()
            true_d_binary = (true_d > 0.5).float()
            d_correct = (pred_d_binary == true_d_binary).sum().item()
            d_total = pred_d.numel()
            accuracies['acc_d'] = d_correct / max(d_total, 1)
            weights_acc['acc_d'] = d_total
            
            # Variable z: Compare predicted leaf assignment vs true
            z_pred_indices = z_logits.argmax(dim=-1)  # [N]
            z_correct = (z_pred_indices == z_labels_indices).sum().item()
            z_total = z_labels_indices.numel()
            accuracies['acc_z'] = z_correct / max(z_total, 1)
            weights_acc['acc_z'] = z_total
            
            # UNIFORM average: each variable contributes equally
            uniform_accuracy = (
                accuracies['acc_a'] + accuracies['acc_b'] + accuracies['acc_c'] +
                accuracies['acc_d'] + accuracies['acc_z']
            ) / 5.0
            
            # WEIGHTED average: weighted by element count
            total_weight_acc = sum(weights_acc.values())
            weighted_accuracy = sum(accuracies[k] * weights_acc[k] for k in accuracies) / max(total_weight_acc, 1)
            z_weight_fraction = weights_acc['acc_z'] / max(total_weight_acc, 1)
            
            logging.info(f"[SIZE-NORM-TEST] [EVAL] Accuracy: uniform={uniform_accuracy:.4f}, weighted={weighted_accuracy:.4f} | "
                        f"Element counts: a={weights_acc['acc_a']}, b={weights_acc['acc_b']}, c={weights_acc['acc_c']}, "
                        f"d={weights_acc['acc_d']}, z={weights_acc['acc_z']} | z fraction={z_weight_fraction:.2%}")

            # === END-TO-END CLASSIFICATION ACCURACY ===
            # This is the metric used at inference time: route sample → get class from C → compare to true label
            # Unlike element-wise accuracy, this measures actual classification performance
            e2e_accuracy = 0.0
            if sample_true_labels is not None:
                # Get predicted leaf for each sample
                z_pred_idx = z_logits.argmax(dim=-1)  # [N] leaf indices (0, 1, 2, 3)

                # Get predicted class for each sample based on predicted leaf and predicted C
                # pred_c is a list of [K, Tl] tensors, one per graph in batch
                # We need to map each sample to its graph's C matrix
                pred_classes_list = []
                sample_offset = 0
                for graph_idx, (pc, z_lbl) in enumerate(zip(pred_c, z_labels_list)):
                    n_samples_in_graph = z_lbl.numel() // 4
                    if n_samples_in_graph > 0:
                        # Get leaf indices for samples in this graph
                        graph_z_pred = z_pred_idx[sample_offset:sample_offset + n_samples_in_graph]
                        # Get class predictions: for each sample, pred_class = argmax(C[:, leaf])
                        # pc is [K, Tl], we need C[:, leaf].argmax() for each sample
                        for leaf_idx in graph_z_pred:
                            leaf_idx_clamped = min(leaf_idx.item(), pc.shape[1] - 1)
                            pred_class = pc[:, leaf_idx_clamped].argmax().item()
                            pred_classes_list.append(pred_class)
                        sample_offset += n_samples_in_graph

                if pred_classes_list:
                    pred_classes_tensor = torch.tensor(pred_classes_list, device=device)
                    pred_classes_tensor = pred_classes_tensor + 1
                    true_labels_adjusted = sample_true_labels[:len(pred_classes_list)].long()
                    e2e_correct = (pred_classes_tensor == true_labels_adjusted).sum().item()
                    e2e_total = len(pred_classes_list)
                    e2e_accuracy = e2e_correct / max(e2e_total, 1)
                    logging.info(f"[VALID] End-to-End Classification Accuracy: {e2e_accuracy:.4f} ({e2e_correct}/{e2e_total})")

                    e2e_diag_total = max(e2e_diag['total_samples'], 1)
                    e2e_diag_errors = max(e2e_diag['total_samples'] - e2e_diag['correct_samples'], 0)
                    e2e_diag_error_denom = max(e2e_diag_errors, 1)
                    logging.info(
                        "[VALID-E2E-DIAG] "
                        f"p_true_mean={e2e_diag['true_prob_sum'] / e2e_diag_total:.4f} | "
                        f"floor_frac={e2e_diag['floor_hits'] / e2e_diag_total:.2%} | "
                        f"route_acc={e2e_diag['route_correct'] / max(e2e_diag['route_total'], 1):.4f} | "
                        f"leaf_acc={e2e_diag['leaf_correct'] / max(e2e_diag['leaf_total'], 1):.4f} | "
                        f"error_split(route_only={e2e_diag['route_only_errors'] / e2e_diag_error_denom:.2%}, "
                        f"leaf_only={e2e_diag['leaf_only_errors'] / e2e_diag_error_denom:.2%}, "
                        f"both={e2e_diag['both_errors'] / e2e_diag_error_denom:.2%}, "
                        f"other={e2e_diag['other_errors'] / e2e_diag_error_denom:.2%}) | "
                        f"margins(z={e2e_diag['z_margin_sum'] / max(e2e_diag['z_margin_count'], 1):.4f}, "
                        f"c={e2e_diag['c_margin_sum'] / max(e2e_diag['c_margin_count'], 1):.4f})"
                    )
                else:
                    logging.warning("[VALID] Could not compute E2E accuracy - no predictions generated")
            else:
                logging.debug("[VALID] sample_true_labels not available - E2E accuracy not computed")

            # Get model weights (normalized, same as train)
            raw_weights = torch.exp(target.log_loss_weights)
            fs_weight = raw_weights[0]
            # normalize fs_weight to FS_WEIGHT_SUM if needed, here just use it.
            fs_weight_normalized = fs_weight / (fs_weight + 1e-8) * FS_WEIGHT_SUM
            weights = torch.cat([fs_weight_normalized.unsqueeze(0), raw_weights[1:]], dim=0)
            
            # Compute aggregate losses (unweighted for clarity)
            first_stage_loss_raw = loss_a + loss_b + loss_c + loss_d
            second_stage_loss_raw = e2e_loss + L_loss
            
            # Weighted losses
            first_stage_loss = weights[0] * first_stage_loss_raw
            total_loss = first_stage_loss + weights[1] * e2e_loss + weights[2] * L_loss

            # Logging - detailed breakdown
            logging.info(f"[VALID] Raw loss a: {loss_a.item():.4f}, b: {loss_b.item():.4f}, "
                        f"c: {loss_c.item():.4f}, d: {loss_d.item():.4f}, e2e: {e2e_loss.item():.4f}, L: {L_loss.item():.4f}")
            logging.info(f"[VALID] fs_loss: {first_stage_loss_raw.item():.4f}, second_stage: {second_stage_loss_raw.item():.4f}")
            logging.info(f"[VALID] Learned weights: FS={weights[0]:.4f}, E2E={weights[1]:.4f}, L={weights[2]:.4f}")
            
            # NOTE: linear_layer logging removed - simplified model uses explicit tree routing

            # GroupedFeatureAttention weights - shows which feature groups the model prioritizes
            if hasattr(target, 'feature_attention'):
                group_weights = target.feature_attention.get_group_weights()
                logging.info(f"[VALID] GroupedFeatureAttention weights:")
                for group_name, weight in group_weights.items():
                    logging.info(f"        {group_name:20s}: {weight:.6f}")
                # Highlight x_aggregated weight for easier tracking
                if 'x_aggregated' in group_weights:
                    logging.info(f"[VALID] >>> x_aggregated weight: {group_weights['x_aggregated']:.6f} <<<")

            # Total loss summary
            logging.info(f"[VALID] Total Loss: {total_loss.item():.4f} | Weighted FS: {first_stage_loss.item():.4f} | "
                        f"Weighted E2E: {(weights[1] * e2e_loss).item():.4f} | Weighted L: {(weights[2] * L_loss).item():.4f}")

            # Store full matrices for a and c (convert logits to probabilities)
            # a: List of [P, Td] matrices, c: List of [K, Tl] matrices
            # Convert logits to probabilities using softmax
            a_probs = [F.softmax(pa, dim=0).detach().cpu() for pa in pred_a]  # [P, Td] probabilities
            c_probs = [F.softmax(pc, dim=0).detach().cpu() for pc in pred_c]  # [K, Tl] probabilities
            
            first_stage_preds = {
                'a': [pa.argmax(dim=0).tolist() for pa in a_probs],  # [Td] - selected feature per decision node
                'a_full': [pa.tolist() for pa in a_probs],  # [P, Td] - full probability matrix
                'b': pred_b.detach().cpu().tolist(),  # [Td] - split thresholds (continuous values)
                'c': [pc.argmax(dim=0).tolist() for pc in c_probs],  # [Tl] - selected class per leaf
                'c_full': [pc.tolist() for pc in c_probs],  # [K, Tl] - full probability matrix
                'd': torch.sigmoid(pred_d).detach().cpu().tolist(),  # [Td] - node activation probabilities
            }
            
            # Store full Z matrices: [N, 4] for both predictions and true labels
            z_pred_indices = z_logits.argmax(dim=-1)  # [N] - predicted class
            z_probs = F.softmax(z_logits, dim=-1)  # [N, 4] - probabilities for all 4 leaves
            
            predictions = {
                'first_stage': first_stage_preds,
                'z': z_pred_indices.detach().cpu().tolist(),  # [N] - leaf assignment per sample (argmax)
                'z_full': z_probs.detach().cpu().tolist(),  # [N, 4] - full probability matrix
                'true_z': z_labels_indices.detach().cpu().tolist(),  # [N] - true class indices
                'true_z_full': z_labels.detach().cpu().tolist(),  # [N, 4] - true one-hot matrix
                'L': all_L.detach().cpu().tolist(),
                'true_L': L_labels_per_batch.detach().cpu().tolist(),
                'e2e_accuracy': e2e_accuracy,  # End-to-end classification accuracy (same as inference)
            }

        # Restore routing temperature after validation
        val_target.routing_temperature = saved_routing_temp

        return total_loss.item(), predictions, e2e_accuracy

    # === CHECKPOINT RESUME LOGIC ===
    # Check if this is a requeue/restart by looking for latest checkpoint
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "none")
    latest_ckpt_path = os.path.join(model_dir, f"latest_{SLURM_JOB_ID}.pt")
    
    current_global = 0
    global_step = 0
    start_epoch = 0
    epochs_no_improve = 0
    early_stop = False
    BEST_VALID_LOSS = float('inf')
    BEST_VALID_ACCURACY = 0.0  # Track best end-to-end classification accuracy
    EARLY_STOP_PATIENCE = 3
    EARLY_STOP_MIN_DELTA = 0.01
    
    # Try to resume from checkpoint if it exists
    if os.path.exists(latest_ckpt_path):
        logging.info("\n" + "="*80)
        logging.info("🔄 RESUMING FROM CHECKPOINT (HPC Requeue Detected)")
        logging.info("="*80)
        try:
            start_epoch, global_step = load_ckpt(latest_ckpt_path, model, encoder_optimizer, device)
            current_global = 0  # Will be incremented during training
            logging.info(f"✓ Resumed from checkpoint: epoch={start_epoch}, global_step={global_step}")
            logging.info(f"  Model weights loaded from: {latest_ckpt_path}")
            logging.info(f"  Optimizer state restored")
            logging.info(f"  Continuing training from epoch {start_epoch}...")
            logging.info("="*80 + "\n")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.error("Starting training from scratch instead")
            start_epoch = 0
            global_step = 0
    else:
        logging.info(f"No checkpoint found at {latest_ckpt_path} - starting fresh training")
        start_epoch = 0
        global_step = 0
    
    # === Staged Training Setup ===
    if use_staged_training:
        logging.info("\n" + "="*80)
        logging.info("📊 STAGED TRAINING ENABLED")
        logging.info(f"   Phase 1 (epochs 0-{phase1_epochs}): Normal baseline training")
        logging.info(f"   Phase 2 (epochs {phase1_epochs}-{num_train_steps}): Freeze A/C/D, train B only")
        logging.info("="*80 + "\n")
    
    # Determine current phase based on resumed epoch
    current_phase = 1 if global_step < phase1_epochs else 2
    phase2_initialized = (global_step >= phase1_epochs)
    
    CKPT_DIR = model_dir
    CKPT_EVERY = 2
    
    # Track last values for return
    last_loss = 0.0
    valid_loss = float('nan')
    last_first_stage_predictions = None
    last_z_predictions = None
    last_z_full = None  # Full Z probability matrix [N, 4]
    last_L_predictions = None
    last_true_z = None
    last_true_z_full = None  # True Z one-hot matrix [N, 4]
    last_true_L = None

    while global_step < num_train_steps and not stop_training and not early_stop:
        # === PHASE TRANSITION LOGIC ===
        # Handle phase transition even after resume (in case we resume in the middle of Phase 1)
        if use_staged_training and global_step == phase1_epochs and not phase2_initialized:
            logging.info("\n" + "="*80)
            logging.info("🔄 PHASE TRANSITION: Entering Phase 2")
            logging.info("="*80)
            
            # Save Phase 1 checkpoint
            save_ckpt(CKPT_DIR, f"phase1_checkpoint_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
            logging.info(f"✓ Phase 1 checkpoint saved")
            
            # Freeze encoder and specified heads
            if phase2_freeze_heads is None:
                phase2_freeze_heads = ['a', 'c', 'd']  # Default: freeze A/C/D, train only B
            
            freeze_model_components_phase2(
                freeze_encoder=phase2_freeze_encoder,
                freeze_heads=phase2_freeze_heads
            )
            
            # Update loss weights for Phase 2
            if phase2_loss_weights is None:
                # Default: only train B
                phase2_loss_weights = {'a': 0.0, 'b': 1.0, 'c': 0.0, 'd': 0.0, 'z': 0.0, 'L': 0.0}
            
            with torch.no_grad():
                # Convert loss weight dict to log-space values
                log_weights = [0.0] * 3
                # Phase 2 typically scales down FS or focuses on specific heads.
                # Here we just keep FS at a moderate level if we've switched.
                log_weights[0] = -2.0  # Reduce overall FS weight
                log_weights[1] = 0.0   # e2e weight (exp(0) = 1.0)
                log_weights[2] = -4.605 # L weight (~0.01)
                
                target.log_loss_weights.data = torch.tensor(log_weights, device=target.log_loss_weights.device)
            
            logging.info(f"✓ Phase 2 configuration:")
            logging.info(f"  - Encoder frozen: {phase2_freeze_encoder}")
            logging.info(f"  - Frozen heads: {phase2_freeze_heads}")
            logging.info(f"  - Loss weights: {phase2_loss_weights}")
            logging.info("="*80 + "\n")
            
            current_phase = 2
            phase2_initialized = True
        
        # === Temperature Annealing: update routing temperature each epoch ===
        current_routing_temp = compute_routing_temperature(
            global_step, num_train_steps,
            tau_start=routing_tau_start, tau_end=routing_tau_end
        )
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.routing_temperature = current_routing_temp

        start = time.time()
        epoch_losses = []

        for train_loader in train_data_loaders:
            loss = train_step_grpo(train_loader, global_step, current_global)
            if loss is not None:
                epoch_losses.append(loss)
                last_loss = loss
            current_global += 1
            
            if stop_training:
                break
        
        if stop_training:
            logging.info("SIGUSR1 received - saving checkpoint and exiting...")
            save_ckpt(CKPT_DIR, f"latest_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
            break
            
        end = time.time()
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        logging.info(f'======== [EPOCH {global_step}/{num_train_steps}] | Avg Train Loss = {avg_epoch_loss:.4f} | Routing tau={current_routing_temp:.4f} | Elapsed = {end - start:.2f}s ========')

        # === LR SCHEDULER STEP (once per EPOCH, not per batch!) ===
        # StepLR with step_size=300 will decay LR at epoch 300
        # Previously this was inside train_step_grpo() causing 10x faster decay
        if encoder_scheduler is not None:
            encoder_scheduler.step()
        if b_scheduler is not None:
            b_scheduler.step()

        # Log dataset normalization statistics every epoch
        all_dataset_stats = dataset_normalizer.get_all_dataset_stats()
        if all_dataset_stats:
            logging.info(f"[DATASET-NORM] Statistics across {len(all_dataset_stats)} datasets:")
            for ds_name, ds_stats in sorted(all_dataset_stats.items()):
                logging.info(
                    f"  {ds_name}: count={ds_stats['count']}, "
                    f"mean={ds_stats['mean']:.4f}, std={ds_stats['std']:.4f}, "
                    f"ema_mean={ds_stats['ema_mean']:.4f}, ema_std={ds_stats['ema_std']:.4f}"
                )

        # Save latest checkpoint every CKPT_EVERY epochs
        if global_step % CKPT_EVERY == 0:
            save_ckpt(CKPT_DIR, f"latest_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
            logging.info(f"Saved latest checkpoint at epoch {global_step}")

        # Validation
        if global_step % eval_every_steps == 0:
            logging.info(f"======== [VALID EPOCH {global_step}] Starting Evaluation ========")

            valid_losses = []
            valid_e2e_accs = []  # Track end-to-end classification accuracy

            # valid_data_loaders is a single DataLoader, iterate over its batches
            for batch_idx, valid_batch in enumerate(valid_data_loaders):
                if batch_idx >= eval_steps:
                    break
                logging.info(f" -----> Evaluating on batch {batch_idx + 1} of {eval_steps}")
                v_loss, eval_preds, e2e_acc = eval_step_grpo(valid_batch, device)
                if eval_preds is not None:
                    last_first_stage_predictions = eval_preds['first_stage']
                    last_z_predictions = eval_preds['z']
                    last_z_full = eval_preds.get('z_full', None)
                    last_true_z = eval_preds['true_z']
                    last_true_z_full = eval_preds.get('true_z_full', None)
                    last_L_predictions = eval_preds['L']
                    last_true_L = eval_preds['true_L']
                if not math.isnan(v_loss):
                    valid_losses.append(v_loss)
                if e2e_acc > 0:  # Only track if e2e accuracy was computed
                    valid_e2e_accs.append(e2e_acc)

            if valid_losses:
                valid_loss = sum(valid_losses) / len(valid_losses)
            else:
                valid_loss = float('nan')

            # Compute average end-to-end accuracy
            if valid_e2e_accs:
                valid_e2e_accuracy = sum(valid_e2e_accs) / len(valid_e2e_accs)
            else:
                valid_e2e_accuracy = 0.0

            logging.info(f"[VALID EPOCH {global_step}] Avg Validation Loss = {valid_loss:.4f}")
            logging.info(f"[VALID EPOCH {global_step}] Avg E2E Classification Accuracy = {valid_e2e_accuracy:.4f} ({valid_e2e_accuracy*100:.2f}%)")

            # ---- Save best model by ACCURACY (this is what matters at inference) ----
            if valid_e2e_accuracy > BEST_VALID_ACCURACY:
                BEST_VALID_ACCURACY = valid_e2e_accuracy
                save_ckpt(CKPT_DIR, f"best_acc_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
                logging.info(f"🎯 New best ACCURACY model saved at epoch {global_step} with E2E accuracy {valid_e2e_accuracy:.4f} ({valid_e2e_accuracy*100:.2f}%)")

            # ---- Early-stopping update (relative improvement) based on LOSS ----
            # Treat the run as "improved" only if it beats best by >= min_delta * best
            improved = (BEST_VALID_LOSS - valid_loss) > (EARLY_STOP_MIN_DELTA * max(BEST_VALID_LOSS, 1e-12))

            if improved or not math.isfinite(BEST_VALID_LOSS):
                BEST_VALID_LOSS = valid_loss
                epochs_no_improve = 0
                save_ckpt(CKPT_DIR, f"best_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
                logging.info(f"New best LOSS model saved at epoch {global_step} with validation loss {valid_loss:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"[EARLY-STOP] No loss improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE}). "
                            f"Best loss so far: {BEST_VALID_LOSS:.4f}, Best accuracy so far: {BEST_VALID_ACCURACY:.4f}")

                # DISABLED: Early stopping commented out to allow full training runs
                # if epochs_no_improve > EARLY_STOP_PATIENCE:
                #     logging.info(f"[EARLY-STOP] Stopping at epoch {global_step} "
                #                 f"(best validation = {BEST_VALID_LOSS:.4f}).")
                #     early_stop = True

        # Log learning rates (encoder + separate B optimizer)
        encoder_lr = encoder_optimizer.param_groups[0]['lr']
        b_lr = b_optimizer.param_groups[0]['lr']
        logging.info(f'[EPOCH {global_step}] - Encoder LR: {encoder_lr:.6f} | B LR: {b_lr:.7f}')

        global_step += 1

    # Save final checkpoint
        if last_first_stage_predictions is None and valid_data_loaders:
            logging.info("[FINAL EVAL] No evaluation predictions captured during training, running one batch to collect them.")
            for batch_idx, valid_batch in enumerate(valid_data_loaders):
                v_loss, eval_preds = eval_step_grpo(valid_batch, device)
                if eval_preds is None:
                    continue
                last_first_stage_predictions = eval_preds['first_stage']
                last_z_predictions = eval_preds['z']
                last_z_full = eval_preds.get('z_full', None)
                last_true_z = eval_preds['true_z']
                last_true_z_full = eval_preds.get('true_z_full', None)
                last_L_predictions = eval_preds['L']
                last_true_L = eval_preds['true_L']
                break
            if last_first_stage_predictions is None:
                logging.warning("[FINAL EVAL] Unable to capture predictions from validation loader; final prints will remain None.")
    save_ckpt(CKPT_DIR, f"final_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
    final_ckpt_path = os.path.join(CKPT_DIR, f"final_{SLURM_JOB_ID}.pt")
    logging.info(f"Training complete - final model saved to: {final_ckpt_path}")

    return (avg_epoch_loss, last_loss, valid_loss, last_first_stage_predictions, 
            last_z_predictions, last_z_full, last_L_predictions, 
            last_true_z, last_true_z_full, last_true_L)


def main(argv):
    try:
        log_available_resources()
        config = get_config()

        # ===== REPRODUCIBILITY SETUP =====
        # Get seed from environment variable or use default
        seed = int(os.getenv('SEED', '42'))
        set_reproducibility_seed(seed)

        # Get git information for tracking code version
        git_info = get_git_info()

        logging.info("\n" + "="*80)
        logging.info("🚀 HPC TRAINING SCRIPT WITH AUTO-RESUME")
        logging.info("="*80)
        logging.info("\n📌 CHECKPOINT/RESUME MECHANISM:")
        logging.info("   ✓ Automatically detects if job was requeued/stopped")
        logging.info("   ✓ Looks for latest checkpoint: <model_dir>/latest_<SLURM_JOB_ID>.pt")
        logging.info("   ✓ If found, resumes from saved epoch and continues")
        logging.info("   ✓ Also saves 'best_<SLURM_JOB_ID>.pt' (best validation loss)")
        logging.info("   ✓ Saves periodic checkpoints every 2 epochs")
        logging.info("   ✓ Final checkpoint: final_<SLURM_JOB_ID>.pt")
        logging.info("\n⚠️  HOW IT WORKS:")
        logging.info("   1. Each checkpoint includes:")
        logging.info("      - Model weights (handles DataParallel)")
        logging.info("      - Optimizer state (Adam momentum, etc.)")
        logging.info("      - Training counters (epoch, global_step)")
        logging.info("      - RNG states (torch, cuda, numpy) for reproducibility")
        logging.info("   2. On requeue, the job automatically resumes from latest_<SLURM_JOB_ID>.pt")
        logging.info("   3. Training loop picks up exactly where it left off")
        logging.info("   4. Learning rate schedulers are also restored")
        logging.info("\n🎯 TYPICAL WORKFLOW:")
        logging.info("   $ sbatch job.sh  # SLURM_JOB_ID=12345")
        logging.info("   [Training saves: latest_12345.pt every 2 epochs]")
        logging.info("   [Job hits time limit or gets preempted...]")
        logging.info("   $ sbatch job.sh  # SLURM_JOB_ID=12346 (new job ID from requeue)")
        logging.info("   [Detects latest_12345.pt, resumes training]")
        logging.info("   [Continues until completion or next preemption]")
        logging.info("\n💡 TIPS:")
        logging.info("   - Use 'sbatch --requeue job.sh' to enable automatic requeue")
        logging.info("   - Set 'export SLURM_JOB_ID=12345' manually to resume specific job")
        logging.info("   - Monitor logs for '🔄 RESUMING FROM CHECKPOINT' message")
        logging.info("="*80 + "\n")

        # Create GRPO config
        # grpo_config = GRPOConfig(
        #     num_candidates=FLAGS.grpo_candidates,
        #     temperature=FLAGS.grpo_temperature,
        #     sampling_dropout=FLAGS.grpo_sampling_dropout,
        #     accuracy_weight=FLAGS.grpo_accuracy_weight,
        #     infeasibility_weight=FLAGS.grpo_infeasibility_weight,
        # )

        grpo_config = GRPOConfig(
            # ====== STABILIZATION STRATEGY (CORRECTED) ======
            # Problem: importance ratios exploding → clipping saturates → zero gradients
            # Solution: Keep candidates closer + wider clipping + reduce learning rate
            # NOTE: KL coefficient was in config but NOT implemented - removed to focus on working levers
            
            num_candidates=4,
            
            # 1. TEMPERATURE (control candidate diversity)
            #    - Higher temp = softer distributions = less diverse candidates = smaller importance ratios
            #    - Rationale: If new and old policies are similar, ratios stay ~1.0, clipping less aggressive
            #    - Was: 1.0 → Now: 1.8 (flatter distribution, softer probabilities)
            temperature=1.8,
            
            # 2. SAMPLING DROPOUT (additional regularization)
            #    - Adds noise during candidate generation
            #    - Rationale: Reduces extreme candidates early (before importance ratio explosion)
            #    - Was: 0.15 → Now: 0.25 (25% dropout, more aggressive regularization)
            sampling_dropout=0.25,
            
            # 3. CLIP EPSILON (reduce clipping aggression with wider window)
            #    - Clipping range [1-ε, 1+ε] determines when gradient goes to zero
            #    - Rationale: Larger ε = more variation allowed before saturation
            #    - Was: 0.2 [0.8, 1.2] → Now: 0.4 [0.6, 1.4] (2x wider window)
            #    - Safe because temperature + dropout keep candidates closer
            clip_epsilon=0.4,
            
            # Other config
            accuracy_weight=1.0,
            infeasibility_weight=0.5,
            use_clipping=True,      # Keep PPO clipping enabled (but less aggressive due to wider window)
            loss_method='supervised_only',  # TEST MODE: Pure supervised learning (no GRPO)
            e2e_weight=0.5,
            
            # NOTE: kl_coef is in GRPOConfig dataclass but NOT used in loss computation
            # Keeping default (0.01) but it has no effect on training
            kl_coef=0.01,  # UNUSED - no KL divergence penalty implemented
        )
        
        # 4. LEARNING RATE (global stabilization - most important lever)
        #    - Reduce encoder LR from 0.001 to 0.0005 (50% reduction)
        #    - Smaller updates = more stable training, harder to trigger importance ratio explosion
        #    - This is the PRIMARY stabilization mechanism (replaces missing KL penalty)
        #    - Decoder LR: 0.0005 * 0.5 = 0.00025
        #    - B optimizer LR: 0.0005 * 0.1 = 0.00005
        #    - Set in config_train_pytorch2.py line 109

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ===== LOG FULL CONFIGURATION FOR REPRODUCIBILITY =====
        log_full_config(config, grpo_config, git_info, seed, device)

        (avg_loss, last_loss, valid_loss, last_first_stage_predictions, 
         last_z_predictions, last_z_full, last_L_predictions, 
         last_true_z, last_true_z_full, last_true_L) = train_and_evaluate_grpo(
            train_problems_datasets=config.train_problems_datasets,
            train_problems_outputs=config.train_problems_outputs,
            train_problems_linear_feats=config.train_problems_linear_feats,
            valid_problems_datasets=config.valid_problems_datasets,
            valid_problems_outputs=config.valid_problems_outputs,
            valid_problems_linear_feats=config.valid_problems_linear_feats,
            device=device,
            learning_rate=config.learning_rate,
            model_dir=config.work_unit_dir,
            decay_steps=config.decay_steps,
            num_train_steps=config.num_train_steps,
            num_train_run_steps=config.num_train_run_steps,
            eval_every_steps=config.eval_every_steps,
            eval_steps=config.eval_steps,
            grad_clip_norm=config.grad_clip_norm,
            model_config=config.model_config,
            grpo_config=grpo_config,
            routing_tau_start=config.routing_tau_start,
            routing_tau_end=config.routing_tau_end,
        )
        
        # Print final metrics
        print("========== FINAL TRAINING RESULTS (GRPO) ==========")
        print(f"Final Average Train Loss: {avg_loss}")
        print(f"Final Loss (last training batch loss): {last_loss}")
        print(f"Final Validation Loss: {valid_loss}")
        
        # Extract last sample from batch for first-stage predictions
        if last_first_stage_predictions is not None:
            print("\nFinal First-Stage Predictions (last sample in batch):")
            
            # A matrix: [P, Td] - show full matrix with argmax per column
            if 'a_full' in last_first_stage_predictions and last_first_stage_predictions['a_full']:
                a_full = last_first_stage_predictions['a_full'][-1]  # Last graph in batch
                a_argmax = last_first_stage_predictions['a'][-1]  # Last graph argmax
                print(f"  A (Feature Selection) - Probabilities [P x Td]:")
                for row_idx, row in enumerate(a_full):
                    formatted_row = [f"{val:.4f}" for val in row]
                    print(f"    Feature {row_idx}: [{', '.join(formatted_row)}]")
                print(f"  A (Selected features per decision node): {a_argmax}")
            
            # B vector: [Td] - split thresholds
            if 'b' in last_first_stage_predictions:
                b_vals = last_first_stage_predictions['b']
                formatted_b = [f"{val:.4f}" for val in b_vals]
                print(f"  B (Split Thresholds): [{', '.join(formatted_b)}]")
            
            # C matrix: [K, Tl] - show full matrix with argmax per column
            if 'c_full' in last_first_stage_predictions and last_first_stage_predictions['c_full']:
                c_full = last_first_stage_predictions['c_full'][-1]  # Last graph in batch
                c_argmax = last_first_stage_predictions['c'][-1]  # Last graph argmax
                print(f"  C (Class Assignment) - Probabilities [K x Tl]:")
                for row_idx, row in enumerate(c_full):
                    formatted_row = [f"{val:.4f}" for val in row]
                    print(f"    Class {row_idx}: [{', '.join(formatted_row)}]")
                print(f"  C (Selected class per leaf node): {c_argmax}")
            
            # D vector: [Td] - node activation
            if 'd' in last_first_stage_predictions:
                d_vals = last_first_stage_predictions['d']
                formatted_d = [f"{val:.4f}" for val in d_vals]
                print(f"  D (Node Activation): [{', '.join(formatted_d)}]")
        else:
            print("Final First-Stage Predictions: None")
        
        # Z matrix: [N, 4] - show full probability matrix for all samples
        print(f"\n--- Z Predictions (Leaf Assignment) ---")
        if last_z_full is not None:
            print(f"Z Predicted Probabilities [N x 4 leaves]:")
            for i, z_prob in enumerate(last_z_full):
                formatted_probs = [f"{p:.4f}" for p in z_prob]
                print(f"  Sample {i:2d}: [{', '.join(formatted_probs)}]")
        print(f"Z Predicted Classes (argmax): {last_z_predictions}")
        
        print(f"\n--- True Z Values ---")
        if last_true_z_full is not None:
            print(f"Z True One-Hot [N x 4 leaves]:")
            for i, z_true in enumerate(last_true_z_full):
                formatted_true = [f"{int(t):.0f}" for t in z_true]
                print(f"  Sample {i:2d}: [{', '.join(formatted_true)}]")
        print(f"Z True Classes (argmax): {last_true_z}")
        
        # L prediction
        print(f"\n--- L Predictions (Total Value) ---")
        print(f"Final L Predictions: {last_L_predictions}")
        print(f"True final L values: {last_true_L}")
        print("===================================================")
    except Exception as e:
        logging.exception("Fatal error occurred during training:")
        raise


if __name__ == '__main__':
    app.run(main)
