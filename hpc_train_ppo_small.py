# ============================================================================
# HPC Train with PPO - SIMPLIFIED MODEL (Explicit Tree Routing)
# ============================================================================
# PPO variant of hpc_train_grpo_small.py.
#
# Key differences from GRPO script:
# - Always uses log-prob importance-ratio objective (no loss_method switching)
# - PPO clipping always enabled (clip_epsilon from PPOConfig)
# - Inner epoch loop: N_inner_epochs PPO updates per candidate batch
# - KL-based early stopping in inner loop (approx_kl > target_kl * 1.5)
# - KL penalty added to loss (kl_coef * |mean log_ratio|)
# - No supervised auxiliary loss during PPO step (pure RL signal)
#   (eval loop still uses supervised metrics for comparison)
#
# Shared with GRPO script (copy):
# - GRPOCandidateSampler: candidate sampling + log_prob_old capture
# - GRPORewardComputer:   ODT reward function
# - DatasetGroupNormalizer: two-level advantage normalization
# - compute_standard_loss: used only in eval, not in PPO training step
# - eval_step_ppo: identical to eval_step_grpo
# ============================================================================

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import two_stage_data_utils3
import two_stage_gps_small
from config_train_pytorch2 import get_config
from two_stage_train_utils import (
    log_available_resources, save_ckpt, load_ckpt,
    signal_handler, flatten_selected_solution_data
)
from ppo_utils import PPOConfig, PPOLossComputer
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

MIN_E2E_WEIGHT = 8.0
MIN_L_WEIGHT = 0.8
E2E_PROB_FLOOR = 1e-3

ENTROPY_WEIGHT_A = 0.0
ENTROPY_WEIGHT_C = 0.0

stop_training = False


# ============================================================================
# Utilities (identical to hpc_train_grpo_small.py)
# ============================================================================

def compute_routing_temperature(global_step, num_train_steps, tau_start=2.0, tau_end=0.1):
    """Linear annealing of routing temperature."""
    progress = min(global_step / max(num_train_steps, 1), 1.0)
    return tau_start + (tau_end - tau_start) * progress


def get_git_info():
    git_info = {
        'commit_hash': 'unknown', 'commit_hash_short': 'unknown',
        'branch': 'unknown', 'is_dirty': False,
        'commit_date': 'unknown', 'commit_message': 'unknown',
    }
    try:
        for cmd, key in [
            (['git', 'rev-parse', 'HEAD'], 'commit_hash'),
            (['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 'branch'),
            (['git', 'log', '-1', '--format=%ci'], 'commit_date'),
            (['git', 'log', '-1', '--format=%s'], 'commit_message'),
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                git_info[key] = r.stdout.strip()[:80]
        git_info['commit_hash_short'] = git_info['commit_hash'][:8]
        r = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=5)
        git_info['is_dirty'] = len(r.stdout.strip()) > 0
    except Exception as e:
        logging.warning(f"Could not get git info: {e}")
    return git_info


def set_reproducibility_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")
    return seed


def log_full_config(config, ppo_config, git_info, seed, device):
    logging.info("=" * 80)
    logging.info("PPO TRAINING CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"PPO Config: {ppo_config}")
    logging.info(f"Device: {device}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Git: {git_info.get('branch')} @ {git_info.get('commit_hash_short')}")
    logging.info("=" * 80)


# ============================================================================
# GRPOCandidateSampler — verbatim from hpc_train_grpo_small.py
# Generates G diverse candidates with sampled actions + log_prob_old capture.
# ============================================================================

class GRPOCandidateSampler:
    def __init__(self, num_candidates=4, temperature=1.8, sampling_dropout=0.25, b_log_std=-1.0):
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.sampling_dropout = sampling_dropout
        self.b_log_std = b_log_std

    def _compute_log_prob_at_sampling(
        self, pred_a, pred_b, pred_c, pred_d, z_logits,
        actions_a, actions_b, actions_c, actions_d, actions_z,
    ):
        device = pred_d.device
        T = self.temperature

        log_prob_a_sum = torch.tensor(0.0, device=device)
        num_a = 0
        for logits, actions in zip(pred_a, actions_a):
            log_sm = F.log_softmax(logits / T, dim=0)
            for t in range(logits.shape[1]):
                log_prob_a_sum = log_prob_a_sum + log_sm[actions[t], t]
                num_a += 1
        log_prob_a = log_prob_a_sum / max(num_a, 1)

        b_std = math.exp(self.b_log_std)
        b_diff_sq = torch.clamp((actions_b - pred_b) ** 2, max=100.0)
        log_prob_b_per = (-0.5 * b_diff_sq / (b_std ** 2)
                         - self.b_log_std - 0.5 * math.log(2 * math.pi))
        log_prob_b_per = torch.clamp(log_prob_b_per, min=-50.0, max=0.0)
        log_prob_b = log_prob_b_per.mean()

        log_prob_c_sum = torch.tensor(0.0, device=device)
        num_c = 0
        for logits, actions in zip(pred_c, actions_c):
            log_sm = F.log_softmax(logits / T, dim=0)
            for t in range(logits.shape[1]):
                log_prob_c_sum = log_prob_c_sum + log_sm[actions[t], t]
                num_c += 1
        log_prob_c = log_prob_c_sum / max(num_c, 1)

        d_probs = torch.clamp(torch.sigmoid(pred_d / T), 1e-7, 1 - 1e-7)
        log_prob_d = (actions_d * torch.log(d_probs)
                     + (1 - actions_d) * torch.log(1 - d_probs)).mean()

        log_sm_z = F.log_softmax(z_logits / T, dim=-1)
        log_prob_z = log_sm_z.gather(dim=-1, index=actions_z.unsqueeze(-1)).squeeze(-1).mean()

        log_prob_a = torch.clamp(log_prob_a, min=-50.0, max=0.0)
        log_prob_b = torch.clamp(log_prob_b, min=-50.0, max=0.0)
        log_prob_c = torch.clamp(log_prob_c, min=-50.0, max=0.0)
        log_prob_d = torch.clamp(log_prob_d, min=-50.0, max=0.0)
        log_prob_z = torch.clamp(log_prob_z, min=-50.0, max=0.0)
        log_prob_total = torch.clamp(
            log_prob_a + log_prob_b + log_prob_c + log_prob_d + log_prob_z,
            min=-250.0, max=0.0
        )

        return {
            'log_prob_a_old': log_prob_a.detach(),
            'log_prob_b_old': log_prob_b.detach(),
            'log_prob_c_old': log_prob_c.detach(),
            'log_prob_d_old': log_prob_d.detach(),
            'log_prob_z_old': log_prob_z.detach(),
            'log_prob_total_old': log_prob_total.detach(),
        }

    def sample_candidates(self, model, inputs, device):
        candidates = []
        was_training = model.training
        model.train()
        target = model.module if hasattr(model, 'module') else model
        original_dropout = getattr(target, 'dropout', 0.0)
        self._set_dropout(model, self.sampling_dropout)

        try:
            for _ in range(self.num_candidates):
                with torch.no_grad():
                    fs_logits, z_logits, L_logits, all_L = model(*inputs)
                    pred_a, pred_b, pred_c, pred_d = fs_logits

                    pred_a_temp = [a / self.temperature for a in pred_a]
                    pred_c_temp = [c / self.temperature for c in pred_c]
                    z_logits_temp = z_logits / self.temperature

                    actions_a = []
                    for pa in pred_a_temp:
                        probs = F.softmax(pa, dim=0)
                        actions_a.append(torch.multinomial(probs.t(), num_samples=1).squeeze(-1))

                    actions_c = []
                    for pc in pred_c_temp:
                        probs = F.softmax(pc, dim=0)
                        actions_c.append(torch.multinomial(probs.t(), num_samples=1).squeeze(-1))

                    probs_d = torch.sigmoid(pred_d / self.temperature)
                    actions_d = torch.bernoulli(probs_d)

                    probs_z = F.softmax(z_logits_temp, dim=-1)
                    actions_z = torch.multinomial(probs_z, num_samples=1).squeeze(-1)

                    noise_b = torch.randn_like(pred_b) * math.exp(self.b_log_std)
                    actions_b = pred_b + noise_b

                    log_probs_old = self._compute_log_prob_at_sampling(
                        pred_a, pred_b, pred_c, pred_d, z_logits,
                        actions_a, actions_b, actions_c, actions_d, actions_z
                    )

                    old_a_probs = [F.softmax(a, dim=0).detach() for a in pred_a_temp]
                    old_b_values = pred_b.detach().clone()
                    old_c_probs = [F.softmax(c, dim=0).detach() for c in pred_c_temp]
                    old_d_probs = probs_d.detach()
                    old_z_probs = probs_z.detach()

                    candidates.append({
                        'pred_a': pred_a, 'pred_b': pred_b, 'pred_c': pred_c,
                        'pred_d': pred_d, 'pred_z': z_logits, 'pred_L': all_L,
                        'pred_a_temp': pred_a_temp, 'pred_c_temp': pred_c_temp,
                        'pred_z_temp': z_logits_temp,
                        'actions_a': actions_a, 'actions_b': actions_b,
                        'actions_c': actions_c, 'actions_d': actions_d, 'actions_z': actions_z,
                        **log_probs_old,
                        'old_a_probs': old_a_probs, 'old_b_values': old_b_values,
                        'old_c_probs': old_c_probs, 'old_d_probs': old_d_probs,
                        'old_z_probs': old_z_probs,
                    })
        finally:
            self._set_dropout(model, original_dropout)
            if not was_training:
                model.eval()

        return candidates

    def _set_dropout(self, model, rate):
        target = model.module if hasattr(model, 'module') else model
        if hasattr(target, 'dropout'):
            target.dropout = rate
        for module in target.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate


# ============================================================================
# GRPORewardComputer — verbatim from hpc_train_grpo_small.py
# ============================================================================

class GRPORewardComputer:
    def __init__(self, accuracy_weight=1.0, infeasibility_weight=0.5, norm_eps=1e-8):
        self.accuracy_weight = accuracy_weight
        self.infeasibility_weight = infeasibility_weight
        self.norm_eps = norm_eps

    def compute_reward(self, candidate, true_labels, variable_shapes, normalized_X=None):
        pred_a = candidate['pred_a']
        pred_b = candidate['pred_b']
        pred_c = candidate['pred_c']
        pred_d = candidate['pred_d']
        pred_z = candidate['pred_z']
        actions_a = candidate['actions_a']
        actions_b = candidate['actions_b']
        actions_c = candidate['actions_c']
        actions_d = candidate['actions_d']
        actions_z = candidate['actions_z']

        true_a_per_graph = true_labels['true_a_per_graph']
        true_c_per_graph = true_labels['true_c_per_graph']
        true_b = true_labels['true_b']
        true_d = true_labels['true_d']
        z_labels_indices = true_labels['z_labels_indices']

        device = pred_d.device
        accuracies = {}
        weights = {}

        a_correct, a_total = 0, 0
        for act, true_a in zip(actions_a, true_a_per_graph):
            true_idx = true_a.argmax(dim=0)
            a_correct += (act == true_idx).sum().item()
            a_total += act.numel()
        accuracies['acc_a'] = a_correct / max(a_total, 1)
        weights['acc_a'] = a_total

        b_tolerance = 0.2
        b_correct = ((actions_b - true_b).abs() < b_tolerance).sum().item()
        accuracies['acc_b'] = b_correct / max(actions_b.numel(), 1)
        weights['acc_b'] = actions_b.numel()

        c_correct, c_total = 0, 0
        for act, true_c in zip(actions_c, true_c_per_graph):
            true_idx = true_c.argmax(dim=0)
            c_correct += (act == true_idx).sum().item()
            c_total += act.numel()
        accuracies['acc_c'] = c_correct / max(c_total, 1)
        weights['acc_c'] = c_total

        true_d_binary = (true_d > 0.5).float()
        d_correct = (actions_d == true_d_binary).sum().item()
        accuracies['acc_d'] = d_correct / max(actions_d.numel(), 1)
        weights['acc_d'] = actions_d.numel()

        z_correct = (actions_z == z_labels_indices).sum().item()
        accuracies['acc_z'] = z_correct / max(actions_z.numel(), 1)
        weights['acc_z'] = actions_z.numel()

        total_w = sum(weights.values())
        weighted_acc = sum(accuracies[k] * weights[k] for k in accuracies) / max(total_w, 1)

        a_probs = F.softmax(torch.cat([a.view(-1) for a in pred_a]), dim=0) if pred_a else torch.tensor(0.0)
        d_probs = torch.sigmoid(pred_d)
        a_sum_per_node = []
        if pred_a and len(pred_a) > 0:
            for pa in pred_a:
                a_sum_per_node.append(F.softmax(pa, dim=0).sum(dim=0))
            d_per_node = d_probs[:a_sum_per_node[0].numel()] if d_probs.numel() >= a_sum_per_node[0].numel() else d_probs
            infeasibility = torch.stack([(s - d_per_node[:s.numel()]).abs().mean() for s in a_sum_per_node]).mean()
        else:
            infeasibility = torch.tensor(0.0, device=device)
        infeasibility = infeasibility.clamp(0, 1)

        reward_val = self.accuracy_weight * weighted_acc - self.infeasibility_weight * infeasibility.item()
        reward = torch.tensor(reward_val, device=device)

        routing_mismatch = 0.0
        if normalized_X is not None:
            z_from_tree = pred_z.argmax(dim=-1)
            routing_mismatch = (actions_z != z_from_tree).float().mean().item()

        details = {
            **accuracies,
            'weighted_accuracy': weighted_acc,
            'infeasibility': infeasibility.item(),
            'total_reward': reward.item(),
            'routing_mismatch': routing_mismatch,
        }
        return reward, details


# ============================================================================
# DatasetGroupNormalizer — verbatim from hpc_train_grpo_small.py
# ============================================================================

class DatasetGroupNormalizer:
    def __init__(self, norm_eps=1e-8, ema_alpha=0.1):
        self.norm_eps = norm_eps
        self.ema_alpha = ema_alpha
        self.dataset_stats = {}

    def update_running_stats(self, dataset_name, new_value):
        if dataset_name not in self.dataset_stats:
            self.dataset_stats[dataset_name] = {
                'count': 0, 'mean': new_value, 'var': 0.0,
                'ema_mean': new_value, 'ema_var': 0.0,
            }
        stats = self.dataset_stats[dataset_name]
        stats['count'] += 1
        n = stats['count']
        old_mean = stats['mean']
        stats['mean'] += (new_value - old_mean) / n
        stats['var'] += (new_value - old_mean) * (new_value - stats['mean'])
        alpha = self.ema_alpha
        stats['ema_mean'] = alpha * new_value + (1 - alpha) * stats['ema_mean']
        stats['ema_var'] = alpha * (new_value - stats['ema_mean']) ** 2 + (1 - alpha) * stats['ema_var']

    def get_dataset_baseline(self, dataset_name):
        if dataset_name not in self.dataset_stats:
            return 0.0, 1.0
        stats = self.dataset_stats[dataset_name]
        n = stats['count']
        mean = stats['ema_mean']
        std = math.sqrt(stats['ema_var'] + self.norm_eps)
        return mean, std

    def compute_normalized_advantages(self, rewards, dataset_name, update_stats=True):
        rewards_tensor = torch.stack(rewards)
        if update_stats:
            self.update_running_stats(dataset_name, rewards_tensor.mean().item())
        baseline_mean, baseline_std = self.get_dataset_baseline(dataset_name)
        normalized = (rewards_tensor - baseline_mean) / max(baseline_std, self.norm_eps)
        mean_r = normalized.mean()
        std_r = normalized.std()
        advantages = (normalized - mean_r) / (std_r + self.norm_eps)
        details = {
            'dataset_name': dataset_name,
            'dataset_baseline_mean': baseline_mean,
            'dataset_baseline_std': baseline_std,
            'dataset_sample_count': self.dataset_stats.get(dataset_name, {}).get('count', 0),
            'group_mean_reward': mean_r.item(),
            'group_std_reward': std_r.item(),
        }
        return advantages, details

    def get_dataset_names_from_batch(self, states):
        names = []
        for s in states:
            meta = s.get('meta_info', {})
            name = meta.get('dataset_name', meta.get('dataset', 'unknown'))
            names.append(name)
        return names

    def get_all_dataset_stats(self):
        return {
            name: {
                'count': stats['count'],
                'mean': stats['mean'],
                'std': math.sqrt(stats['var'] / max(stats['count'] - 1, 1) + self.norm_eps),
                'ema_mean': stats['ema_mean'],
                'ema_std': math.sqrt(stats['ema_var'] + self.norm_eps),
            }
            for name, stats in self.dataset_stats.items()
        }


# ============================================================================
# Main training function
# ============================================================================

def train_and_evaluate_ppo(
    train_problems_datasets, train_problems_outputs, train_problems_linear_feats,
    valid_problems_datasets, valid_problems_outputs, valid_problems_linear_feats,
    device, learning_rate, model_dir, decay_steps, num_train_steps,
    num_train_run_steps, eval_every_steps, eval_steps, grad_clip_norm,
    model_config, ppo_config: PPOConfig,
    routing_tau_start=2.0, routing_tau_end=0.1,
):
    global stop_training
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "none")

    logging.info(f'[PPO] Loading data...')
    logging.info(f'[PPO] Config: {ppo_config}')

    num_files = int(os.getenv('NUM_FILES', '100'))
    batch_size_env = os.getenv('BATCH_SIZE')
    if batch_size_env is not None:
        batch_size = int(batch_size_env)
    else:
        if num_files == 100:
            batch_size = 10
        elif num_files in (250, 500):
            batch_size = num_files // 10
        elif num_files == 1000:
            batch_size = 50
        else:
            batch_size = max(10, num_files // 15)

    datasets_env = os.getenv("DATASETS", "glass,small_toy")
    dataset_names = [d.strip() for d in datasets_env.split(',')]
    n_datasets = len(dataset_names)
    files_per_dataset = num_files // n_datasets

    def select_balanced_files(file_list, ds_names, per_dataset):
        from collections import defaultdict
        buckets = defaultdict(list)
        for f in file_list:
            for ds in ds_names:
                if f'/{ds}/' in f:
                    buckets[ds].append(f)
                    break
        selected = []
        for ds in ds_names:
            selected.extend(buckets[ds][:per_dataset])
        logging.info(f"[BALANCED] Selected {len(selected)} files: " +
                     ", ".join(f"{ds}={len(buckets[ds][:per_dataset])}" for ds in ds_names))
        return selected

    train_datasets_sel = select_balanced_files(train_problems_datasets[0][0], dataset_names, files_per_dataset)
    train_outputs_sel  = select_balanced_files(train_problems_outputs[0][0],  dataset_names, files_per_dataset)
    train_linear_sel   = select_balanced_files(train_problems_linear_feats[0][0], dataset_names, files_per_dataset)

    train_data_loaders = two_stage_data_utils3.get_dataset(
        train_datasets_sel, train_outputs_sel, train_linear_sel, batch_size=batch_size
    )
    logging.info(f'Train datasets loaded: {len(train_data_loaders)} batches')

    valid_per_dataset = max(1, 10 // n_datasets)
    valid_datasets_sel = select_balanced_files(valid_problems_datasets[0][0], dataset_names, valid_per_dataset)
    valid_outputs_sel  = select_balanced_files(valid_problems_outputs[0][0],  dataset_names, valid_per_dataset)
    valid_linear_sel   = select_balanced_files(valid_problems_linear_feats[0][0], dataset_names, valid_per_dataset)

    valid_data_loaders = two_stage_data_utils3.get_dataset(
        valid_datasets_sel, valid_outputs_sel, valid_linear_sel, batch_size=1
    )
    logging.info(f'Valid datasets loaded: {len(valid_data_loaders)}')

    # =========================================================================
    # Model + optimizer setup (identical to GRPO script)
    # =========================================================================

    model = two_stage_gps_small.get_model(**model_config.params).to(device)
    torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    target = model.module if isinstance(model, nn.DataParallel) else model

    encoder_params = [
        {'params': target.pos_encoding.parameters(), 'weight_decay': 0.0},
        {'params': target.feature_attention.parameters(), 'weight_decay': 0.0},
        {'params': target.first_linear.parameters(), 'weight_decay': 0.0},
        {'params': target.constraint_linear.parameters(), 'weight_decay': 0.0},
        {'params': target.first_layers.parameters(), 'weight_decay': 0.0},
        {'params': target.a_head.parameters(), 'weight_decay': 0.0},
        {'params': target.c_head.parameters(), 'weight_decay': 0.0},
        {'params': target.d_head.parameters(), 'weight_decay': 0.0},
    ]
    b_params = [{'params': target.b_head.parameters(), 'weight_decay': 0.0}]

    B_LR_SCALE = 0.01
    encoder_optimizer = optim.AdamW(encoder_params, lr=learning_rate)
    b_optimizer = optim.AdamW(b_params, lr=learning_rate * B_LR_SCALE)
    loss_weight_optimizer = optim.AdamW([target.log_loss_weights], lr=min(1e-4, learning_rate * 0.05))

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=300, gamma=0.5)
    b_scheduler = optim.lr_scheduler.StepLR(b_optimizer, step_size=300, gamma=0.5)

    with torch.no_grad():
        fixed_values = [math.log(9.0), math.log(5.0), math.log(1.0)]
        target.log_loss_weights.data = torch.tensor(fixed_values, device=target.log_loss_weights.device)
        target.log_loss_weights.requires_grad = True

    FS_WEIGHT_SUM = float(math.exp(fixed_values[0]))
    prior_log_weights = torch.tensor([fixed_values[0]], device=device)

    # =========================================================================
    # PPO-specific components
    # =========================================================================

    b_log_std = ppo_config.b_log_std

    candidate_sampler = GRPOCandidateSampler(
        num_candidates=ppo_config.num_candidates,
        temperature=ppo_config.temperature,
        sampling_dropout=ppo_config.sampling_dropout,
        b_log_std=b_log_std,
    )
    reward_computer = GRPORewardComputer(
        accuracy_weight=ppo_config.accuracy_weight,
        infeasibility_weight=ppo_config.infeasibility_weight,
    )
    ppo_loss_computer = PPOLossComputer(
        device=device,
        clip_epsilon=ppo_config.clip_epsilon,
        kl_coef=ppo_config.kl_coef,
        temperature=ppo_config.temperature,
        b_log_std=b_log_std,
        entropy_coef=ppo_config.entropy_coef,
    )
    dataset_normalizer = DatasetGroupNormalizer()

    B_GRAD_CLIP = 80.0
    A_GRAD_CLIP = 100.0

    # =========================================================================
    # compute_standard_loss — used in eval (not in PPO training step)
    # =========================================================================

    def compute_standard_loss(fs_logits, z_logits, L_logits, all_L, true_labels):
        pred_a, pred_b, pred_c, pred_d = fs_logits
        true_a_per_graph = true_labels['true_a_per_graph']
        true_c_per_graph = true_labels['true_c_per_graph']
        true_b = true_labels['true_b']
        true_d = true_labels['true_d']
        z_labels_indices = true_labels['z_labels_indices']
        L_labels_per_batch = true_labels['L_labels_per_batch']

        loss_a = 0
        num_a_cols = 0
        entropy_a = 0
        for pred_a_graph, true_a_graph in zip(pred_a, true_a_per_graph):
            for col in range(pred_a_graph.shape[1]):
                loss_a += F.cross_entropy(
                    pred_a_graph[:, col].unsqueeze(0),
                    true_a_graph[:, col].argmax().unsqueeze(0)
                )
                num_a_cols += 1
            a_probs = F.softmax(pred_a_graph, dim=0)
            entropy_a += (-(a_probs * torch.log(a_probs + 1e-8)).sum(dim=0)).mean()
        loss_a = loss_a / max(num_a_cols, 1) * 10.0
        entropy_a = entropy_a / max(len(pred_a), 1)

        loss_b = F.mse_loss(pred_b, true_b, reduction='mean')

        loss_c = 0
        num_c_cols = 0
        entropy_c = 0
        for pred_c_graph, true_c_graph in zip(pred_c, true_c_per_graph):
            for col in range(pred_c_graph.shape[1]):
                loss_c += F.cross_entropy(
                    pred_c_graph[:, col].unsqueeze(0),
                    true_c_graph[:, col].argmax().unsqueeze(0)
                )
                num_c_cols += 1
            c_probs = F.softmax(pred_c_graph, dim=0)
            entropy_c += (-(c_probs * torch.log(c_probs + 1e-8)).sum(dim=0)).mean()
        loss_c = loss_c / max(num_c_cols, 1) + ENTROPY_WEIGHT_C * entropy_c / max(len(pred_c), 1)
        entropy_c = entropy_c / max(len(pred_c), 1)

        loss_d = F.binary_cross_entropy_with_logits(pred_d, true_d, reduction='mean') * 0.1
        first_stage_loss = loss_a + loss_b + loss_c + loss_d

        e2e_loss = torch.tensor(0.0, device=z_logits.device)
        sample_true_labels = true_labels.get('sample_true_labels')
        e2e_diag = {k: 0 for k in [
            'total_samples', 'correct_samples', 'floor_hits',
            'route_correct', 'route_total', 'leaf_correct', 'leaf_total',
            'route_only_errors', 'leaf_only_errors', 'both_errors', 'other_errors',
        ]}
        e2e_diag['true_prob_sum'] = 0.0
        e2e_diag['z_margin_sum'] = 0.0; e2e_diag['z_margin_count'] = 0
        e2e_diag['c_margin_sum'] = 0.0; e2e_diag['c_margin_count'] = 0

        if sample_true_labels is not None:
            true_labels_idx = sample_true_labels.long() - 1
            z_soft = F.softmax(z_logits, dim=-1)
            z_hard_idx = z_soft.argmax(dim=-1)
            if z_soft.shape[1] >= 2:
                z_top2 = torch.topk(z_soft, k=2, dim=-1).values
                z_margins = z_top2[:, 0] - z_top2[:, 1]
                e2e_diag['z_margin_sum'] += z_margins.sum().item()
                e2e_diag['z_margin_count'] += z_margins.numel()

            z_labels_list = true_labels.get('z_labels_list', [])
            if len(z_labels_list) > 0 and len(pred_c) == len(z_labels_list):
                expected_class_probs_list = []
                z_offset = 0
                for graph_idx, (pred_c_graph, z_lbl) in enumerate(zip(pred_c, z_labels_list)):
                    N_i = z_lbl.numel() // 4
                    if N_i > 0:
                        true_labels_idx_i = true_labels_idx[z_offset:z_offset + N_i]
                        z_pred_i = z_hard_idx[z_offset:z_offset + N_i]
                        z_probs_i = z_soft[z_offset:z_offset + N_i]
                        c_soft_i = F.softmax(pred_c_graph, dim=0)
                        if c_soft_i.shape[0] >= 2:
                            c_top2 = torch.topk(c_soft_i, k=2, dim=0).values
                            c_margins = c_top2[0] - c_top2[1]
                            e2e_diag['c_margin_sum'] += c_margins.sum().item()
                            e2e_diag['c_margin_count'] += c_margins.numel()
                        c_hard_idx = c_soft_i.argmax(dim=0)
                        c_hard = F.one_hot(c_hard_idx, num_classes=c_soft_i.shape[0]).float().transpose(0, 1)
                        c_st = c_hard + (c_soft_i - c_soft_i.detach())
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
                        log_exp_class_probs_i = torch.log(exp_class_probs_i)
                        expected_class_probs_list.append(log_exp_class_probs_i)
                        z_offset += N_i
                if expected_class_probs_list:
                    log_exp_class_probs = torch.cat(expected_class_probs_list, dim=0)
                    e2e_loss = F.nll_loss(log_exp_class_probs, true_labels_idx)

        L_loss = F.smooth_l1_loss(all_L, L_labels_per_batch, reduction='mean')

        raw_weights = torch.exp(target.log_loss_weights)
        fs_weight = raw_weights[0] / (raw_weights[0] + 1e-8) * FS_WEIGHT_SUM
        weights = torch.cat([fs_weight.unsqueeze(0), raw_weights[1:3]], dim=0)
        lambda_w = 5e-4
        weight_reg = F.mse_loss(target.log_loss_weights[0:1], prior_log_weights)
        total_loss = (
            weights[0] * first_stage_loss +
            weights[1] * e2e_loss +
            weights[2] * L_loss
        ) + lambda_w * weight_reg

        e2e_total = max(e2e_diag['total_samples'], 1)
        loss_components = {
            'loss_a': loss_a.item() if torch.is_tensor(loss_a) else loss_a,
            'loss_b': loss_b.item() if torch.is_tensor(loss_b) else loss_b,
            'loss_c': loss_c.item() if torch.is_tensor(loss_c) else loss_c,
            'loss_d': loss_d.item() if torch.is_tensor(loss_d) else loss_d,
            'e2e_loss': e2e_loss.item() if torch.is_tensor(e2e_loss) else e2e_loss,
            'L_loss': L_loss.item(),
            'entropy_a': entropy_a.item() if torch.is_tensor(entropy_a) else float(entropy_a),
            'entropy_c': entropy_c.item() if torch.is_tensor(entropy_c) else float(entropy_c),
            'e2e_diag_total': int(e2e_diag['total_samples']),
            'e2e_true_prob_mean': e2e_diag['true_prob_sum'] / e2e_total,
            'e2e_floor_frac': e2e_diag['floor_hits'] / e2e_total,
            'e2e_route_acc': e2e_diag['route_correct'] / max(e2e_diag['route_total'], 1),
            'e2e_z_margin_mean': e2e_diag['z_margin_sum'] / max(e2e_diag['z_margin_count'], 1),
            'e2e_c_margin_mean': e2e_diag['c_margin_sum'] / max(e2e_diag['c_margin_count'], 1),
        }
        return total_loss, loss_components

    # =========================================================================
    # PPO training step — KEY DIFFERENCE from GRPO
    # =========================================================================

    def train_step_ppo(train_loader, epoch_idx, global_step):
        """PPO training step with inner epoch loop and KL early stopping."""
        global stop_training
        model.train()

        first_stage_graph = train_loader[1].to(device)
        states = train_loader[0]
        second_stage_states = train_loader[4]

        first_stage_variable_indices = [s['first_stage_variable_indices'].to(device) for s in states]
        variable_shapes = [s['variable_shapes'] for s in states]
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

        true_a_per_graph, true_c_per_graph = [], []
        a_offset, c_offset = 0, 0
        for shapes in variable_shapes:
            P, Td = shapes[0]; K, Tl = shapes[2]
            a_size, c_size = P * Td, K * Tl
            true_a_per_graph.append(true_a[a_offset:a_offset + a_size].view(P, Td))
            true_c_per_graph.append(true_c[c_offset:c_offset + c_size].view(K, Tl))
            a_offset += a_size; c_offset += c_size

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

        inputs = (
            first_stage_graph, True, first_stage_variable_indices,
            variable_shapes, states, device
        )

        true_labels = {
            'true_a_per_graph': true_a_per_graph, 'true_b': true_b,
            'true_c_per_graph': true_c_per_graph, 'true_d': true_d,
            'z_labels_indices': z_labels_indices, 'L_labels_per_batch': L_labels_per_batch,
            'z_labels_list': z_labels_list,
        }

        normalized_X_list = [s['normalized_X'].to(device) for s in states if 'normalized_X' in s]
        if normalized_X_list:
            max_p = max(x.shape[1] for x in normalized_X_list)
            padded = []
            for x in normalized_X_list:
                if x.shape[1] < max_p:
                    x = F.pad(x, (0, max_p - x.shape[1]))
                padded.append(x)
            normalized_X = torch.cat(padded, dim=0)
        else:
            normalized_X = None

        sample_true_labels_list = [s['sample_true_labels'].to(device) for s in states if 'sample_true_labels' in s]
        sample_true_labels = torch.cat(sample_true_labels_list, dim=0) if sample_true_labels_list else None
        true_labels['sample_true_labels'] = sample_true_labels

        dataset_names_batch = dataset_normalizer.get_dataset_names_from_batch(states)
        current_dataset = dataset_names_batch[0] if dataset_names_batch else 'unknown'

        # ===== Step 1: Sample candidates once (no_grad) =====
        candidates = candidate_sampler.sample_candidates(model, inputs, device)

        rewards, reward_details = [], []
        for cand in candidates:
            reward, details = reward_computer.compute_reward(
                cand, true_labels, variable_shapes, normalized_X=normalized_X
            )
            rewards.append(reward)
            reward_details.append(details)

        advantages, norm_details = dataset_normalizer.compute_normalized_advantages(
            rewards, current_dataset, update_stats=True
        )

        avg_reward = sum(r.item() for r in rewards) / max(len(rewards), 1)
        best_reward = max(r.item() for r in rewards)

        # ===== Step 2: Inner PPO epoch loop =====
        inner_epochs_used = 0
        last_approx_kl = 0.0
        last_ppo_metrics = {}

        for inner_epoch in range(ppo_config.n_inner_epochs):
            encoder_optimizer.zero_grad()
            b_optimizer.zero_grad()
            loss_weight_optimizer.zero_grad()

            ppo_loss, ppo_metrics = ppo_loss_computer.ppo_step(
                model, inputs, candidates, advantages
            )

            approx_kl = ppo_metrics['approx_kl']

            # KL early stopping (skip check on first inner epoch — ratio=1.0)
            if inner_epoch > 0 and approx_kl > ppo_config.target_kl * 1.5:
                logging.info(
                    f"[PPO] Early stop inner_epoch={inner_epoch}: "
                    f"approx_kl={approx_kl:.4f} > {ppo_config.target_kl * 1.5:.4f}"
                )
                break

            if torch.isnan(ppo_loss) or torch.isinf(ppo_loss):
                logging.warning(f"[PPO] NaN/Inf in PPO loss at inner_epoch={inner_epoch}, skipping")
                break

            ppo_loss.backward()

            target_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.nn.utils.clip_grad_norm_(target_model.b_head.parameters(), B_GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(target_model.a_head.parameters(), A_GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            encoder_optimizer.step()
            b_optimizer.step()
            loss_weight_optimizer.step()

            inner_epochs_used += 1
            last_approx_kl = approx_kl
            last_ppo_metrics = ppo_metrics

        if inner_epochs_used == 0:
            logging.warning(f"[PPO] No inner epochs completed at step {global_step}, skipping batch")
            return None

        # ===== Logging =====
        pg_loss = last_ppo_metrics.get('policy_loss', 0.0)
        kl = last_ppo_metrics.get('kl', 0.0)
        kl_penalty = last_ppo_metrics.get('kl_penalty', 0.0)
        entropy = last_ppo_metrics.get('entropy', 0.0)
        clip_frac = last_ppo_metrics.get('clip_frac', 0.0)
        avg_ratio = last_ppo_metrics.get('avg_ratio', 1.0)

        logging.info(
            f"[PPO] Epoch {epoch_idx} | Step {global_step} | "
            f"PPO_loss={ppo_loss.item():.4f} | "
            f"policy_loss={pg_loss:.4f} | kl={kl:.4f} | entropy={entropy:.4f}"
        )
        logging.info(
            f"[PPO] ratio={avg_ratio:.4f} | clip_frac={clip_frac:.2%} | "
            f"approx_kl={last_approx_kl:.4f} | inner_epochs={inner_epochs_used}/{ppo_config.n_inner_epochs}"
        )
        logging.info(
            f"[PPO] Avg Reward={avg_reward:.4f} | Best Reward={best_reward:.4f} | "
            f"Dataset='{current_dataset}' | "
            f"baseline_mean={norm_details['dataset_baseline_mean']:.4f}"
        )
        if reward_details:
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i].item())
            bd = reward_details[best_idx]
            logging.info(
                f"[PPO] Best candidate: acc_a={bd.get('acc_a', 0):.3f}, "
                f"acc_b={bd.get('acc_b', 0):.3f}, acc_c={bd.get('acc_c', 0):.3f}, "
                f"acc_d={bd.get('acc_d', 0):.3f}, acc_z={bd.get('acc_z', 0):.3f}"
            )

        return ppo_loss.item()

    # =========================================================================
    # Eval step — identical to eval_step_grpo; uses supervised metrics
    # =========================================================================

    def eval_step_ppo(valid_loader, device):
        model.eval()
        torch.cuda.empty_cache()

        try:
            states = valid_loader[0]
            first_stage_graph = valid_loader[1].to(device)
            second_stage_states = valid_loader[4]
        except Exception as e:
            logging.warning(f"[VALID] Invalid batch: {e}")
            return float('nan'), None, 0.0

        first_stage_variable_indices = [s['first_stage_variable_indices'].to(device) for s in states]
        variable_shapes = [s['variable_shapes'] for s in states]
        variable_labels = [flatten_selected_solution_data(s['solution_data']).to(device) for s in states]

        sample_true_labels_list = [s['sample_true_labels'].to(device) for s in states if 'sample_true_labels' in s]
        sample_true_labels = torch.cat(sample_true_labels_list, dim=0) if sample_true_labels_list else None

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

        true_a_per_graph, true_c_per_graph = [], []
        a_offset, c_offset = 0, 0
        for shapes in variable_shapes:
            P, Td = shapes[0]; K, Tl = shapes[2]
            a_size, c_size = P * Td, K * Tl
            true_a_per_graph.append(true_a[a_offset:a_offset + a_size].view(P, Td))
            true_c_per_graph.append(true_c[c_offset:c_offset + c_size].view(K, Tl))
            a_offset += a_size; c_offset += c_size

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

        inputs = (
            first_stage_graph, False, first_stage_variable_indices,
            variable_shapes, states, device
        )
        true_labels = {
            'true_a_per_graph': true_a_per_graph, 'true_b': true_b,
            'true_c_per_graph': true_c_per_graph, 'true_d': true_d,
            'z_labels_indices': z_labels_indices, 'L_labels_per_batch': L_labels_per_batch,
            'z_labels_list': z_labels_list, 'sample_true_labels': sample_true_labels,
        }

        with torch.no_grad():
            val_target = model.module if isinstance(model, nn.DataParallel) else model
            saved_routing_temp = val_target.routing_temperature
            val_target.routing_temperature = 0.1

            fs_logits, z_logits, L_logits, all_L = model(*inputs)
            pred_a, pred_b, pred_c, pred_d = fs_logits

            loss_a = sum(
                F.cross_entropy(pa[:, c].unsqueeze(0), ta[:, c].argmax().unsqueeze(0))
                for pa, ta in zip(pred_a, true_a_per_graph)
                for c in range(pa.shape[1])
            ) / max(sum(pa.shape[1] for pa in pred_a), 1)
            loss_b = F.mse_loss(pred_b, true_b)
            loss_c = sum(
                F.cross_entropy(pc[:, c].unsqueeze(0), tc[:, c].argmax().unsqueeze(0))
                for pc, tc in zip(pred_c, true_c_per_graph)
                for c in range(pc.shape[1])
            ) / max(sum(pc.shape[1] for pc in pred_c), 1)
            loss_d = F.binary_cross_entropy_with_logits(pred_d, true_d) * 0.1
            first_stage_loss = loss_a + loss_b + loss_c + loss_d
            L_loss = F.smooth_l1_loss(all_L, L_labels_per_batch)

            # Element-wise accuracy
            acc_a = sum(
                (pa.argmax(dim=0) == ta.argmax(dim=0)).sum().item()
                for pa, ta in zip(pred_a, true_a_per_graph)
            ) / max(sum(pa.shape[1] for pa in pred_a), 1)
            acc_b = ((pred_b - true_b).abs() < 0.1).float().mean().item()
            acc_c = sum(
                (pc.argmax(dim=0) == tc.argmax(dim=0)).sum().item()
                for pc, tc in zip(pred_c, true_c_per_graph)
            ) / max(sum(pc.shape[1] for pc in pred_c), 1)
            acc_d = ((torch.sigmoid(pred_d) > 0.5).float() == (true_d > 0.5).float()).float().mean().item()
            acc_z = (z_logits.argmax(dim=-1) == z_labels_indices).float().mean().item()
            uniform_acc = (acc_a + acc_b + acc_c + acc_d + acc_z) / 5.0

            # End-to-end classification accuracy
            e2e_accuracy = 0.0
            if sample_true_labels is not None:
                z_pred_idx = z_logits.argmax(dim=-1)
                pred_classes_list = []
                sample_offset = 0
                for pc, z_lbl in zip(pred_c, z_labels_list):
                    n_samples = z_lbl.numel() // 4
                    if n_samples > 0:
                        graph_z_pred = z_pred_idx[sample_offset:sample_offset + n_samples]
                        for leaf_idx in graph_z_pred:
                            leaf_clamped = min(leaf_idx.item(), pc.shape[1] - 1)
                            pred_classes_list.append(pc[:, leaf_clamped].argmax().item())
                        sample_offset += n_samples
                if pred_classes_list:
                    pred_tensor = torch.tensor(pred_classes_list, device=device) + 1
                    true_adj = sample_true_labels[:len(pred_classes_list)].long()
                    e2e_accuracy = (pred_tensor == true_adj).float().mean().item()
                    logging.info(f"[VALID] E2E accuracy: {e2e_accuracy:.4f} ({(pred_tensor == true_adj).sum().item()}/{len(pred_classes_list)})")

            raw_weights = torch.exp(val_target.log_loss_weights)
            fs_weight = raw_weights[0] / (raw_weights[0] + 1e-8) * FS_WEIGHT_SUM
            weights = torch.cat([fs_weight.unsqueeze(0), raw_weights[1:3]], dim=0)
            total_loss = weights[0] * first_stage_loss + weights[2] * L_loss

            logging.info(
                f"[VALID] loss_a={loss_a.item():.4f}, loss_b={loss_b.item():.4f}, "
                f"loss_c={loss_c.item():.4f}, loss_d={loss_d.item():.4f}, L={L_loss.item():.4f}"
            )
            logging.info(
                f"[VALID] acc: a={acc_a:.3f}, b={acc_b:.3f}, c={acc_c:.3f}, "
                f"d={acc_d:.3f}, z={acc_z:.3f} | uniform={uniform_acc:.3f}"
            )

            a_probs = [F.softmax(pa, dim=0).detach().cpu() for pa in pred_a]
            c_probs = [F.softmax(pc, dim=0).detach().cpu() for pc in pred_c]
            predictions = {
                'first_stage': {
                    'a': [pa.argmax(dim=0).tolist() for pa in a_probs],
                    'a_full': [pa.tolist() for pa in a_probs],
                    'b': pred_b.detach().cpu().tolist(),
                    'c': [pc.argmax(dim=0).tolist() for pc in c_probs],
                    'c_full': [pc.tolist() for pc in c_probs],
                    'd': torch.sigmoid(pred_d).detach().cpu().tolist(),
                },
                'z': z_logits.argmax(dim=-1).detach().cpu().tolist(),
                'z_full': F.softmax(z_logits, dim=-1).detach().cpu().tolist(),
                'true_z': z_labels_indices.detach().cpu().tolist(),
                'true_z_full': z_labels.detach().cpu().tolist(),
                'L': all_L.detach().cpu().tolist(),
                'true_L': L_labels_per_batch.detach().cpu().tolist(),
                'e2e_accuracy': e2e_accuracy,
            }
            val_target.routing_temperature = saved_routing_temp

        return total_loss.item(), predictions, e2e_accuracy

    # =========================================================================
    # Checkpoint resume
    # =========================================================================

    latest_ckpt_path = os.path.join(model_dir, f"latest_{SLURM_JOB_ID}.pt")
    current_global = 0
    global_step = 0
    start_epoch = 0
    BEST_VALID_LOSS = float('inf')
    BEST_VALID_ACCURACY = 0.0
    EARLY_STOP_PATIENCE = 3
    EARLY_STOP_MIN_DELTA = 0.01
    epochs_no_improve = 0
    early_stop = False

    if os.path.exists(latest_ckpt_path):
        logging.info("RESUMING FROM CHECKPOINT")
        try:
            start_epoch, global_step = load_ckpt(latest_ckpt_path, model, encoder_optimizer, device)
            logging.info(f"Resumed: epoch={start_epoch}, global_step={global_step}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            start_epoch = global_step = 0
    else:
        logging.info(f"No checkpoint at {latest_ckpt_path} — starting fresh")

    CKPT_DIR = model_dir
    CKPT_EVERY = 2

    last_loss = 0.0
    valid_loss = float('nan')
    last_first_stage_predictions = None
    last_z_predictions = None
    last_z_full = None
    last_L_predictions = None
    last_true_z = None
    last_true_z_full = None
    last_true_L = None
    avg_epoch_loss = 0.0

    signal.signal(signal.SIGUSR1, signal_handler)

    # =========================================================================
    # Main training loop
    # =========================================================================

    while global_step < num_train_steps and not stop_training and not early_stop:
        current_routing_temp = compute_routing_temperature(
            global_step, num_train_steps,
            tau_start=routing_tau_start, tau_end=routing_tau_end
        )
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        target_model.routing_temperature = current_routing_temp

        start = time.time()
        epoch_losses = []

        for train_loader in train_data_loaders:
            loss = train_step_ppo(train_loader, global_step, current_global)
            if loss is not None:
                epoch_losses.append(loss)
                last_loss = loss
            current_global += 1
            if stop_training:
                break

        if stop_training:
            logging.info("SIGUSR1 received — saving checkpoint")
            save_ckpt(CKPT_DIR, f"latest_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
            break

        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        elapsed = time.time() - start
        logging.info(
            f"======== [PPO EPOCH {global_step}/{num_train_steps}] "
            f"Avg Loss={avg_epoch_loss:.4f} | tau={current_routing_temp:.4f} | {elapsed:.2f}s ========"
        )

        if encoder_scheduler is not None:
            encoder_scheduler.step()
        if b_scheduler is not None:
            b_scheduler.step()

        all_dataset_stats = dataset_normalizer.get_all_dataset_stats()
        if all_dataset_stats:
            for ds_name, ds_stats in sorted(all_dataset_stats.items()):
                logging.info(
                    f"[DATASET-NORM] {ds_name}: count={ds_stats['count']}, "
                    f"ema_mean={ds_stats['ema_mean']:.4f}, ema_std={ds_stats['ema_std']:.4f}"
                )

        if global_step % CKPT_EVERY == 0:
            save_ckpt(CKPT_DIR, f"latest_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)

        if global_step % eval_every_steps == 0:
            logging.info(f"======== [VALID EPOCH {global_step}] ========")
            valid_losses, valid_e2e_accs = [], []

            for batch_idx, valid_batch in enumerate(valid_data_loaders):
                if batch_idx >= eval_steps:
                    break
                v_loss, eval_preds, e2e_acc = eval_step_ppo(valid_batch, device)
                if eval_preds is not None:
                    last_first_stage_predictions = eval_preds['first_stage']
                    last_z_predictions = eval_preds['z']
                    last_z_full = eval_preds.get('z_full')
                    last_true_z = eval_preds['true_z']
                    last_true_z_full = eval_preds.get('true_z_full')
                    last_L_predictions = eval_preds['L']
                    last_true_L = eval_preds['true_L']
                if not math.isnan(v_loss):
                    valid_losses.append(v_loss)
                if e2e_acc > 0:
                    valid_e2e_accs.append(e2e_acc)

            valid_loss = sum(valid_losses) / max(len(valid_losses), 1) if valid_losses else float('nan')
            valid_e2e_accuracy = sum(valid_e2e_accs) / max(len(valid_e2e_accs), 1) if valid_e2e_accs else 0.0

            logging.info(f"[VALID EPOCH {global_step}] Avg Loss={valid_loss:.4f} | E2E acc={valid_e2e_accuracy:.4f}")

            if valid_e2e_accuracy > BEST_VALID_ACCURACY:
                BEST_VALID_ACCURACY = valid_e2e_accuracy
                save_ckpt(CKPT_DIR, f"best_acc_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
                logging.info(f"New best accuracy: {valid_e2e_accuracy:.4f}")

            improved = (BEST_VALID_LOSS - valid_loss) > (EARLY_STOP_MIN_DELTA * max(BEST_VALID_LOSS, 1e-12))
            if improved or not math.isfinite(BEST_VALID_LOSS):
                BEST_VALID_LOSS = valid_loss
                epochs_no_improve = 0
                save_ckpt(CKPT_DIR, f"best_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
            else:
                epochs_no_improve += 1

        encoder_lr = encoder_optimizer.param_groups[0]['lr']
        b_lr = b_optimizer.param_groups[0]['lr']
        logging.info(f'[EPOCH {global_step}] Encoder LR={encoder_lr:.6f} | B LR={b_lr:.7f}')

        global_step += 1

    save_ckpt(CKPT_DIR, f"final_{SLURM_JOB_ID}", model, encoder_optimizer, current_global, global_step)
    logging.info(f"[PPO] Training complete. Final model saved.")

    return (
        avg_epoch_loss, last_loss, valid_loss,
        last_first_stage_predictions, last_z_predictions, last_z_full,
        last_L_predictions, last_true_z, last_true_z_full, last_true_L
    )


# ============================================================================
# Entry point
# ============================================================================

def main(argv):
    try:
        log_available_resources()
        config = get_config()

        seed = int(os.getenv('SEED', '42'))
        set_reproducibility_seed(seed)
        git_info = get_git_info()

        logging.info("=" * 80)
        logging.info("PPO TRAINING SCRIPT — two_stage_gps_small")
        logging.info("=" * 80)

        ppo_config = PPOConfig(
            num_candidates=4,
            temperature=1.8,
            sampling_dropout=0.25,
            b_log_std=-1.0,
            accuracy_weight=1.0,
            infeasibility_weight=0.5,
            clip_epsilon=0.4,    # [1-ε, 1+ε] = [0.6, 1.4]
            kl_coef=0.01,
            n_inner_epochs=4,
            target_kl=0.02,
            entropy_coef=0.01,  # Set between 0.005 and 0.05 depending on need for exploration
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_full_config(config, ppo_config, git_info, seed, device)

        (avg_loss, last_loss, valid_loss, last_first_stage_predictions,
         last_z_predictions, last_z_full, last_L_predictions,
         last_true_z, last_true_z_full, last_true_L) = train_and_evaluate_ppo(
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
            ppo_config=ppo_config,
            routing_tau_start=config.routing_tau_start,
            routing_tau_end=config.routing_tau_end,
        )

        print("========== FINAL PPO TRAINING RESULTS ==========")
        print(f"Final Average Train Loss: {avg_loss}")
        print(f"Final Loss (last batch): {last_loss}")
        print(f"Final Validation Loss: {valid_loss}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    app.run(main)
