# ============================================================================
# GRPO (Group Relative Policy Optimization) Utilities for ODT
# ============================================================================
# Implements three key GRPO techniques:
# 1. Group-Based Comparison - Learn relative quality instead of absolute targets
# 2. Baseline Normalization - Normalize within dataset groups
# 3. Multiple Candidate Sampling - Generate G predictions per input
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from absl import logging


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Number of candidates to generate per input
    num_candidates: int = 4
    
    # Temperature for sampling diversity (higher = more diverse)
    temperature: float = 1.2
    
    # Dropout rate for candidate diversity during sampling
    sampling_dropout: float = 0.15
    
    # Reward function weights
    accuracy_weight: float = 1.0      # w1: Classification quality (a, b, c, d)
    infeasibility_weight: float = 0.5 # w2: Constraint violation penalty
    
    # Normalization epsilon (for numerical stability)
    norm_eps: float = 1e-8
    
    # KL divergence penalty coefficient (beta in GRPO)
    kl_coef: float = 0.01
    
    # PPO-style clipping options
    use_clipping: bool = False       # Whether to use PPO-style clipping on importance ratio
    clip_epsilon: float = 0.2        # Clipping range: ratio clipped to [1-ε, 1+ε]
    
    # Legacy: clip_range (kept for compatibility)
    clip_range: float = 0.2
    
    # Whether to use advantage weighting in loss
    use_advantage_weighting: bool = True
    
    # Loss method: 'log_prob', 'value_ratio', 'e2e_accuracy', or 'supervised_only'
    # - 'log_prob': Original GRPO with log-probability ratios
    # - 'value_ratio': Element-wise value ratios (advisor's approach)
    # - 'e2e_accuracy': End-to-end accuracy-based policy gradient
    # - 'supervised_only': Pure supervised learning (no GRPO, just cross-entropy)
    loss_method: str = 'supervised_only'  # Changed default to supervised_only for first-stage learning
    
    # Weight for E2E accuracy vs variable accuracy (only used when loss_method='e2e_accuracy')
    # 0.0 = only variable accuracy, 1.0 = only E2E accuracy
    e2e_weight: float = 0.5  # Changed from 1.0 to balance variable and E2E accuracy
    
    # Minimum group size for stable statistics
    min_group_size: int = 2

    use_value_ratios: bool = True

class ODTRewardFunction:
    """
    Reward function for Optimal Decision Tree predictions.
    
    Computes: r(pred) = w1 * accuracy - w2 * infeasibility
    
    Accuracy is measured as the match between predicted values of 
    variables a, b, c, d and their ground truth values from CPLEX solutions.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
    
    def compute_accuracy_reward(
        self,
        pred_a: torch.Tensor,   # [P, Td] - split feature selection (logits)
        pred_b: torch.Tensor,   # [Td] - split thresholds
        pred_c: torch.Tensor,   # [K, Tl] - leaf class assignment (logits)
        pred_d: torch.Tensor,   # [Td] - node activation (logits)
        true_a: torch.Tensor,   # [P, Td] - ground truth feature selection (one-hot)
        true_b: torch.Tensor,   # [Td] - ground truth thresholds
        true_c: torch.Tensor,   # [K, Tl] - ground truth class assignment (one-hot)
        true_d: torch.Tensor,   # [Td] - ground truth node activation (binary)
        b_tolerance: float = 0.2,  # Tolerance for continuous b values (relaxed from 0.1)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute accuracy as match between predictions and ground truth labels.
        
        Variables:
        - a: Categorical accuracy (argmax match for feature selection)
        - b: Tolerance-based accuracy (|pred - true| < tolerance)
        - c: Categorical accuracy (argmax match for class assignment)
        - d: Binary accuracy (threshold at 0.5)
        
        Returns:
            accuracy: Weighted average accuracy in [0, 1]
            details: Per-variable accuracy breakdown
        """
        device = pred_d.device
        accuracies = {}
        weights = {}  # Weight by number of elements
        
        # === Variable a: Feature selection (categorical) ===
        # For each decision node, which feature is selected
        pred_a_idx = pred_a.argmax(dim=0)  # [Td] - which feature per node
        true_a_idx = true_a.argmax(dim=0)  # [Td]
        a_correct = (pred_a_idx == true_a_idx).sum().item()
        a_total = pred_a_idx.numel()
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
        pred_c_idx = pred_c.argmax(dim=0)  # [Tl] - which class per leaf
        true_c_idx = true_c.argmax(dim=0)  # [Tl]
        c_correct = (pred_c_idx == true_c_idx).sum().item()
        c_total = pred_c_idx.numel()
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
        
        # === Compute weighted average accuracy ===
        # Weight by number of elements (larger variables contribute more)
        total_weight = sum(weights.values())
        weighted_accuracy = sum(
            accuracies[k] * weights[k] for k in accuracies
        ) / max(total_weight, 1)
        
        return torch.tensor(weighted_accuracy, device=device), accuracies
    
    def compute_infeasibility_penalty(
        self,
        pred_a: torch.Tensor,  # [P, Td]
        pred_d: torch.Tensor,  # [Td]
    ) -> torch.Tensor:
        """
        Compute constraint violation penalty.
        
        Checks key ODT constraint:
        - Constraint 2g: sum(a_j) = d (feature selection sums to activation)
        
        Returns:
            Normalized infeasibility in [0, 1]
        """
        violations = []
        
        # Constraint 2g: sum_j(a_jt) = d_t for each decision node t
        a_probs = F.softmax(pred_a, dim=0)  # [P, Td]
        d_probs = torch.sigmoid(pred_d)     # [Td]
        
        # Each column of a should sum to d_t (or 1 if d_t = 1)
        a_sum = a_probs.sum(dim=0)  # [Td]
        
        # Violation: |sum(a) - d|
        constraint_2g_violation = (a_sum - d_probs).abs().mean()
        violations.append(constraint_2g_violation)
        
        # Average all violations
        total_violation = torch.stack(violations).mean()
        
        return total_violation.clamp(0, 1)
    
    def compute_reward(
        self,
        pred_a: torch.Tensor,
        pred_b: torch.Tensor,
        pred_c: torch.Tensor,
        pred_d: torch.Tensor,
        true_a: torch.Tensor,
        true_b: torch.Tensor,
        true_c: torch.Tensor,
        true_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total reward for a prediction.
        
        r(pred) = w1 * accuracy - w2 * infeasibility
        
        Args:
            pred_a: [P, Td] predicted feature selection logits
            pred_b: [Td] predicted split thresholds
            pred_c: [K, Tl] predicted class assignment logits
            pred_d: [Td] predicted node activation logits
            true_a: [P, Td] ground truth feature selection (one-hot)
            true_b: [Td] ground truth thresholds
            true_c: [K, Tl] ground truth class assignment (one-hot)
            true_d: [Td] ground truth node activation (binary)
        """
        accuracy, accuracy_details = self.compute_accuracy_reward(
            pred_a, pred_b, pred_c, pred_d,
            true_a, true_b, true_c, true_d
        )
        infeasibility = self.compute_infeasibility_penalty(pred_a, pred_d)
        
        reward = (
            self.config.accuracy_weight * accuracy
            - self.config.infeasibility_weight * infeasibility
        )
        
        return reward, {
            'accuracy': accuracy.item(),
            'infeasibility': infeasibility.item(),
            'total_reward': reward.item(),
            **accuracy_details,  # Include per-variable breakdown
        }


class GRPOTrainer:
    """
    GRPO Training utilities implementing three key techniques:
    1. Group-Based Comparison
    2. Baseline Normalization
    3. Multiple Candidate Sampling
    """
    
    def __init__(self, config: GRPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.reward_fn = ODTRewardFunction(config)
    
    # =========================================================================
    # Technique 1: Group-Based Comparison
    # =========================================================================
    
    def normalize_rewards_within_group(
        self,
        rewards: torch.Tensor,  # [G] rewards for G candidates
    ) -> torch.Tensor:
        """
        Normalize rewards within a group of candidates.
        
        r_norm = (r - mean(r)) / std(r)
        
        This converts absolute rewards to relative advantages,
        where positive = better than average, negative = worse.
        """
        if rewards.numel() < self.config.min_group_size:
            # Not enough samples for stable statistics
            return torch.zeros_like(rewards)
        
        mean_r = rewards.mean()
        std_r = rewards.std()
        
        # Normalize with epsilon for stability
        normalized = (rewards - mean_r) / (std_r + self.config.norm_eps)
        
        return normalized
    
    def compute_group_advantages(
        self,
        candidate_rewards: List[torch.Tensor],  # List of [G] rewards per input
    ) -> List[torch.Tensor]:
        """
        Compute advantages for all candidate groups.
        
        For each input, normalizes rewards within its candidate group.
        """
        advantages = []
        for rewards in candidate_rewards:
            adv = self.normalize_rewards_within_group(rewards)
            advantages.append(adv)
        return advantages
    
    # =========================================================================
    # Technique 2: Baseline Normalization Within Dataset Groups
    # =========================================================================
    
    def normalize_by_dataset_group(
        self,
        losses: torch.Tensor,           # [B] losses per sample
        dataset_ids: torch.Tensor,      # [B] dataset identifier per sample
    ) -> torch.Tensor:
        """
        Normalize losses within each dataset group.
        
        This handles the case where different datasets have different
        difficulty levels (e.g., small_toy is easy, pendigits is hard).
        
        loss_norm = (loss - mean_group) / std_group
        """
        unique_datasets = dataset_ids.unique()
        normalized_losses = torch.zeros_like(losses)
        
        for dataset_id in unique_datasets:
            mask = dataset_ids == dataset_id
            group_losses = losses[mask]
            
            if group_losses.numel() >= self.config.min_group_size:
                mean_loss = group_losses.mean()
                std_loss = group_losses.std()
                normalized_losses[mask] = (group_losses - mean_loss) / (std_loss + self.config.norm_eps)
            else:
                # Not enough samples, use global normalization
                normalized_losses[mask] = group_losses - losses.mean()
        
        return normalized_losses
    
    # =========================================================================
    # Technique 3: Multiple Candidate Sampling
    # =========================================================================
    
    def sample_multiple_candidates(
        self,
        model: nn.Module,
        inputs: Tuple,
        num_candidates: int = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate G diverse candidate predictions for a single input.
        
        Uses:
        - Dropout for diversity (even during inference)
        - Temperature scaling for exploration
        - Multiple forward passes
        
        Args:
            model: The GPS model
            inputs: Tuple of model inputs
            num_candidates: Number of candidates (default: config.num_candidates)
            
        Returns:
            List of G prediction dictionaries
        """
        if num_candidates is None:
            num_candidates = self.config.num_candidates
        
        candidates = []
        
        # Store original training mode
        was_training = model.training
        
        # Enable dropout for diversity (even if was in eval mode)
        model.train()
        
        # Temporarily increase dropout if needed
        original_dropout = self._get_model_dropout(model)
        self._set_model_dropout(model, self.config.sampling_dropout)
        
        try:
            for _ in range(num_candidates):
                with torch.no_grad():
                    # Forward pass with dropout for diversity
                    fs_logits, z_logits, L_logits, all_L = model(*inputs)
                    
                    # Apply temperature scaling
                    pred_a, pred_b, pred_c, pred_d = fs_logits
                    
                    # Temperature scale the logits for diversity
                    pred_a_scaled = [a / self.config.temperature for a in pred_a]
                    pred_c_scaled = [c / self.config.temperature for c in pred_c]
                    z_logits_scaled = z_logits / self.config.temperature
                    
                    candidates.append({
                        'pred_a': pred_a_scaled,
                        'pred_b': pred_b,
                        'pred_c': pred_c_scaled,
                        'pred_d': pred_d,
                        'pred_z': z_logits_scaled,
                        'pred_L': all_L,
                        # Also keep unscaled for loss computation
                        'pred_a_raw': pred_a,
                        'pred_c_raw': pred_c,
                        'pred_z_raw': z_logits,
                    })
        finally:
            # Restore original state
            self._set_model_dropout(model, original_dropout)
            if not was_training:
                model.eval()
        
        return candidates
    
    def _get_model_dropout(self, model: nn.Module) -> float:
        """Get current dropout rate from model."""
        target = model.module if hasattr(model, 'module') else model
        return getattr(target, 'dropout', 0.0)
    
    def _set_model_dropout(self, model: nn.Module, rate: float):
        """Set dropout rate in model."""
        target = model.module if hasattr(model, 'module') else model
        if hasattr(target, 'dropout'):
            target.dropout = rate
        
        # Also update dropout layers in GPS layers
        for module in target.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
    
    # =========================================================================
    # Combined GRPO Training Step
    # =========================================================================
    
    def compute_grpo_loss(
        self,
        model: nn.Module,
        inputs: Tuple,
        true_labels: Dict[str, torch.Tensor],
        supervised_loss_fn: callable,
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute GRPO-style loss combining all three techniques.
        
        Steps:
        1. Generate G candidate predictions (Multiple Candidate Sampling)
        2. Compute rewards for each candidate
        3. Normalize rewards within group (Group-Based Comparison)
        4. Optionally normalize by dataset (Baseline Normalization)
        5. Weight supervised loss by advantage
        
        Args:
            model: The GPS model
            inputs: Model inputs tuple
            true_labels: Dict with ground truth labels:
                - true_a: [P, Td] feature selection (one-hot)
                - true_b: [Td] split thresholds
                - true_c: [K, Tl] class assignment (one-hot)
                - true_d: [Td] node activation (binary)
            supervised_loss_fn: Function to compute supervised loss given predictions
            dataset_ids: Optional dataset identifiers for baseline normalization
            
        Returns:
            total_loss: Combined GRPO loss
            metrics: Dictionary of metrics for logging
        """
        # Step 1: Generate multiple candidates
        candidates = self.sample_multiple_candidates(model, inputs)
        
        num_candidates = len(candidates)
        metrics = {
            'num_candidates': num_candidates,
            'rewards': [],
            'advantages': [],
        }
        
        # Step 2: Compute rewards for each candidate
        rewards = []
        reward_details = []
        
        for cand in candidates:
            # Use first graph's predictions for reward computation
            # (assuming batch size 1 for simplicity, can be extended)
            reward, details = self.reward_fn.compute_reward(
                pred_a=cand['pred_a'][0] if isinstance(cand['pred_a'], list) else cand['pred_a'],
                pred_b=cand['pred_b'],
                pred_c=cand['pred_c'][0] if isinstance(cand['pred_c'], list) else cand['pred_c'],
                pred_d=cand['pred_d'],
                true_a=true_labels['true_a'],
                true_b=true_labels['true_b'],
                true_c=true_labels['true_c'],
                true_d=true_labels['true_d'],
            )
            rewards.append(reward)
            reward_details.append(details)
        
        rewards_tensor = torch.stack(rewards)
        metrics['rewards'] = rewards_tensor.tolist()
        metrics['reward_details'] = reward_details
        
        # Step 3: Normalize rewards within group (Group-Based Comparison)
        advantages = self.normalize_rewards_within_group(rewards_tensor)
        metrics['advantages'] = advantages.tolist()
        
        # Step 4: Compute weighted loss
        # For GRPO: weight loss by exp(advantage) or use advantage directly
        if self.config.use_advantage_weighting:
            # Positive advantage = good candidate = increase probability
            # Negative advantage = bad candidate = decrease probability
            weights = F.softmax(advantages, dim=0)  # Convert to probability weights
        else:
            weights = torch.ones_like(advantages) / num_candidates
        
        # Step 5: Compute supervised loss for each candidate and weight
        total_loss = torch.tensor(0.0, device=self.device)
        
        # For training, we need gradients, so do one more forward pass
        # with the best candidate's influence
        model.train()
        fs_logits, z_logits, L_logits, all_L = model(*inputs)
        
        # Compute standard supervised loss
        sup_loss = supervised_loss_fn(fs_logits, z_logits, L_logits, all_L)
        
        # Weight by best advantage (encourage model toward best candidates)
        best_idx = advantages.argmax()
        best_advantage = advantages[best_idx]
        
        # GRPO-style: scale loss by (1 + advantage) to encourage better solutions
        # If best candidate is much better than average, reduce loss weight
        # If best candidate is only slightly better, maintain loss weight
        advantage_scale = torch.clamp(1.0 + 0.5 * best_advantage, 0.5, 2.0)
        
        total_loss = sup_loss * advantage_scale
        
        metrics['supervised_loss'] = sup_loss.item()
        metrics['advantage_scale'] = advantage_scale.item()
        metrics['best_candidate_idx'] = best_idx.item()
        metrics['best_reward'] = rewards[best_idx].item()
        
        # Optional: Add KL divergence penalty (for stability)
        # This prevents the policy from diverging too far from reference
        # kl_loss = self.config.kl_coef * kl_divergence
        # total_loss = total_loss + kl_loss
        
        return total_loss, metrics
    
    def select_best_candidate(
        self,
        candidates: List[Dict[str, torch.Tensor]],
        true_labels: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Select the best candidate based on reward.
        
        Used during inference to return the highest-quality prediction.
        
        Args:
            candidates: List of candidate prediction dictionaries
            true_labels: Dict with true_a, true_b, true_c, true_d
        """
        best_reward = float('-inf')
        best_candidate = None
        
        for cand in candidates:
            reward, _ = self.reward_fn.compute_reward(
                pred_a=cand['pred_a'][0] if isinstance(cand['pred_a'], list) else cand['pred_a'],
                pred_b=cand['pred_b'],
                pred_c=cand['pred_c'][0] if isinstance(cand['pred_c'], list) else cand['pred_c'],
                pred_d=cand['pred_d'],
                true_a=true_labels['true_a'],
                true_b=true_labels['true_b'],
                true_c=true_labels['true_c'],
                true_d=true_labels['true_d'],
            )
            
            if reward > best_reward:
                best_reward = reward
                best_candidate = cand
        
        return best_candidate, best_reward


# ============================================================================
# Helper functions for integration with existing training code
# ============================================================================

def create_grpo_trainer(
    num_candidates: int = 4,
    temperature: float = 1.2,
    sampling_dropout: float = 0.15,
    accuracy_weight: float = 1.0,
    infeasibility_weight: float = 0.5,
    device: torch.device = None,
) -> GRPOTrainer:
    """
    Factory function to create a GRPO trainer with custom configuration.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = GRPOConfig(
        num_candidates=num_candidates,
        temperature=temperature,
        sampling_dropout=sampling_dropout,
        accuracy_weight=accuracy_weight,
        infeasibility_weight=infeasibility_weight,
    )
    
    return GRPOTrainer(config, device)


def grpo_normalize_batch_losses(
    losses: Dict[str, torch.Tensor],
    dataset_names: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Apply dataset-group normalization to a batch of losses.
    
    Args:
        losses: Dict with loss tensors (loss_a, loss_b, etc.)
        dataset_names: List of dataset names for each sample in batch
        device: Torch device
        
    Returns:
        Normalized losses dictionary
    """
    # Create dataset ID mapping
    unique_datasets = list(set(dataset_names))
    dataset_to_id = {name: i for i, name in enumerate(unique_datasets)}
    dataset_ids = torch.tensor([dataset_to_id[name] for name in dataset_names], device=device)
    
    # Create temporary trainer for normalization
    config = GRPOConfig()
    trainer = GRPOTrainer(config, device)
    
    normalized_losses = {}
    for key, loss_tensor in losses.items():
        if loss_tensor.dim() > 0 and loss_tensor.numel() > 1:
            normalized_losses[key] = trainer.normalize_by_dataset_group(loss_tensor, dataset_ids)
        else:
            normalized_losses[key] = loss_tensor
    
    return normalized_losses
