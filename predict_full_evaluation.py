"""
Full Cross-Evaluation Script for ODT Predictions

This script:
1. Loads a trained model checkpoint
2. Iterates over ALL validation datasets (100 files)
3. For EACH validation dataset:
   - Predicts all variables (A, B, C, D, Z, L)
   - Builds the optimal decision tree
   - Evaluates on ALL test datasets (100 files)
4. Tracks performance of each tree across all test sets
5. Identifies the BEST-performing tree structure
6. Saves comprehensive results with tree structure statistics

Usage:
    srun --partition=IMSE --nodelist=GPU51 --gres=gpu:1 --cpus-per-task=32 --mem=64G --pty bash

    conda activate neural_diving_pytorch

    export HOME_REPO="$HOME/two_stage_neural_diving"

    python predict_full_evaluation.py \
    --ckpt_path saved_models/latest_303677.pt \
    --dataset seeds \
    --output_dir validation_predictions

Author: Jose Navarro
"""

import os
import sys
import argparse 
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Add parent directory to path for importing from two_stage_neural_diving
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Custom imports
import two_stage_data_utils3
from config_train_pytorch2 import get_config
from read_solution_final2 import process_file_to_numpy

# Model imports are done dynamically based on --model_type argument
# Default to skipped5, can be overridden with --model_type small


def normalize_features_with_params(features: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    """Normalize features using pre-computed min/max parameters."""
    range_data = data_max - data_min
    range_data[range_data == 0] = 1
    return (features - data_min) / range_data


def load_model(ckpt_path: str, device: torch.device, model_config, model_type: str = 'skipped5') -> nn.Module:
    """Load model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Torch device
        model_config: Model configuration
        model_type: Model architecture type ('skipped5' or 'small')
    """
    # Import the appropriate model module based on model_type
    if model_type == 'small':
        import two_stage_gps_small as model_module
        print(f"[INFO] Using GPS small architecture (explicit tree routing)")
    else:
        import two_stage_gps_skipped5 as model_module
        print(f"[INFO] Using GPS skipped5 architecture (with second-stage decoder)")

    # Load checkpoint first to auto-detect configuration
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)

    # Auto-detect number of layers from checkpoint
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith('first_layers.'):
            parts = key.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    layer_indices.add(layer_idx)
                except ValueError:
                    pass

    config = model_config.model_config

    # Override n_layers if detected from checkpoint
    if layer_indices:
        detected_n_layers = max(layer_indices) + 1
        print(f"[INFO] Auto-detected n_layers={detected_n_layers} from checkpoint")
        config.params['n_layers'] = detected_n_layers

    model = model_module.get_model(**config.params).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load state dict with strict=False to handle minor mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[INFO] Missing keys: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
    if unexpected_keys:
        print(f"[INFO] Unexpected keys: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")

    model.eval()

    print(f"[INFO] Loaded model from: {ckpt_path}")
    if 'global_step' in checkpoint:
        print(f"[INFO] Checkpoint step: {checkpoint['global_step']}")

    return model


def predict_on_dataset(model: nn.Module, dataset_path: str, device: torch.device,
                       num_workers: int = 0, model_type: str = 'skipped5') -> Optional[Dict]:
    """Run model prediction on a single dataset.

    Args:
        model: The loaded model
        dataset_path: Path to dataset file
        device: Torch device
        num_workers: Number of data loader workers
        model_type: Model architecture type ('skipped5' or 'small')
    """
    try:
        data_loader = two_stage_data_utils3.get_dataset(
            dataset_paths=[dataset_path],
            outputs_paths=None,
            scale_features=False,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
        )

        predictions = None

        with torch.no_grad():
            for batch in data_loader:
                if batch is None:
                    return None

                states = batch[0]
                first_stage_graph = batch[1].to(device)
                second_stage_states = batch[4]

                # === DIAGNOSTIC: Log input graph statistics per validation file ===
                diag_info = {
                    'file': os.path.basename(dataset_path),
                    'graph_x_shape': list(first_stage_graph.x.shape),
                    'graph_x_mean': float(first_stage_graph.x.mean()),
                    'graph_x_std': float(first_stage_graph.x.std()),
                    'graph_x_min': float(first_stage_graph.x.min()),
                    'graph_x_max': float(first_stage_graph.x.max()),
                }
                print(f"[DIAG] File: {os.path.basename(dataset_path)}")
                print(f"  graph.x shape: {first_stage_graph.x.shape}")
                print(f"  graph.x stats: mean={first_stage_graph.x.mean():.6f}, "
                      f"std={first_stage_graph.x.std():.6f}, "
                      f"min={first_stage_graph.x.min():.6f}, "
                      f"max={first_stage_graph.x.max():.6f}")

                first_stage_variable_indices = [s['first_stage_variable_indices'].to(device) for s in states]
                variable_shapes = [s['variable_shapes'] for s in states]
                first_stage_constraint_shapes = [s['constraint_shapes'].tolist() for s in states]
                second_stage_constraint_shapes = [ss['constraint_shapes'].tolist() for ss in second_stage_states]
                second_stage_variable_features = [ss['variable_features'].to(device) for ss in second_stage_states]
                second_stage_constraint_features = [ss['constraint_features'].to(device) for ss in second_stage_states]
                second_stage_edge_indices = [ss['edge_indices'].to(device) for ss in second_stage_states]

                normalized_X = states[0]['normalized_X'].cpu().numpy()
                true_labels = states[0]['sample_true_labels'].cpu().numpy()

                # Build inputs tuple based on model type
                if model_type == 'small':
                    # GPS small model uses first_stage_states instead of second_stage features
                    inputs = (
                        first_stage_graph,
                        False,  # is_training=False
                        first_stage_variable_indices,
                        variable_shapes,
                        states,  # first_stage_states for explicit tree routing
                        device
                    )
                else:
                    # Original skipped5 model
                    inputs = (
                        first_stage_graph,
                        False,
                        first_stage_variable_indices,
                        variable_shapes,
                        second_stage_constraint_features,
                        second_stage_variable_features,
                        second_stage_edge_indices,
                        first_stage_constraint_shapes,
                        second_stage_constraint_shapes,
                        device
                    )

                fs_logits, z_logits_reshaped, _, all_L_predictions = model(*inputs)
                pred_a, pred_b, pred_c, pred_d = fs_logits
                
                sample_var_shapes = variable_shapes[0]
                a_shape = sample_var_shapes[0]
                c_shape = sample_var_shapes[2]
                
                pred_a_reshaped = pred_a[0].reshape(a_shape[0], a_shape[1])
                pred_c_reshaped = pred_c[0].reshape(c_shape[0], c_shape[1])
                
                pred_a_softmax = F.softmax(pred_a_reshaped, dim=0)
                pred_a_onehot = torch.zeros_like(pred_a_softmax)
                for col in range(pred_a_softmax.shape[1]):
                    max_idx = pred_a_softmax[:, col].argmax()
                    pred_a_onehot[max_idx, col] = 1.0
                
                pred_c_softmax = F.softmax(pred_c_reshaped, dim=0)
                pred_c_onehot = torch.zeros_like(pred_c_softmax)
                for col in range(pred_c_softmax.shape[1]):
                    max_idx = pred_c_softmax[:, col].argmax()
                    pred_c_onehot[max_idx, col] = 1.0
                
                pred_b_values = torch.sigmoid(pred_b).squeeze()
                pred_d_probs = torch.sigmoid(pred_d).squeeze()
                pred_d_binary = (pred_d_probs > 0.5).float()
                
                z_probs = torch.softmax(z_logits_reshaped, dim=-1)
                z_predictions = z_probs.argmax(dim=-1).cpu().numpy()
                L_predictions = all_L_predictions.detach().cpu().numpy()
                
                # Add prediction head diagnostics
                # Note: pred_a and pred_c are lists of tensors (one per graph), use [0] for single batch
                print(f"[PRED-HEADS] pred_a_logits: mean={pred_a[0].mean():.4f}, std={pred_a[0].std():.4f}")
                print(f"[PRED-HEADS] pred_b: mean={pred_b.mean():.4f}, std={pred_b.std():.4f}, "
                      f"sigmoid_values={pred_b_values.cpu().numpy()}")
                print(f"[PRED-HEADS] pred_c_logits: mean={pred_c[0].mean():.4f}, std={pred_c[0].std():.4f}")
                print(f"[PRED-HEADS] pred_d: mean={pred_d.mean():.4f}, std={pred_d.std():.4f}")

                diag_info['pred_a_mean'] = float(pred_a[0].mean())
                diag_info['pred_a_std'] = float(pred_a[0].std())
                diag_info['pred_b_mean'] = float(pred_b.mean())
                diag_info['pred_b_std'] = float(pred_b.std())
                diag_info['pred_b_values'] = pred_b_values.cpu().numpy().tolist()
                diag_info['pred_c_mean'] = float(pred_c[0].mean())
                diag_info['pred_c_std'] = float(pred_c[0].std())

                predictions = {
                    'pred_a': pred_a_onehot.cpu().numpy(),
                    'pred_a_probs': pred_a_softmax.cpu().numpy(),
                    'pred_b': pred_b_values.cpu().numpy(),
                    'pred_c': pred_c_onehot.cpu().numpy(),
                    'pred_c_probs': pred_c_softmax.cpu().numpy(),
                    'pred_d': pred_d_binary.cpu().numpy(),
                    'pred_d_probs': pred_d_probs.cpu().numpy(),
                    'pred_z': z_predictions,
                    'pred_z_probs': z_probs.cpu().numpy(),
                    'pred_L': L_predictions,
                    'normalized_X': normalized_X,
                    'true_labels': true_labels,
                    'variable_shapes': sample_var_shapes,
                    'a_shape': a_shape,
                    'c_shape': c_shape,
                    'diagnostics': diag_info,
                }
                break
        
        return predictions
    except Exception as e:
        logging.warning(f"Failed to predict on {dataset_path}: {e}")
        return None


def build_decision_tree_structure(predictions: Dict) -> Dict:
    """Build decision tree structure from first-stage variable predictions."""
    A = predictions['pred_a']
    B = predictions['pred_b']
    C = predictions['pred_c']
    D = predictions['pred_d']
    
    n_features = A.shape[0]
    n_classes = C.shape[0]
    n_decision_nodes = A.shape[1]
    n_leaf_nodes = C.shape[1]
    
    selected_features = np.argmax(A, axis=0)
    leaf_classes = np.argmax(C, axis=0) + 1
    
    return {
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_decision_nodes': n_decision_nodes,
        'n_leaf_nodes': n_leaf_nodes,
        'selected_features': selected_features,
        'thresholds': B,
        'leaf_classes': leaf_classes,
    }


def get_tree_signature(tree: Dict) -> str:
    """Create a unique signature for tree structure comparison."""
    feats = tuple(tree['selected_features'])
    thresholds = tuple(np.round(tree['thresholds'], 4))
    classes = tuple(tree['leaf_classes'])
    return f"F{feats}_T{thresholds}_C{classes}"


def compute_z_metrics(z_predictions: np.ndarray, normalized_X: np.ndarray) -> Dict:
    """
    Compute metrics for Z (leaf routing assignments).
    
    Args:
        z_predictions: [N] leaf indices for each sample
        normalized_X: [N, P] sample features
    
    Returns:
        Dict with Z metrics (distribution, entropy, etc.)
    """
    N = len(z_predictions)
    unique_leaves, counts = np.unique(z_predictions, return_counts=True)
    
    # Leaf distribution
    leaf_dist = {int(leaf): int(count) for leaf, count in zip(unique_leaves, counts)}
    
    # Entropy of leaf assignment
    probs = counts / N
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Balance metric (how evenly distributed are samples across leaves)
    max_count = np.max(counts)
    min_count = np.min(counts)
    balance_ratio = min_count / (max_count + 1e-10)
    
    return {
        'n_samples': N,
        'unique_leaves': int(len(unique_leaves)),
        'leaf_distribution': leaf_dist,
        'entropy': float(entropy),
        'balance_ratio': float(balance_ratio),
        'max_leaf_size': int(max_count),
        'min_leaf_size': int(min_count),
    }


def compute_l_metrics(L_predictions: np.ndarray) -> Dict:
    """
    Compute metrics for L (dual variables / constraint values).
    
    Args:
        L_predictions: [M] or [M, K] dual variable predictions
    
    Returns:
        Dict with L statistics
    """
    L_flat = L_predictions.flatten()
    
    return {
        'shape': L_predictions.shape,
        'mean': float(np.mean(L_flat)),
        'std': float(np.std(L_flat)),
        'min': float(np.min(L_flat)),
        'max': float(np.max(L_flat)),
        'median': float(np.median(L_flat)),
        'n_positive': int(np.sum(L_flat > 0)),
        'n_negative': int(np.sum(L_flat < 0)),
        'n_zero': int(np.sum(np.abs(L_flat) < 1e-6)),
        'l2_norm': float(np.linalg.norm(L_flat)),
        'sparsity': float(np.sum(np.abs(L_flat) < 1e-6) / len(L_flat)),
    }


def load_ground_truth_tree(valid_file: str) -> Optional[Dict]:
    """
    Load ground truth first-stage variables (A, B, C, D) from outputs folder.
    
    Maps validation file to corresponding outputs file and extracts ground truth variables.
    Handles the naming pattern: training file 'glass_4801.glass' maps to 
    outputs file 'info-glass_4801.glass-sd1-2-CMS-*.out' (CMS suffix may vary)
    
    Args:
        valid_file: Path to validation dataset file (e.g., .../glass/training/glass_4801.glass)
    
    Returns:
        Dict with ground truth tree structure, or None if not found
    """
    try:
        # Extract directory path and basename
        valid_dir = os.path.dirname(valid_file)
        valid_basename = os.path.basename(valid_file)
        
        # Navigate to outputs folder
        outputs_dir = valid_dir.replace('training', 'outputs')
        
        # Try to find the outputs file with flexible CMS suffix
        # Pattern: info-{basename}-sd1-2-CMS-*.out
        pattern_prefix = f"info-{valid_basename}-sd1-2-CMS-"
        outputs_file = None
        
        if os.path.exists(outputs_dir):
            for filename in os.listdir(outputs_dir):
                if filename.startswith(pattern_prefix) and filename.endswith('.out'):
                    outputs_file = os.path.join(outputs_dir, filename)
                    break
        
        if outputs_file is None or not os.path.exists(outputs_file):
            logging.warning(f"Ground truth outputs file not found for: {valid_basename}")
            return None
        
        # Load ground truth solution
        from read_solution_final2 import process_solution_file, extract_matrices_and_vectors
        
        solution_data = process_solution_file(outputs_file)
        warm_start, final_solution = extract_matrices_and_vectors(outputs_file)
        
        if not warm_start and not final_solution:
            logging.warning(f"Could not extract ground truth variables from {outputs_file}")
            return None
        
        # Prefer final_solution (optimal) when available; fall back to warm_start
        gt_vars = final_solution if final_solution else warm_start
        
        # Extract ground truth first-stage variables
        # extract_matrices_and_vectors returns dicts with keys: 'variable_a', 'variable_b', 'variable_c', 'variable_d'
        A_gt = gt_vars.get('variable_a')  # Feature selection matrix
        B_gt = gt_vars.get('variable_b')  # Thresholds
        C_gt = gt_vars.get('variable_c')  # Class assignment matrix
        D_gt = gt_vars.get('variable_d')  # Leaf indicators
        
        if A_gt is None or B_gt is None or C_gt is None:
            logging.warning(f"Missing ground truth variables in {outputs_file}")
            return None
        
        # Build tree structure from ground truth
        n_features = A_gt.shape[0]
        n_classes = C_gt.shape[0]
        n_decision_nodes = A_gt.shape[1]
        n_leaf_nodes = C_gt.shape[1]
        
        selected_features = np.argmax(A_gt, axis=0)
        leaf_classes = np.argmax(C_gt, axis=0) + 1
        
        return {
            'A': A_gt,
            'B': B_gt,
            'C': C_gt,
            'D': D_gt if D_gt is not None else np.zeros(n_decision_nodes),
            'n_features': n_features,
            'n_classes': n_classes,
            'n_decision_nodes': n_decision_nodes,
            'n_leaf_nodes': n_leaf_nodes,
            'selected_features': selected_features,
            'thresholds': B_gt,
            'leaf_classes': leaf_classes,
            'is_ground_truth': True,
        }
    except Exception as e:
        logging.warning(f"Error loading ground truth tree for {valid_file}: {e}")
        return None


def compute_ground_truth_L(normalized_X: np.ndarray, true_labels: np.ndarray, ground_truth_tree: Dict) -> float:
    """
    Compute ground truth L using the optimal tree structure.
    
    This uses ground truth first-stage variables (A, B, C, D) to build the optimal tree
    and counts actual misclassification errors.
    
    Args:
        normalized_X: [N, P] normalized feature matrix
        true_labels: [N] true class labels
        ground_truth_tree: Tree structure dict built from ground truth variables
    
    Returns:
        float: Sum of misclassification errors using optimal tree (ground truth L)
    """
    if ground_truth_tree is None:
        return np.nan
    
    predictions = classify_samples(normalized_X, ground_truth_tree)
    misclassifications = np.sum(predictions != true_labels)
    return float(misclassifications)


def compute_empirical_L(predictions: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    Compute empirical L by counting misclassifications.
    
    For each observation, if predicted class != true class, loss = 1, else 0.
    Empirical L = sum of all individual losses = number of misclassifications.
    
    Args:
        predictions: [N] predicted class labels for each observation
        true_labels: [N] ground truth class labels
    
    Returns:
        Dict with empirical L metrics
    """
    n_samples = len(true_labels)
    
    # Compute per-observation loss (1 if mismatch, 0 if match)
    per_observation_loss = (predictions != true_labels).astype(int)
    
    # Empirical L is the sum of all misclassifications
    empirical_L = np.sum(per_observation_loss)
    
    # Additional breakdown
    n_correct = n_samples - empirical_L
    accuracy = n_correct / n_samples if n_samples > 0 else 0.0
    
    return {
        'empirical_L': int(empirical_L),
        'n_samples': n_samples,
        'n_correct': int(n_correct),
        'n_misclassified': int(empirical_L),
        'accuracy': float(accuracy),
        'per_observation_loss': per_observation_loss,  # [N] array of 0s and 1s
    }


def compute_L_prediction_error(predicted_L: float, ground_truth_L: float, empirical_L: float = None) -> Dict:
    """
    Compute error metrics for L prediction vs ground truth and empirical L.
    
    Args:
        predicted_L: Model's predicted L value (scalar)
        ground_truth_L: Actual computed L value from optimal tree
        empirical_L: Empirical L computed from predicted tree misclassifications
    
    Returns:
        Dict with prediction error metrics
    """
    absolute_error = abs(predicted_L - ground_truth_L)
    relative_error = absolute_error / (ground_truth_L + 1e-8)
    
    result = {
        'predicted_L': float(predicted_L),
        'ground_truth_L': float(ground_truth_L),
        'absolute_error': float(absolute_error),
        'relative_error': float(relative_error),
        'prediction_correct': absolute_error < 0.5,  # Within 0.5 misclassifications
    }
    
    # Add empirical L comparison if available
    if empirical_L is not None:
        result['empirical_L'] = float(empirical_L)
        result['pred_vs_empirical_error'] = abs(predicted_L - empirical_L)
        result['gt_vs_empirical_error'] = abs(ground_truth_L - empirical_L)
    
    return result


def classify_samples(X: np.ndarray, tree: Dict) -> np.ndarray:
    """Classify samples using the decision tree.
    
    Handles two C matrix formats:
    - C with 7 columns: All tree nodes (0-2 decision, 3-6 leaves). Use current_node directly.
    - C with 4 columns: Leaf nodes only. Use leaf_idx = current_node - 3.
    """
    A = tree['A']
    B = tree['B']
    C = tree['C']
    
    N = X.shape[0]
    predictions = np.zeros(N, dtype=np.int32)
    
    # Determine C matrix format based on number of columns
    # 7 columns = all nodes format (from ground truth files)
    # 4 columns = leaves only format (from GPS predictions)
    c_has_all_nodes = (C.shape[1] == 7)
    
    for i in range(N):
        x = X[i]
        current_node = 0
        
        for depth in range(2):
            if current_node >= len(B):
                break
            
            feature_idx = int(np.argmax(A[:, current_node]))
            threshold = float(B[current_node])
            
            if x[feature_idx] >= threshold:
                current_node = 2 * current_node + 2
            else:
                current_node = 2 * current_node + 1
        
        # Get class from C matrix based on its format
        if c_has_all_nodes:
            # C has 7 columns for all nodes - use current_node directly
            c_col_idx = current_node
        else:
            # C has 4 columns for leaves only - compute leaf index
            c_col_idx = current_node - 3
        
        if c_col_idx < 0 or c_col_idx >= C.shape[1]:
            c_col_idx = 0 if not c_has_all_nodes else 3
        
        predictions[i] = int(np.argmax(C[:, c_col_idx])) + 1
    
    return predictions


# =============================================================================
# Variable-Level Comparison Statistics (Predicted vs Ground Truth)
# =============================================================================

def route_samples_through_tree_for_Z(X: np.ndarray, tree: Dict) -> np.ndarray:
    """
    Route normalized samples through a tree defined by A, B to get leaf node indices.
    
    Args:
        X: [N, P] normalized features
        tree: Dict with 'A' [P, >=3] and 'B' [>=3]
    
    Returns:
        [N] array of leaf node indices (values in {3, 4, 5, 6})
    """
    A = tree['A']
    B = tree['B']
    N = X.shape[0]
    leaf_indices = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        node = 0
        for depth in range(2):
            if node >= 3:
                break
            feat = int(np.argmax(A[:, node]))
            thresh = float(B[node]) if node < len(B) else 0.5
            if X[i, feat] >= thresh:
                node = 2 * node + 2
            else:
                node = 2 * node + 1
        leaf_indices[i] = node
    
    return leaf_indices


def compute_variable_A_stats(pred_A: np.ndarray, gt_A: np.ndarray) -> Dict:
    """
    Compare predicted vs ground truth Variable A (Feature Selection).
    A is a binary one-hot matrix [P, n_nodes]. Per decision node, one feature is selected.
    """
    n_nodes = min(3, pred_A.shape[1], gt_A.shape[1])
    pred_A_dec = pred_A[:, :n_nodes]
    gt_A_dec = gt_A[:, :n_nodes]
    
    pred_feats = np.argmax(pred_A_dec, axis=0)
    gt_feats = np.argmax(gt_A_dec, axis=0)
    per_node_match = (pred_feats == gt_feats).astype(int)
    overall_accuracy = float(np.mean(per_node_match))
    
    pred_binary = (pred_A_dec > 0.5).astype(float)
    gt_binary = (gt_A_dec > 0.5).astype(float)
    elementwise_accuracy = float(np.mean(pred_binary == gt_binary))
    hamming_distance = float(np.sum(pred_binary != gt_binary))
    
    return {
        'pred_features': pred_feats.tolist(),
        'gt_features': gt_feats.tolist(),
        'per_node_match': per_node_match.tolist(),
        'feature_selection_accuracy': overall_accuracy,
        'elementwise_accuracy': elementwise_accuracy,
        'hamming_distance': hamming_distance,
        'n_nodes': n_nodes,
    }


def compute_variable_B_stats(pred_B: np.ndarray, gt_B: np.ndarray) -> Dict:
    """
    Compare predicted vs ground truth Variable B (Thresholds).
    B is continuous [n_nodes] with threshold values, typically in [0, 1].
    """
    n = min(3, len(pred_B), len(gt_B))
    pred_b = np.array(pred_B[:n], dtype=float).flatten()
    gt_b = np.array(gt_B[:n], dtype=float).flatten()
    
    per_node_error = np.abs(pred_b - gt_b)
    mae = float(np.mean(per_node_error))
    mse = float(np.mean(per_node_error ** 2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(per_node_error))
    per_node_rel_error = per_node_error / (np.abs(gt_b) + 1e-8)
    mean_rel_error = float(np.mean(per_node_rel_error))
    
    return {
        'pred_thresholds': pred_b.tolist(),
        'gt_thresholds': gt_b.tolist(),
        'per_node_abs_error': per_node_error.tolist(),
        'per_node_rel_error': per_node_rel_error.tolist(),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'max_error': max_error,
        'mean_rel_error': mean_rel_error,
        'n_nodes': n,
    }


def compute_variable_C_stats(pred_C: np.ndarray, gt_C: np.ndarray) -> Dict:
    """
    Compare predicted vs ground truth Variable C (Class Assignment).
    C is a binary one-hot matrix [K, n_leaves].
    Handles both 4-column (leaf-only) and 7-column (all-nodes) formats.
    """
    def get_leaf_columns(C_mat):
        if C_mat.shape[1] == 7:
            return C_mat[:, 3:7]
        elif C_mat.shape[1] == 4:
            return C_mat
        else:
            return C_mat[:, -min(4, C_mat.shape[1]):]
    
    pred_C_leaves = get_leaf_columns(pred_C)
    gt_C_leaves = get_leaf_columns(gt_C)
    
    n_leaves = min(pred_C_leaves.shape[1], gt_C_leaves.shape[1], 4)
    n_classes = min(pred_C_leaves.shape[0], gt_C_leaves.shape[0])
    
    pred_classes = np.argmax(pred_C_leaves[:n_classes, :n_leaves], axis=0)
    gt_classes = np.argmax(gt_C_leaves[:n_classes, :n_leaves], axis=0)
    per_leaf_match = (pred_classes == gt_classes).astype(int)
    overall_accuracy = float(np.mean(per_leaf_match))
    
    pred_binary = (pred_C_leaves[:n_classes, :n_leaves] > 0.5).astype(float)
    gt_binary = (gt_C_leaves[:n_classes, :n_leaves] > 0.5).astype(float)
    elementwise_accuracy = float(np.mean(pred_binary == gt_binary))
    
    return {
        'pred_classes': (pred_classes + 1).tolist(),
        'gt_classes': (gt_classes + 1).tolist(),
        'per_leaf_match': per_leaf_match.tolist(),
        'class_assignment_accuracy': overall_accuracy,
        'elementwise_accuracy': elementwise_accuracy,
        'n_leaves': n_leaves,
    }


def compute_variable_D_stats(pred_D: np.ndarray, gt_D: np.ndarray) -> Dict:
    """
    Compare predicted vs ground truth Variable D (Leaf Indicators).
    D is binary [n_nodes]. D[i]=0 means decision node, D[i]=1 means leaf (pruned).
    """
    n = min(3, len(pred_D), len(gt_D))
    pred_d = np.array(pred_D[:n], dtype=float).flatten()
    gt_d = np.array(gt_D[:n], dtype=float).flatten()
    
    pred_binary = (pred_d > 0.5).astype(int)
    gt_binary = (gt_d > 0.5).astype(int)
    per_node_match = (pred_binary == gt_binary).astype(int)
    overall_accuracy = float(np.mean(per_node_match))
    
    return {
        'pred_D': pred_binary.tolist(),
        'gt_D': gt_binary.tolist(),
        'per_node_match': per_node_match.tolist(),
        'accuracy': overall_accuracy,
        'n_nodes': n,
    }


def compute_variable_Z_comparison(pred_Z: np.ndarray, gt_Z: np.ndarray) -> Optional[Dict]:
    """
    Compare predicted vs ground truth leaf routing assignments (Variable Z).
    Both should use the same node indexing (e.g., 3-6 for leaves in a depth-2 tree).
    """
    if gt_Z is None:
        return None
    
    n = min(len(pred_Z), len(gt_Z))
    pred_z = np.array(pred_Z[:n])
    gt_z = np.array(gt_Z[:n])
    
    routing_accuracy = float(np.mean(pred_z == gt_z))
    n_correct = int(np.sum(pred_z == gt_z))
    
    unique_gt_leaves = np.unique(gt_z)
    per_leaf_accuracy = {}
    per_leaf_count = {}
    for leaf in unique_gt_leaves:
        mask = gt_z == leaf
        count = int(np.sum(mask))
        per_leaf_count[int(leaf)] = count
        per_leaf_accuracy[int(leaf)] = float(np.mean(pred_z[mask] == gt_z[mask]))
    
    return {
        'routing_accuracy': routing_accuracy,
        'n_samples': n,
        'n_correct': n_correct,
        'per_leaf_accuracy': per_leaf_accuracy,
        'per_leaf_count': per_leaf_count,
    }


def compute_all_variable_stats(pred_tree: Dict, gt_tree: Optional[Dict],
                               pred_Z_routing: np.ndarray,
                               gt_Z_routing: Optional[np.ndarray]) -> Optional[Dict]:
    """
    Compute comparison statistics for all first-stage variables (A, B, C, D) and Z routing.
    Returns None if no ground truth is available.
    """
    if gt_tree is None:
        return None
    
    try:
        a_stats = compute_variable_A_stats(pred_tree['A'], gt_tree['A'])
        b_stats = compute_variable_B_stats(pred_tree['B'], gt_tree['B'])
        c_stats = compute_variable_C_stats(pred_tree['C'], gt_tree['C'])
        d_stats = compute_variable_D_stats(pred_tree['D'], gt_tree['D'])
        z_stats = compute_variable_Z_comparison(pred_Z_routing, gt_Z_routing)
        
        return {
            'A': a_stats,
            'B': b_stats,
            'C': c_stats,
            'D': d_stats,
            'Z_routing': z_stats,
        }
    except Exception as e:
        logging.warning(f"Error computing variable stats: {e}")
        return None


def write_variable_comparison_section(f, all_valid_results: List, valid_files: List[str]):
    """Write the per-variable comparison section to the output file."""
    results_with_stats = [(i, r) for i, r in enumerate(all_valid_results)
                          if r is not None and r.get('variable_stats') is not None]
    
    if not results_with_stats:
        f.write("No ground truth data available for variable-level comparison.\n\n")
        return
    
    n_with_stats = len(results_with_stats)
    
    # === Variable A ===
    f.write("Variable A - Feature Selection [Binary one-hot, P x 3 decision nodes]\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'#':<4} {'Valid File':<22} {'N0':>4} {'N1':>4} {'N2':>4} {'Acc':>8} {'ElemAcc':>8} {'Pred Features':<18} {'GT Features':<18}\n")
    f.write("-" * 100 + "\n")
    
    for idx, result in results_with_stats:
        valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0][:22]
        a = result['variable_stats']['A']
        m = a['per_node_match']
        f.write(f"{idx+1:<4} {valid_name:<22} {m[0]:>4} {m[1]:>4} {m[2] if len(m) > 2 else '-':>4} "
                f"{a['feature_selection_accuracy']:>8.3f} {a['elementwise_accuracy']:>8.3f} "
                f"{str(a['pred_features']):<18} {str(a['gt_features']):<18}\n")
    
    a_all = [r['variable_stats']['A'] for _, r in results_with_stats]
    a_accs = [a['feature_selection_accuracy'] for a in a_all]
    a_elem_accs = [a['elementwise_accuracy'] for a in a_all]
    a_hammings = [a['hamming_distance'] for a in a_all]
    n_nodes = a_all[0]['n_nodes']
    
    f.write(f"\n  Aggregate A Statistics ({n_with_stats} trees with ground truth):\n")
    for node in range(n_nodes):
        node_acc = np.mean([a['per_node_match'][node] for a in a_all])
        f.write(f"    Node {node} Accuracy: {node_acc:.4f} ({node_acc*100:.1f}%)\n")
    f.write(f"    Overall Feature Selection Accuracy: mean={np.mean(a_accs):.4f}, std={np.std(a_accs):.4f}, min={np.min(a_accs):.4f}, max={np.max(a_accs):.4f}\n")
    f.write(f"    Element-wise Accuracy: mean={np.mean(a_elem_accs):.4f}, std={np.std(a_elem_accs):.4f}\n")
    f.write(f"    Hamming Distance: mean={np.mean(a_hammings):.2f}, std={np.std(a_hammings):.2f}, min={np.min(a_hammings):.1f}, max={np.max(a_hammings):.1f}\n\n")
    
    # === Variable B ===
    f.write("Variable B - Thresholds [Continuous, 3 decision nodes]\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'#':<4} {'Valid File':<22} {'Err N0':>8} {'Err N1':>8} {'Err N2':>8} {'MAE':>8} {'RMSE':>8} {'Pred B':<24} {'GT B':<24}\n")
    f.write("-" * 100 + "\n")
    
    for idx, result in results_with_stats:
        valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0][:22]
        b = result['variable_stats']['B']
        e = b['per_node_abs_error']
        pred_str = '[' + ', '.join(f'{v:.4f}' for v in b['pred_thresholds']) + ']'
        gt_str = '[' + ', '.join(f'{v:.4f}' for v in b['gt_thresholds']) + ']'
        e2 = e[2] if len(e) > 2 else 0.0
        f.write(f"{idx+1:<4} {valid_name:<22} {e[0]:>8.4f} {e[1]:>8.4f} {e2:>8.4f} {b['mae']:>8.4f} {b['rmse']:>8.4f} {pred_str:<24} {gt_str:<24}\n")
    
    b_all = [r['variable_stats']['B'] for _, r in results_with_stats]
    b_maes = [b['mae'] for b in b_all]
    b_rmses = [b['rmse'] for b in b_all]
    b_max_errs = [b['max_error'] for b in b_all]
    b_rel_errs = [b['mean_rel_error'] for b in b_all]
    n_nodes_b = b_all[0]['n_nodes']
    
    f.write(f"\n  Aggregate B Statistics ({n_with_stats} trees with ground truth):\n")
    for node in range(n_nodes_b):
        node_errors = [b['per_node_abs_error'][node] for b in b_all if node < len(b['per_node_abs_error'])]
        f.write(f"    Node {node} Mean Abs Error: {np.mean(node_errors):.6f}\n")
    f.write(f"    MAE: mean={np.mean(b_maes):.6f}, std={np.std(b_maes):.6f}, min={np.min(b_maes):.6f}, max={np.max(b_maes):.6f}\n")
    f.write(f"    RMSE: mean={np.mean(b_rmses):.6f}, std={np.std(b_rmses):.6f}, min={np.min(b_rmses):.6f}, max={np.max(b_rmses):.6f}\n")
    f.write(f"    Max Error: mean={np.mean(b_max_errs):.6f}, std={np.std(b_max_errs):.6f}, min={np.min(b_max_errs):.6f}, max={np.max(b_max_errs):.6f}\n")
    f.write(f"    Mean Relative Error: mean={np.mean(b_rel_errs):.4f}, std={np.std(b_rel_errs):.4f}\n\n")
    
    # === Variable C ===
    f.write("Variable C - Class Assignment [Binary one-hot, K x 4 leaf nodes]\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'#':<4} {'Valid File':<22} {'L0':>4} {'L1':>4} {'L2':>4} {'L3':>4} {'Acc':>8} {'ElemAcc':>8} {'Pred Classes':<16} {'GT Classes':<16}\n")
    f.write("-" * 100 + "\n")
    
    for idx, result in results_with_stats:
        valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0][:22]
        c = result['variable_stats']['C']
        m = c['per_leaf_match']
        m0 = m[0] if len(m) > 0 else '-'
        m1 = m[1] if len(m) > 1 else '-'
        m2 = m[2] if len(m) > 2 else '-'
        m3 = m[3] if len(m) > 3 else '-'
        f.write(f"{idx+1:<4} {valid_name:<22} {m0:>4} {m1:>4} {m2:>4} {m3:>4} "
                f"{c['class_assignment_accuracy']:>8.3f} {c['elementwise_accuracy']:>8.3f} "
                f"{str(c['pred_classes']):<16} {str(c['gt_classes']):<16}\n")
    
    c_all = [r['variable_stats']['C'] for _, r in results_with_stats]
    c_accs = [c['class_assignment_accuracy'] for c in c_all]
    c_elem_accs = [c['elementwise_accuracy'] for c in c_all]
    n_leaves = c_all[0]['n_leaves']
    
    f.write(f"\n  Aggregate C Statistics ({n_with_stats} trees with ground truth):\n")
    for leaf in range(n_leaves):
        leaf_matches = [c['per_leaf_match'][leaf] for c in c_all if leaf < len(c['per_leaf_match'])]
        leaf_acc = np.mean(leaf_matches) if leaf_matches else 0.0
        f.write(f"    Leaf {leaf} Accuracy: {leaf_acc:.4f} ({leaf_acc*100:.1f}%)\n")
    f.write(f"    Overall Class Assignment Accuracy: mean={np.mean(c_accs):.4f}, std={np.std(c_accs):.4f}, min={np.min(c_accs):.4f}, max={np.max(c_accs):.4f}\n")
    f.write(f"    Element-wise Accuracy: mean={np.mean(c_elem_accs):.4f}, std={np.std(c_elem_accs):.4f}\n\n")
    
    # === Variable D ===
    f.write("Variable D - Leaf Indicators [Binary, 3 decision nodes]\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'#':<4} {'Valid File':<22} {'N0':>4} {'N1':>4} {'N2':>4} {'Acc':>8} {'Pred D':<16} {'GT D':<16}\n")
    f.write("-" * 100 + "\n")
    
    for idx, result in results_with_stats:
        valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0][:22]
        d = result['variable_stats']['D']
        m = d['per_node_match']
        f.write(f"{idx+1:<4} {valid_name:<22} {m[0]:>4} {m[1]:>4} {m[2] if len(m) > 2 else '-':>4} "
                f"{d['accuracy']:>8.3f} {str(d['pred_D']):<16} {str(d['gt_D']):<16}\n")
    
    d_all = [r['variable_stats']['D'] for _, r in results_with_stats]
    d_accs = [d['accuracy'] for d in d_all]
    n_nodes_d = d_all[0]['n_nodes']
    
    f.write(f"\n  Aggregate D Statistics ({n_with_stats} trees with ground truth):\n")
    for node in range(n_nodes_d):
        node_matches = [d['per_node_match'][node] for d in d_all]
        node_acc = np.mean(node_matches)
        f.write(f"    Node {node} Accuracy: {node_acc:.4f} ({node_acc*100:.1f}%)\n")
    f.write(f"    Overall D Accuracy: mean={np.mean(d_accs):.4f}, std={np.std(d_accs):.4f}, min={np.min(d_accs):.4f}, max={np.max(d_accs):.4f}\n\n")
    
    # === Variable Z routing ===
    z_results = [(i, r) for i, r in results_with_stats
                 if r['variable_stats'].get('Z_routing') is not None]
    
    if z_results:
        f.write("Variable Z - Leaf Routing [Integer, N samples routed through tree]\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'#':<4} {'Valid File':<22} {'N':>6} {'Correct':>8} {'Rout Acc':>9} {'Per-Leaf Accuracy':<40}\n")
        f.write("-" * 100 + "\n")
        
        for idx, result in z_results:
            valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0][:22]
            z = result['variable_stats']['Z_routing']
            pla_str = ', '.join(f'{k}:{v:.3f}' for k, v in sorted(z['per_leaf_accuracy'].items()))
            f.write(f"{idx+1:<4} {valid_name:<22} {z['n_samples']:>6} {z['n_correct']:>8} "
                    f"{z['routing_accuracy']:>9.4f} {pla_str:<40}\n")
        
        z_all = [r['variable_stats']['Z_routing'] for _, r in z_results]
        z_rout_accs = [z['routing_accuracy'] for z in z_all]
        
        f.write(f"\n  Aggregate Z Routing Statistics ({len(z_results)} trees with ground truth):\n")
        f.write(f"    Routing Accuracy: mean={np.mean(z_rout_accs):.4f}, std={np.std(z_rout_accs):.4f}, "
                f"min={np.min(z_rout_accs):.4f}, max={np.max(z_rout_accs):.4f}\n")
        
        all_leaf_ids = set()
        for z in z_all:
            all_leaf_ids.update(z['per_leaf_accuracy'].keys())
        for leaf_id in sorted(all_leaf_ids):
            leaf_accs_list = [z['per_leaf_accuracy'][leaf_id] for z in z_all if leaf_id in z['per_leaf_accuracy']]
            if leaf_accs_list:
                f.write(f"    Leaf {leaf_id} Routing Accuracy: mean={np.mean(leaf_accs_list):.4f}, "
                        f"std={np.std(leaf_accs_list):.4f} (n={len(leaf_accs_list)})\n")
        f.write("\n")
    
    # === Combined Summary Table ===
    f.write("VARIABLE COMPARISON SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"  {'Variable':<20} {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}\n")
    f.write(f"  {'-'*20} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")
    
    f.write(f"  {'A (Features)':<20} {'Selection Accuracy':<25} {np.mean(a_accs):>8.4f} {np.std(a_accs):>8.4f} {np.min(a_accs):>8.4f} {np.max(a_accs):>8.4f}\n")
    f.write(f"  {'A (Features)':<20} {'Element-wise Accuracy':<25} {np.mean(a_elem_accs):>8.4f} {np.std(a_elem_accs):>8.4f} {np.min(a_elem_accs):>8.4f} {np.max(a_elem_accs):>8.4f}\n")
    f.write(f"  {'B (Thresholds)':<20} {'MAE':<25} {np.mean(b_maes):>8.4f} {np.std(b_maes):>8.4f} {np.min(b_maes):>8.4f} {np.max(b_maes):>8.4f}\n")
    f.write(f"  {'B (Thresholds)':<20} {'RMSE':<25} {np.mean(b_rmses):>8.4f} {np.std(b_rmses):>8.4f} {np.min(b_rmses):>8.4f} {np.max(b_rmses):>8.4f}\n")
    f.write(f"  {'C (Classes)':<20} {'Assignment Accuracy':<25} {np.mean(c_accs):>8.4f} {np.std(c_accs):>8.4f} {np.min(c_accs):>8.4f} {np.max(c_accs):>8.4f}\n")
    f.write(f"  {'C (Classes)':<20} {'Element-wise Accuracy':<25} {np.mean(c_elem_accs):>8.4f} {np.std(c_elem_accs):>8.4f} {np.min(c_elem_accs):>8.4f} {np.max(c_elem_accs):>8.4f}\n")
    f.write(f"  {'D (Leaf Ind.)':<20} {'Accuracy':<25} {np.mean(d_accs):>8.4f} {np.std(d_accs):>8.4f} {np.min(d_accs):>8.4f} {np.max(d_accs):>8.4f}\n")
    if z_results:
        z_rout_accs_summary = [r['variable_stats']['Z_routing']['routing_accuracy'] for _, r in z_results]
        f.write(f"  {'Z (Routing)':<20} {'Routing Accuracy':<25} {np.mean(z_rout_accs_summary):>8.4f} {np.std(z_rout_accs_summary):>8.4f} {np.min(z_rout_accs_summary):>8.4f} {np.max(z_rout_accs_summary):>8.4f}\n")
    f.write("\n")


def evaluate_on_test_dataset(test_path: str, tree: Dict, train_min: np.ndarray, train_max: np.ndarray) -> Optional[Dict]:
    """Evaluate the decision tree on a test dataset."""
    try:
        raw_data = process_file_to_numpy(test_path)
        if raw_data is None:
            return None
        
        test_features = raw_data[:, :-1]
        test_labels = raw_data[:, -1].astype(int)
        
        normalized_features = normalize_features_with_params(test_features, train_min, train_max)
        predictions = classify_samples(normalized_features, tree)
        accuracy = np.mean(predictions == test_labels)
        
        return {
            'accuracy': accuracy,
            'n_samples': len(test_labels),
            'n_correct': int(np.sum(predictions == test_labels)),
        }
    except Exception as e:
        return None


def find_dataset_files(base_path: str, dataset_name: str = None) -> List[str]:
    """Find dataset files in the given path for a specific dataset."""
    files = []
    
    if dataset_name:
        dataset_path = os.path.join(base_path, dataset_name)
        if os.path.exists(dataset_path):
            for root, _, filenames in os.walk(dataset_path):
                for f in filenames:
                    if f.endswith('.seeds') or f.endswith(f'.{dataset_name}'):
                        files.append(os.path.join(root, f))
        else:
            for root, _, filenames in os.walk(base_path):
                for f in filenames:
                    if (f.endswith('.seeds') or f.endswith(f'.{dataset_name}')) and dataset_name in root:
                        files.append(os.path.join(root, f))
    else:
        for root, _, filenames in os.walk(base_path):
            for f in filenames:
                if '.seeds' in f or any(ext in f for ext in ['.glass', '.body', '.iris', '.wine', '.banknote']):
                    files.append(os.path.join(root, f))
    
    return sorted(files)


def save_full_results(
    output_dir: str,
    all_valid_results: List[Dict],
    valid_files: List[str],
    test_files: List[str],
    ckpt_path: str,
    dataset_name: str,
    best_idx: int,
) -> str:
    """Save comprehensive full evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"gps_full_evaluation_{dataset_name}_{timestamp}.txt")
    
    # Filter successful results
    successful_results = [(i, r) for i, r in enumerate(all_valid_results) if r is not None]
    
    # Compute statistics across all trees
    all_mean_accuracies = [r['mean_test_accuracy'] for _, r in successful_results]
    all_valid_accuracies = [r['valid_accuracy'] for _, r in successful_results]
    
    with open(result_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("FULL CROSS-EVALUATION RESULTS\n")
        f.write("ALL VALIDATION TREES × ALL TEST DATASETS\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint: {ckpt_path}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Validation Files: {len(valid_files)}\n")
        f.write(f"Number of Test Files: {len(test_files)}\n")
        f.write(f"Successfully Evaluated Trees: {len(successful_results)}\n\n")

        # === DIAGNOSTICS SECTION ===
        f.write("=" * 90 + "\n")
        f.write("INPUT/ENCODER DIAGNOSTICS (Model Collapse Detection)\n")
        f.write("=" * 90 + "\n\n")

        # Collect diagnostics from all results
        has_diagnostics = False
        for idx, result in successful_results:
            if 'diagnostics' in result['predictions']:
                has_diagnostics = True
                diag = result['predictions']['diagnostics']
                f.write(f"File: {diag.get('file', 'N/A')}\n")
                f.write(f"  Input graph.x: shape={diag.get('graph_x_shape', 'N/A')}, "
                        f"mean={diag.get('graph_x_mean', 0):.6f}, "
                        f"std={diag.get('graph_x_std', 0):.6f}\n")
                f.write(f"  Pred B values: {diag.get('pred_b_values', 'N/A')}\n")
                f.write(f"  Pred logits: A(mean={diag.get('pred_a_mean', 0):.4f}, std={diag.get('pred_a_std', 0):.4f}), "
                        f"C(mean={diag.get('pred_c_mean', 0):.4f}, std={diag.get('pred_c_std', 0):.4f})\n")
                f.write("\n")

        if not has_diagnostics:
            f.write("No diagnostics available (run with updated code to generate)\n\n")

        # Check for collapse indicators
        if has_diagnostics:
            b_values_list = [result['predictions']['diagnostics'].get('pred_b_values', [])
                            for _, result in successful_results
                            if 'diagnostics' in result['predictions']]
            if len(b_values_list) > 1:
                b_array = np.array(b_values_list)
                b_std_across_files = np.std(b_array, axis=0)
                f.write("COLLAPSE DETECTION:\n")
                f.write(f"  Variable B std across files: {b_std_across_files}\n")
                if np.all(b_std_across_files < 0.01):
                    f.write("  ⚠️  WARNING: Variable B values are CONSTANT across all files (std < 0.01)\n")
                    f.write("  This indicates MODEL COLLAPSE - predictions are not conditioned on input!\n")
                else:
                    f.write("  ✓ Variable B shows variation across files (no collapse detected)\n")
                f.write("\n")

        # === BEST TREE ===
        f.write("=" * 90 + "\n")
        f.write("🏆 BEST PERFORMING TREE\n")
        f.write("=" * 90 + "\n\n")
        
        if best_idx >= 0 and all_valid_results[best_idx] is not None:
            best_result = all_valid_results[best_idx]
            best_valid_file = valid_files[best_idx]
            
            f.write(f"Source Validation File: {os.path.basename(best_valid_file)}\n")
            f.write(f"Validation File Index: {best_idx + 1}/{len(valid_files)}\n\n")
            
            f.write("Tree Structure:\n")
            f.write(f"  Selected Features: {list(best_result['tree']['selected_features'])}\n")
            f.write(f"  Thresholds: {[f'{t:.4f}' for t in best_result['tree']['thresholds']]}\n")
            f.write(f"  Leaf Classes: {list(best_result['tree']['leaf_classes'])}\n\n")
            
            f.write("Performance:\n")
            f.write(f"  Validation Accuracy: {best_result['valid_accuracy']:.4f} ({best_result['valid_accuracy']*100:.2f}%)\n")
            f.write(f"  Test Mean Accuracy: {best_result['mean_test_accuracy']:.4f} ({best_result['mean_test_accuracy']*100:.2f}%)\n")
            f.write(f"  Test Std Accuracy: {best_result['std_test_accuracy']:.4f}\n")
            f.write(f"  Test Min Accuracy: {best_result['min_test_accuracy']:.4f}\n")
            f.write(f"  Test Max Accuracy: {best_result['max_test_accuracy']:.4f}\n\n")
            
            # Tree visualization
            tree = best_result['tree']
            f.write("Tree Visualization:\n")
            f.write("```\n")
            f.write("                    [Node 0]\n")
            f.write(f"              Feature {tree['selected_features'][0]}, Threshold {tree['thresholds'][0]:.4f}\n")
            f.write("                  /         \\\n")
            f.write("              <              >=\n")
            f.write("             /                \\\n")
            f.write("        [Node 1]            [Node 2]\n")
            left_feat = tree['selected_features'][1] if len(tree['selected_features']) > 1 else '?'
            right_feat = tree['selected_features'][2] if len(tree['selected_features']) > 2 else '?'
            left_thresh = float(tree['thresholds'][1]) if len(tree['thresholds']) > 1 else 0.0
            right_thresh = float(tree['thresholds'][2]) if len(tree['thresholds']) > 2 else 0.0
            f.write(f"   F{left_feat}, T{left_thresh:.4f}")
            f.write(f"      F{right_feat}, T{right_thresh:.4f}\n")
            f.write("       /    \\              /    \\\n")
            f.write("      <     >=            <     >=\n")
            f.write("     /       \\          /       \\\n")
            f.write(f"  [L0]     [L1]     [L2]     [L3]\n")
            f.write(f" C={tree['leaf_classes'][0]}     C={tree['leaf_classes'][1]}     C={tree['leaf_classes'][2]}     C={tree['leaf_classes'][3]}\n")
            f.write("```\n\n")
            
            # First-stage variables for best tree
            f.write("First-Stage Variables (A, B, C, D):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Variable A (Feature Selection):\n{np.array2string(best_result['predictions']['pred_a'], precision=4)}\n\n")
            f.write(f"Variable B (Thresholds): {best_result['predictions']['pred_b']}\n\n")
            f.write(f"Variable C (Class Assignment):\n{np.array2string(best_result['predictions']['pred_c'], precision=4)}\n\n")
            f.write(f"Variable D (Leaf Indicators): {best_result['predictions']['pred_d']}\n\n")
        
        # === AGGREGATE STATISTICS ACROSS ALL TREES ===
        f.write("=" * 90 + "\n")
        f.write("AGGREGATE STATISTICS ACROSS ALL TREES\n")
        f.write("=" * 90 + "\n\n")
        
        if all_mean_accuracies:
            f.write("Test Accuracy (Mean of each tree's mean):\n")
            f.write(f"  Overall Mean: {np.mean(all_mean_accuracies):.4f} ({np.mean(all_mean_accuracies)*100:.2f}%)\n")
            f.write(f"  Overall Std:  {np.std(all_mean_accuracies):.4f}\n")
            f.write(f"  Best Tree:    {np.max(all_mean_accuracies):.4f} ({np.max(all_mean_accuracies)*100:.2f}%)\n")
            f.write(f"  Worst Tree:   {np.min(all_mean_accuracies):.4f} ({np.min(all_mean_accuracies)*100:.2f}%)\n\n")
            
            f.write("Validation Accuracy:\n")
            f.write(f"  Mean: {np.mean(all_valid_accuracies):.4f}\n")
            f.write(f"  Std:  {np.std(all_valid_accuracies):.4f}\n")
            f.write(f"  Min:  {np.min(all_valid_accuracies):.4f}\n")
            f.write(f"  Max:  {np.max(all_valid_accuracies):.4f}\n\n")
        
        # === TREE STRUCTURE ANALYSIS ===
        f.write("=" * 90 + "\n")
        f.write("TREE STRUCTURE ANALYSIS\n")
        f.write("=" * 90 + "\n\n")
        
        # Count unique tree structures
        tree_signatures = defaultdict(list)
        for idx, result in successful_results:
            sig = get_tree_signature(result['tree'])
            tree_signatures[sig].append(idx)
        
        f.write(f"Unique Tree Structures: {len(tree_signatures)}\n\n")
        
        # Feature usage statistics
        feature_usage = defaultdict(int)
        for idx, result in successful_results:
            for feat in result['tree']['selected_features']:
                feature_usage[int(feat)] += 1
        
        f.write("Feature Usage (across all trees):\n")
        for feat, count in sorted(feature_usage.items()):
            pct = 100.0 * count / (len(successful_results) * 3)  # 3 decision nodes per tree
            f.write(f"  Feature {feat}: {count} times ({pct:.1f}%)\n")
        f.write("\n")
        
        # Leaf class distribution
        leaf_class_patterns = defaultdict(int)
        for idx, result in successful_results:
            pattern = tuple(result['tree']['leaf_classes'])
            leaf_class_patterns[pattern] += 1
        
        f.write("Leaf Class Patterns (most common):\n")
        sorted_patterns = sorted(leaf_class_patterns.items(), key=lambda x: -x[1])[:10]
        for pattern, count in sorted_patterns:
            pct = 100.0 * count / len(successful_results)
            f.write(f"  {pattern}: {count} trees ({pct:.1f}%)\n")
        f.write("\n")
        
        # === VARIABLE-LEVEL COMPARISON ===
        f.write("=" * 90 + "\n")
        f.write("FIRST-STAGE & SECOND-STAGE VARIABLE COMPARISON (Predicted vs Ground Truth)\n")
        f.write("=" * 90 + "\n\n")
        
        write_variable_comparison_section(f, all_valid_results, valid_files)
        
        # === TOP 10 BEST TREES ===
        f.write("=" * 90 + "\n")
        f.write("TOP 10 BEST PERFORMING TREES\n")
        f.write("=" * 90 + "\n\n")
        
        # Sort by mean test accuracy
        sorted_results = sorted(successful_results, key=lambda x: -x[1]['mean_test_accuracy'])[:10]
        
        f.write(f"{'Rank':<5} {'Valid File':<25} {'Valid Acc':<12} {'Test Mean':<12} {'Test Std':<10} {'Features':<20} {'Leaf Classes':<15}\n")
        f.write("-" * 90 + "\n")
        
        for rank, (idx, result) in enumerate(sorted_results, 1):
            valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0]
            feats = str(list(result['tree']['selected_features']))
            classes = str(list(result['tree']['leaf_classes']))
            f.write(f"{rank:<5} {valid_name:<25} {result['valid_accuracy']:.4f}       {result['mean_test_accuracy']:.4f}       {result['std_test_accuracy']:.4f}     {feats:<20} {classes:<15}\n")
        f.write("\n")
        
        # === BOTTOM 10 WORST TREES ===
        f.write("=" * 90 + "\n")
        f.write("BOTTOM 10 WORST PERFORMING TREES\n")
        f.write("=" * 90 + "\n\n")
        
        sorted_results_worst = sorted(successful_results, key=lambda x: x[1]['mean_test_accuracy'])[:10]
        
        f.write(f"{'Rank':<5} {'Valid File':<25} {'Valid Acc':<12} {'Test Mean':<12} {'Test Std':<10} {'Features':<20} {'Leaf Classes':<15}\n")
        f.write("-" * 90 + "\n")
        
        for rank, (idx, result) in enumerate(sorted_results_worst, 1):
            valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0]
            feats = str(list(result['tree']['selected_features']))
            classes = str(list(result['tree']['leaf_classes']))
            f.write(f"{rank:<5} {valid_name:<25} {result['valid_accuracy']:.4f}       {result['mean_test_accuracy']:.4f}       {result['std_test_accuracy']:.4f}     {feats:<20} {classes:<15}\n")
        f.write("\n")
        
        # === ALL TREES SUMMARY TABLE ===
        f.write("=" * 90 + "\n")
        f.write("ALL TREES SUMMARY\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"{'#':<4} {'Valid File':<25} {'Valid Acc':<10} {'Test Mean':<10} {'Test Std':<10} {'Test Min':<10} {'Test Max':<10}\n")
        f.write("-" * 90 + "\n")
        
        for idx, result in enumerate(all_valid_results):
            valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0]
            if result is not None:
                f.write(f"{idx+1:<4} {valid_name:<25} {result['valid_accuracy']:.4f}     {result['mean_test_accuracy']:.4f}     {result['std_test_accuracy']:.4f}     {result['min_test_accuracy']:.4f}     {result['max_test_accuracy']:.4f}\n")
            else:
                f.write(f"{idx+1:<4} {valid_name:<25} {'FAILED':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10}\n")
        
        f.write("\n")
        
        # === Z AND L PREDICTIONS FOR ALL TREES ===
        f.write("=" * 90 + "\n")
        f.write("Z AND L PREDICTIONS FOR ALL TREES\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("Z Metrics (Leaf Routing Assignments)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'#':<4} {'Valid File':<25} {'Leaves':<8} {'Entropy':<10} {'Balance':<10} {'Max Size':<10} {'Min Size':<10}\n")
        f.write("-" * 90 + "\n")
        
        for idx, result in enumerate(all_valid_results):
            valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0]
            if result is not None and 'z_metrics' in result:
                z_m = result['z_metrics']
                f.write(f"{idx+1:<4} {valid_name:<25} {z_m['unique_leaves']:<8} {z_m['entropy']:<10.4f} {z_m['balance_ratio']:<10.4f} {z_m['max_leaf_size']:<10} {z_m['min_leaf_size']:<10}\n")
                f.write(f"       Leaf Distribution: {z_m['leaf_distribution']}\n")
            else:
                f.write(f"{idx+1:<4} {valid_name:<25} {'N/A':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<10}\n")
        
        f.write("\n")
        f.write("L Metrics (Dual Variables)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'#':<4} {'Valid File':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Sparsity':<10}\n")
        f.write("-" * 90 + "\n")
        
        for idx, result in enumerate(all_valid_results):
            valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0]
            if result is not None and 'l_metrics' in result:
                l_m = result['l_metrics']
                sparsity_pct = 100.0 * l_m['sparsity']
                f.write(f"{idx+1:<4} {valid_name:<25} {l_m['mean']:<12.6f} {l_m['std']:<12.6f} {l_m['min']:<12.6f} {l_m['max']:<12.6f} {sparsity_pct:<10.1f}%\n")
                f.write(f"       L shape: {l_m['shape']}, Sign counts: +{l_m['n_positive']} / 0:{l_m['n_zero']} / -{l_m['n_negative']}, L2 norm: {l_m['l2_norm']:.6f}\n")
            else:
                f.write(f"{idx+1:<4} {valid_name:<25} {'N/A':<12} {'-':<12} {'-':<12} {'-':<12} {'-':<10}\n")
        
        f.write("\n")
        
        # === Z AND L AGGREGATED STATISTICS ===
        f.write("=" * 90 + "\n")
        f.write("AGGREGATED Z AND L STATISTICS\n")
        f.write("=" * 90 + "\n\n")
        
        z_stats = [r['z_metrics'] for r in all_valid_results if r is not None and 'z_metrics' in r]
        l_stats = [r['l_metrics'] for r in all_valid_results if r is not None and 'l_metrics' in r]
        l_pred_errors = [r['l_prediction_error'] for r in all_valid_results if r is not None and 'l_prediction_error' in r and r['l_prediction_error'] is not None]
        
        if z_stats:
            f.write("Z Statistics Across All Trees:\n")
            z_entropies = [z['entropy'] for z in z_stats]
            z_balances = [z['balance_ratio'] for z in z_stats]
            z_leaves = [z['unique_leaves'] for z in z_stats]
            f.write(f"  Entropy: mean={np.mean(z_entropies):.4f}, std={np.std(z_entropies):.4f}, min={np.min(z_entropies):.4f}, max={np.max(z_entropies):.4f}\n")
            f.write(f"  Balance Ratio: mean={np.mean(z_balances):.4f}, std={np.std(z_balances):.4f}, min={np.min(z_balances):.4f}, max={np.max(z_balances):.4f}\n")
            f.write(f"  Unique Leaves Used: mean={np.mean(z_leaves):.2f}, min={np.min(z_leaves)}, max={np.max(z_leaves)}\n\n")
        
        if l_stats:
            f.write("L Statistics Across All Trees:\n")
            l_means = [l['mean'] for l in l_stats]
            l_stds = [l['std'] for l in l_stats]
            l_sparsities = [l['sparsity'] for l in l_stats]
            l_norms = [l['l2_norm'] for l in l_stats]
            f.write(f"  Mean: mean={np.mean(l_means):.6f}, std={np.std(l_means):.6f}, min={np.min(l_means):.6f}, max={np.max(l_means):.6f}\n")
            f.write(f"  Std: mean={np.mean(l_stds):.6f}, std={np.std(l_stds):.6f}, min={np.min(l_stds):.6f}, max={np.max(l_stds):.6f}\n")
            f.write(f"  Sparsity: mean={np.mean(l_sparsities)*100:.1f}%, std={np.std(l_sparsities)*100:.1f}%, min={np.min(l_sparsities)*100:.1f}%, max={np.max(l_sparsities)*100:.1f}%\n")
            f.write(f"  L2 Norm: mean={np.mean(l_norms):.6f}, std={np.std(l_norms):.6f}, min={np.min(l_norms):.6f}, max={np.max(l_norms):.6f}\n\n")
        
        # === EMPIRICAL L ANALYSIS ===
        f.write("=" * 90 + "\n")
        f.write("EMPIRICAL L ANALYSIS (Misclassifications using Predicted Tree)\n")
        f.write("=" * 90 + "\n\n")
        f.write("Empirical L = Sum of misclassification errors when classifying validation data with the PREDICTED tree structure\n")
        f.write("For each observation: loss = 1 if predicted_class != true_class, else 0\n")
        f.write("Empirical L = sum of all per-observation losses\n\n")
        
        empirical_L_list = [r['empirical_L_metrics'] for r in all_valid_results if r is not None and 'empirical_L_metrics' in r]
        
        if empirical_L_list:
            f.write(f"{'#':<4} {'Valid File':<25} {'Empirical L':<12} {'N Samples':<12} {'N Correct':<12} {'Accuracy':<12}\n")
            f.write("-" * 90 + "\n")
            
            for idx, result in enumerate(all_valid_results):
                if result is not None and 'empirical_L_metrics' in result:
                    valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0]
                    emp = result['empirical_L_metrics']
                    f.write(f"{idx+1:<4} {valid_name:<25} {emp['empirical_L']:<12} {emp['n_samples']:<12} {emp['n_correct']:<12} {emp['accuracy']:<12.4f}\n")
            
            f.write("\n")
            
            # Summary statistics for empirical L
            emp_L_vals = [e['empirical_L'] for e in empirical_L_list]
            emp_acc_vals = [e['accuracy'] for e in empirical_L_list]
            
            f.write("Empirical L Summary Statistics:\n")
            f.write(f"  Empirical L: mean={np.mean(emp_L_vals):.2f}, std={np.std(emp_L_vals):.2f}, min={np.min(emp_L_vals):.0f}, max={np.max(emp_L_vals):.0f}\n")
            f.write(f"  Accuracy: mean={np.mean(emp_acc_vals):.4f}, std={np.std(emp_acc_vals):.4f}, min={np.min(emp_acc_vals):.4f}, max={np.max(emp_acc_vals):.4f}\n\n")
        
        # === L PREDICTION ERROR ANALYSIS ===
        f.write("=" * 90 + "\n")
        f.write("L PREDICTION ERROR ANALYSIS (Predicted vs Ground Truth vs Empirical)\n")
        f.write("=" * 90 + "\n\n")
        f.write("Predicted L = Model's predicted L value from GPS architecture\n")
        f.write("Ground Truth L = Sum of misclassification errors using the OPTIMAL tree from hpc_datasets2/valid/<dataset>/outputs\n")
        f.write("Empirical L = Sum of misclassification errors using the PREDICTED tree structure\n\n")
        
        if l_pred_errors:
            f.write(f"{'#':<4} {'Valid File':<20} {'Pred L':<10} {'GT L':<10} {'Emp L':<10} {'Pred-GT':<10} {'Pred-Emp':<10} {'GT-Emp':<10}\n")
            f.write("-" * 90 + "\n")
            
            for idx, result in enumerate(all_valid_results):
                if result is not None and result.get('l_prediction_error'):
                    valid_name = os.path.splitext(os.path.basename(valid_files[idx]))[0][:20]
                    err = result['l_prediction_error']
                    emp_L = err.get('empirical_L', float('nan'))
                    pred_emp_err = err.get('pred_vs_empirical_error', float('nan'))
                    gt_emp_err = err.get('gt_vs_empirical_error', float('nan'))
                    f.write(f"{idx+1:<4} {valid_name:<20} {err['predicted_L']:<10.2f} {err['ground_truth_L']:<10.2f} {emp_L:<10.2f} {err['absolute_error']:<10.2f} {pred_emp_err:<10.2f} {gt_emp_err:<10.2f}\n")
            
            f.write("\n")
            
            # Summary statistics
            pred_vals = [e['predicted_L'] for e in l_pred_errors]
            truth_vals = [e['ground_truth_L'] for e in l_pred_errors]
            abs_errors = [e['absolute_error'] for e in l_pred_errors]
            rel_errors = [e['relative_error'] for e in l_pred_errors]
            correct_preds = sum(1 for e in l_pred_errors if e['prediction_correct'])
            
            # Empirical L comparison stats
            emp_vals = [e.get('empirical_L', np.nan) for e in l_pred_errors if 'empirical_L' in e]
            pred_emp_errors = [e.get('pred_vs_empirical_error', np.nan) for e in l_pred_errors if 'pred_vs_empirical_error' in e]
            gt_emp_errors = [e.get('gt_vs_empirical_error', np.nan) for e in l_pred_errors if 'gt_vs_empirical_error' in e]
            
            f.write("Summary Statistics:\n")
            f.write(f"  Predicted L: mean={np.mean(pred_vals):.2f}, std={np.std(pred_vals):.2f}, min={np.min(pred_vals):.2f}, max={np.max(pred_vals):.2f}\n")
            f.write(f"  Ground Truth L: mean={np.mean(truth_vals):.2f}, std={np.std(truth_vals):.2f}, min={np.min(truth_vals):.2f}, max={np.max(truth_vals):.2f}\n")
            if emp_vals:
                f.write(f"  Empirical L: mean={np.mean(emp_vals):.2f}, std={np.std(emp_vals):.2f}, min={np.min(emp_vals):.2f}, max={np.max(emp_vals):.2f}\n")
            f.write(f"\n")
            f.write(f"  Pred vs GT Error: mean={np.mean(abs_errors):.4f}, std={np.std(abs_errors):.4f}, min={np.min(abs_errors):.4f}, max={np.max(abs_errors):.4f}\n")
            if pred_emp_errors:
                f.write(f"  Pred vs Empirical Error: mean={np.mean(pred_emp_errors):.4f}, std={np.std(pred_emp_errors):.4f}, min={np.min(pred_emp_errors):.4f}, max={np.max(pred_emp_errors):.4f}\n")
            if gt_emp_errors:
                f.write(f"  GT vs Empirical Error: mean={np.mean(gt_emp_errors):.4f}, std={np.std(gt_emp_errors):.4f}, min={np.min(gt_emp_errors):.4f}, max={np.max(gt_emp_errors):.4f}\n")
            f.write(f"  Prediction Accuracy (within 0.5 of GT): {correct_preds}/{len(l_pred_errors)} ({100*correct_preds/len(l_pred_errors):.1f}%)\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 90 + "\n")
    
    print(f"[SUCCESS] Results saved to: {result_file}")
    return result_file


def is_hpc_environment():
    """Check if running on HPC."""
    if os.getenv("SLURM_JOB_ID"):
        return True
    if os.path.exists('/home/n/navarrodelacruz'):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Full cross-evaluation: all validation trees × all test datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation on glass dataset (original GPS architecture)
  python predict_full_evaluation.py --ckpt_path saved_models/best_288097.pt --dataset glass

  # Quick test with limited files
  python predict_full_evaluation.py --ckpt_path saved_models/best_288097.pt --dataset glass --max_valid 10 --max_test 10

  # GPS small architecture (explicit tree routing)
  python predict_full_evaluation.py --ckpt_path saved_models/best_small.pt --dataset seeds --model_type small

Available datasets: glass, seeds, small_toy, body, banknote, etc.
        """
    )
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--valid_dir', type=str, default='hpc_datasets2/valid',
                        help="Directory containing validation files")
    parser.add_argument('--test_dir', type=str, default='hpc_datasets2/test',
                        help="Directory containing test files")
    parser.add_argument('--output_dir', type=str, default='validation_predictions',
                        help="Output directory for results")
    parser.add_argument('--dataset', type=str, default='glass',
                        help="Dataset name (subdirectory in hpc_datasets2/)")
    parser.add_argument('--max_valid', type=int, default=None,
                        help="Maximum number of validation files (default: all)")
    parser.add_argument('--max_test', type=int, default=None,
                        help="Maximum number of test files (default: all)")
    parser.add_argument('--model_type', type=str, default='skipped5', choices=['skipped5', 'small'],
                        help="Model architecture: skipped5 (original) or small (explicit tree routing)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Determine paths
    if is_hpc_environment():
        base_dir = "/home/n/navarrodelacruz/two_stage_neural_diving"
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.getenv("HOME_REPO"):
        os.environ["HOME_REPO"] = base_dir
    
    # Resolve paths
    if not os.path.isabs(args.ckpt_path):
        args.ckpt_path = os.path.join(base_dir, args.ckpt_path)
    if not os.path.isabs(args.valid_dir):
        args.valid_dir = os.path.join(base_dir, args.valid_dir)
    if not os.path.isabs(args.test_dir):
        args.test_dir = os.path.join(base_dir, args.test_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
    
    # Find all validation files
    print(f"[INFO] Searching for dataset '{args.dataset}'...")
    valid_files = find_dataset_files(args.valid_dir, args.dataset)
    test_files = find_dataset_files(args.test_dir, args.dataset)
    
    if not valid_files:
        raise ValueError(f"No validation files found in {args.valid_dir}/{args.dataset}")
    if not test_files:
        raise ValueError(f"No test files found in {args.test_dir}/{args.dataset}")
    
    if args.max_valid:
        valid_files = valid_files[:args.max_valid]
    if args.max_test:
        test_files = test_files[:args.max_test]
    
    print(f"[INFO] Found {len(valid_files)} validation files")
    print(f"[INFO] Found {len(test_files)} test files")
    print(f"[INFO] Total evaluations: {len(valid_files)} trees × {len(test_files)} tests = {len(valid_files) * len(test_files)}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Add safety check when running on HPC
    if is_hpc_environment() and not torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("[ERROR] Running on HPC but no GPU detected!")
        print("=" * 70)
        print("[REASON] You are likely on a login node instead of a compute node.")
        print("         Login nodes don't have GPU access.\n")
        print("[FIX]    Use 'srun' to get an interactive GPU session:")
        print("         srun --partition=IMSE --gres=gpu:1 --cpus-per-task=32 --mem=64G --pty bash")
        print("         Then activate your conda environment and run the script.\n")
        print("[CHECK]  Verify GPU access with: nvidia-smi")
        print("=" * 70 + "\n")
        import sys
        sys.exit(1)

    # Load model config and model
    model_config = get_config(test=True)

    print("\n" + "=" * 70)
    print("STEP 1: Loading Model")
    print("=" * 70)
    model = load_model(args.ckpt_path, device, model_config, args.model_type)
    
    # === MAIN LOOP: Iterate over all validation files ===
    print("\n" + "=" * 70)
    print(f"STEP 2: Evaluating {len(valid_files)} Trees on {len(test_files)} Test Sets")
    print("=" * 70 + "\n")
    
    all_valid_results = []
    best_mean_accuracy = -1
    best_idx = -1
    
    for valid_idx, valid_file in enumerate(tqdm(valid_files, desc="Processing validation files")):
        valid_name = os.path.splitext(os.path.basename(valid_file))[0]
        
        # Predict on validation dataset
        predictions = predict_on_dataset(model, valid_file, device, model_type=args.model_type)
        if predictions is None:
            all_valid_results.append(None)
            continue
        
        # Build tree
        tree = build_decision_tree_structure(predictions)
        
        # Compute validation accuracy
        valid_preds = classify_samples(predictions['normalized_X'], tree)
        valid_accuracy = np.mean(valid_preds == predictions['true_labels'])
        
        # Get normalization parameters
        raw_valid = process_file_to_numpy(valid_file)
        if raw_valid is None:
            all_valid_results.append(None)
            continue
        valid_features_raw = raw_valid[:, :-1]
        train_min = np.min(valid_features_raw, axis=0)
        train_max = np.max(valid_features_raw, axis=0)
        
        # Evaluate on ALL test datasets
        test_accuracies = []
        for test_file in test_files:
            result = evaluate_on_test_dataset(test_file, tree, train_min, train_max)
            if result is not None:
                test_accuracies.append(result['accuracy'])
        
        if not test_accuracies:
            all_valid_results.append(None)
            continue
        
        # Compute statistics
        mean_acc = np.mean(test_accuracies)
        
        # Compute Z and L metrics
        z_metrics = compute_z_metrics(predictions['pred_z'], predictions['normalized_X'])
        l_metrics = compute_l_metrics(predictions['pred_L'])
        
        # Compute EMPIRICAL L using the predicted tree structure
        # This counts actual misclassifications when classifying validation data with predicted tree
        empirical_L_metrics = compute_empirical_L(valid_preds, predictions['true_labels'])
        empirical_L_value = empirical_L_metrics['empirical_L']
        
        # Load ground truth tree from outputs folder
        ground_truth_tree = load_ground_truth_tree(valid_file)
        
        # Compute ground truth L using optimal tree (from ground truth variables)
        ground_truth_L = compute_ground_truth_L(predictions['normalized_X'], predictions['true_labels'], ground_truth_tree)
        
        # Extract predicted L (should be a scalar)
        predicted_L_value = float(predictions['pred_L'].flatten()[0]) if hasattr(predictions['pred_L'], '__iter__') else float(predictions['pred_L'])
        
        # Compute L prediction error (comparing predicted L from GPS with ground truth and empirical L)
        l_prediction_error = compute_L_prediction_error(predicted_L_value, ground_truth_L, empirical_L_value) if not np.isnan(ground_truth_L) else None
        
        # Compute routing-based Z for predicted tree and ground truth tree
        pred_Z_routing = route_samples_through_tree_for_Z(predictions['normalized_X'], tree)
        gt_Z_routing = None
        if ground_truth_tree is not None:
            gt_Z_routing = route_samples_through_tree_for_Z(predictions['normalized_X'], ground_truth_tree)
        
        # Compute per-variable comparison statistics (A, B, C, D, Z vs ground truth)
        variable_stats = compute_all_variable_stats(tree, ground_truth_tree, pred_Z_routing, gt_Z_routing)
        
        result_entry = {
            'valid_file': valid_file,
            'predictions': predictions,
            'tree': tree,
            'ground_truth_tree': ground_truth_tree,
            'valid_accuracy': valid_accuracy,
            'test_accuracies': test_accuracies,
            'mean_test_accuracy': mean_acc,
            'std_test_accuracy': np.std(test_accuracies),
            'min_test_accuracy': np.min(test_accuracies),
            'max_test_accuracy': np.max(test_accuracies),
            'z_metrics': z_metrics,
            'l_metrics': l_metrics,
            'l_prediction_error': l_prediction_error,
            'empirical_L_metrics': empirical_L_metrics,
            'variable_stats': variable_stats,
        }
        
        all_valid_results.append(result_entry)
        
        # Track best
        if mean_acc > best_mean_accuracy:
            best_mean_accuracy = mean_acc
            best_idx = valid_idx
        
        # Progress update every 10 files
        if (valid_idx + 1) % 10 == 0:
            tqdm.write(f"  [{valid_idx + 1}/{len(valid_files)}] Current best: {best_mean_accuracy:.4f} from {os.path.basename(valid_files[best_idx])}")
    
    # === RESULTS SUMMARY ===
    print("\n" + "=" * 70)
    print("STEP 3: Saving Results")
    print("=" * 70)
    
    result_file = save_full_results(
        args.output_dir,
        all_valid_results,
        valid_files,
        test_files,
        args.ckpt_path,
        args.dataset,
        best_idx,
    )
    
    # Final summary
    successful = [r for r in all_valid_results if r is not None]
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Validation Files Processed: {len(successful)}/{len(valid_files)}")
    print(f"Test Files per Tree: {len(test_files)}")
    
    if best_idx >= 0 and all_valid_results[best_idx] is not None:
        best = all_valid_results[best_idx]
        print(f"\n🏆 BEST TREE:")
        print(f"   Source: {os.path.basename(valid_files[best_idx])}")
        print(f"   Validation Accuracy: {best['valid_accuracy']:.4f}")
        print(f"   Test Mean Accuracy: {best['mean_test_accuracy']:.4f} ± {best['std_test_accuracy']:.4f}")
        print(f"   Test Range: [{best['min_test_accuracy']:.4f}, {best['max_test_accuracy']:.4f}]")
        print(f"   Features: {list(best['tree']['selected_features'])}")
        print(f"   Thresholds: {[f'{t:.4f}' for t in best['tree']['thresholds']]}")
        print(f"   Leaf Classes: {list(best['tree']['leaf_classes'])}")
    
    if successful:
        all_means = [r['mean_test_accuracy'] for r in successful]
        print(f"\nOverall Statistics (across {len(successful)} trees):")
        print(f"   Mean of Means: {np.mean(all_means):.4f}")
        print(f"   Best Tree: {np.max(all_means):.4f}")
        print(f"   Worst Tree: {np.min(all_means):.4f}")
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
