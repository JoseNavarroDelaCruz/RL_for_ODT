import numpy as np
import pandas as pd
import torch
import re
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import re
from typing import Dict, List, Tuple, Set, Optional
# --------------- EXTRACT VARIABLES DATA -------------------- #

available_cpus =  int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())) # Gets all logical CPUs on the node

#Variables:
# a -  first-stage
# b -  first-stage
# c - first-stage
# d - first-stage
# x - first-stage
# y - first-stage

# z - second-stage
# L - second-stage

# Positional feature added for Z nodes (small numeric positional signal)
# This augments the per-variable feature vector with `Z_POS_DIM` extra columns.
Z_POS_DIM = 4



def build_variable_name(var_name, row_idx, col_idx):
    """
    Build a descriptive name for the variable element using subscript notation.

    For each variable, we decide which indices represent (s, t, k, j, p, etc.).
    You may need to adjust these rules according to your actual problem setup.
    """
    # Examples below assume:
    #   a_{j,t} -> shape (P, T_D)
    #   b_t     -> shape (T_D,)
    #   c_{k,t} -> shape (K, T_L)
    #   d_t     -> shape (T_D,)
    #   L_s     -> shape (S,)
    #   z_{s,t} -> shape (S, T_L)
    #   y_{s,k} -> shape (S, K)
    #   x_{s,p} -> shape (S, P)
    
    if var_name == 'variable_a':
        # a_{j,t}: row -> j, col -> t
        return f"variable_a_j{row_idx}_t{col_idx}"
    elif var_name == 'variable_b':
        # b_t: row -> t
        return f"variable_b_t{row_idx}"
    elif var_name == 'variable_c':
        # c_{k,t}: row -> k, col -> t
        return f"variable_c_k{row_idx}_t{col_idx}"
    elif var_name == 'variable_d':
        # d_t: row -> t
        return f"variable_d_t{row_idx}"
    elif var_name == 'variable_L':
        # L_s: row -> s
        return f"variable_L_s{row_idx}"
    elif var_name == 'variable_z':
        # z_{s,t}: row -> s, col -> t
        return f"variable_z_s{row_idx}_t{col_idx}"
    elif var_name == 'variable_y':
        # y_{s,k}: row -> s, col -> k
        return f"variable_y_s{row_idx}_k{col_idx}"
    elif var_name == 'variable_x':
        # x_{s,p}: row -> s, col -> p
        return f"variable_x_s{row_idx}_p{col_idx}"
    else:
        # Fallback
        return f"{var_name}_{row_idx}_{col_idx}"




# Function to extract variable info
def extract_variable_info(var_name):
    if var_name.startswith('variable_a') or var_name.startswith('variable_c') or \
       var_name.startswith('variable_d') or var_name.startswith('variable_L') or \
       var_name.startswith('variable_z') or var_name.startswith('variable_y'):
        var_type = 'binary'
        lower_bound = 0
        upper_bound = 1
    elif var_name.startswith('variable_b') or var_name.startswith('variable_x'):
        var_type = 'continuous'
        lower_bound = 0  # Assuming splits are between 0 and 1
        upper_bound = 1
    else:
        var_type = 'unknown'
        lower_bound = None
        upper_bound = None

    if var_name.startswith('variable_a'):
        node_type_split = 1
        node_type_threshold = 0
        node_type_leaf = 0
        node_type_activation = 0
        node_type_prediction = 0
        node_type_dataset = 0
        node_type_label = 0
        node_type_loss = 0

    if var_name.startswith('variable_b'):
        node_type_split = 0
        node_type_threshold = 1
        node_type_leaf = 0
        node_type_activation = 0
        node_type_prediction = 0
        node_type_dataset = 0
        node_type_label = 0
        node_type_loss = 0

    if var_name.startswith('variable_c'):
        node_type_split = 0
        node_type_threshold = 0
        node_type_leaf = 1
        node_type_activation = 0
        node_type_prediction = 0
        node_type_dataset = 0
        node_type_label = 0
        node_type_loss = 0
        
    if var_name.startswith('variable_d'):
        node_type_split = 0
        node_type_threshold = 0
        node_type_leaf = 0
        node_type_activation = 1
        node_type_prediction = 0
        node_type_dataset = 0
        node_type_label = 0
        node_type_loss = 0

    if var_name.startswith('variable_z'):
        node_type_split = 0
        node_type_threshold = 0
        node_type_leaf = 0
        node_type_activation = 0
        node_type_prediction = 1
        node_type_dataset = 0
        node_type_label = 0
        node_type_loss = 0

    if var_name.startswith('variable_x'):
        node_type_split = 0
        node_type_threshold = 0
        node_type_leaf = 0
        node_type_activation = 0
        node_type_prediction = 0
        node_type_dataset = 1
        node_type_label = 0
        node_type_loss = 0

    if var_name.startswith('variable_y'):
        node_type_split = 0
        node_type_threshold = 0
        node_type_leaf = 0
        node_type_activation = 0
        node_type_prediction = 0
        node_type_dataset = 0
        node_type_label = 1
        node_type_loss = 0
        node_type_loss = 0

    if var_name.startswith('variable_L'):
        node_type_split = 0
        node_type_threshold = 0
        node_type_leaf = 0
        node_type_activation = 0
        node_type_prediction = 0
        node_type_dataset = 0
        node_type_label = 0
        node_type_loss = 1

    variable_info = {
        'name': var_name,
        'type': var_type,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'node_type_split': node_type_split,
        'node_type_threshold': node_type_threshold,
        'node_type_leaf': node_type_leaf,
        'node_type_activation': node_type_activation,
        'node_type_prediction': node_type_prediction,
        'node_type_dataset': node_type_dataset,
        'node_type_label': node_type_label,
        'node_type_loss': node_type_loss
    }
    variable_info['has_lower_bound'] = 1 if lower_bound is not None else 0
    variable_info['has_upper_bound'] = 1 if upper_bound is not None else 0

    return variable_info



# Function to check bounds
def check_bounds(variable, lower_bound, upper_bound):
    if lower_bound is not None:
        at_lower_bound = np.isclose(variable, lower_bound).astype(int)
    else:
        at_lower_bound = np.zeros(variable.shape, dtype=int)

    if upper_bound is not None:
        at_upper_bound = np.isclose(variable, upper_bound).astype(int)
    else:
        at_upper_bound = np.zeros(variable.shape, dtype=int)

    return at_lower_bound, at_upper_bound




def compute_structural_features(var_name, row_idx, col_idx, tree_depth):
    def get_depth(index):
        node_count = 0
        depth = 0
        while True:
            width = 2 ** depth
            if index < node_count + width:
                return depth
            node_count += width
            depth += 1

    # Default values
    depth = -1
    is_root = 0
    leaf = 0
    is_left = 0
    is_right = 0

    if var_name == 'variable_a':
        depth = get_depth(col_idx)
        is_root = int(depth == 0)
        leaf = 0
        if depth > 0:
            rel_idx = col_idx - (2 ** (depth - 1))
            is_left = int(rel_idx < 2 ** (depth - 1))
            is_right = 1 - is_left

    elif var_name in ['variable_b', 'variable_d']:
        depth = get_depth(row_idx)
        is_root = int(depth == 0)
        leaf = 0
        if depth > 0:
            rel_idx = row_idx - (2 ** (depth - 1))
            is_left = int(rel_idx < 2 ** (depth - 1))
            is_right = 1 - is_left

    elif var_name in ['variable_c', 'variable_z']:
        depth = tree_depth
        leaf = col_idx + 1
        is_left = int(col_idx < 2 ** (tree_depth - 1))
        is_right = 1 - is_left
        

    # X, Y, L are not tree-structured
    return depth, is_root, leaf, is_left, is_right


def assign_decision_node_id(var_name, row_idx, col_idx, tree_depth):
    """
    Return an integer ID indicating the decision node this variable belongs to.
    Only applies to variables A, B, D. Others get ID 0.
    """
    if var_name == 'variable_a':
        return col_idx + 1  # column index maps to decision node
    elif var_name in ['variable_b', 'variable_d']:
        return row_idx + 1  # row index maps to decision node
    else:
        return 0  # not a decision node

def assign_feature_id(var_name, row_idx, col_idx):
    """
    Assigns a feature ID based on which feature the variable is related to.
    - Applies only to variable_a (row index j) and variable_x (col index p).
    - All others return 0.
    """
    if var_name == 'variable_a':
        return row_idx + 1  # j = feature index
    elif var_name == 'variable_x':
        return col_idx + 1  # p = feature index
    else:
        return 0  # Not related to a feature


def compute_class_features_from_dataset(X: np.ndarray, y: np.ndarray):
    """
    Given raw dataset with features and class labels, compute:
    - class_prior_k: fraction of samples in each class
    - class_feature_mean_k: average feature vector for each class
    - normalized_class_id: index / num_classes for each class
    """
    assert X.shape[0] == y.shape[0], "Mismatched rows between X and y"

    classes = np.unique(y)
    num_classes = len(classes)
    n_rows, n_cols = X.shape
    class_prior_k = {}
    class_feature_mean_k = {}
    normalized_class_id = {}

    for class_id in classes:
        indices = np.where(y == class_id)[0]
        class_samples = X[indices]
        prior = len(indices) / len(y)
        class_prior_k[class_id] = prior
        class_feature_mean_k[class_id] = float(np.mean(np.mean(class_samples, axis=0)))
        normalized_class_id[class_id] = class_id / num_classes

    return class_prior_k, class_feature_mean_k, normalized_class_id, num_classes, n_rows, n_cols


# Constants for fixed feature dimensions (class-agnostic)
X_AGG_DIM = 7  # Fixed dimension for all variable types

# NEW: Constants for variable type encoding
VAR_TYPE_DIM = 5  # 4 one-hot type encoding + 1 within-type position


def compute_variable_type_encoding(var_name):
    """
    One-hot encoding for variable type: [is_a, is_b, is_c, is_d]

    This provides explicit inter-group differentiation that is invariant
    to message passing in GPS layers.

    Args:
        var_name: Variable name string ('variable_a', 'variable_b', etc.)

    Returns:
        List of 4 floats representing one-hot encoding
    """
    type_map = {
        'variable_a': [1.0, 0.0, 0.0, 0.0],
        'variable_b': [0.0, 1.0, 0.0, 0.0],
        'variable_c': [0.0, 0.0, 1.0, 0.0],
        'variable_d': [0.0, 0.0, 0.0, 1.0],
    }
    # Return zeros for non-first-stage variables (x, y, z, L)
    return type_map.get(var_name, [0.0, 0.0, 0.0, 0.0])


def compute_within_type_position(var_name, row_idx, col_idx, shape):
    """
    Compute normalized position within the variable type group.

    This provides intra-group differentiation by assigning a unique
    position index to each element within a variable type.

    Args:
        var_name: Variable name string
        row_idx: Row index in variable matrix
        col_idx: Column index in variable matrix
        shape: Shape tuple of the variable (e.g., (P, T_D) for a)

    Returns:
        Float in [0, 1] representing normalized position within type
    """
    if len(shape) == 1:
        # 1D variables: b, d, L
        flat_idx = row_idx
        total = shape[0]
    elif len(shape) == 2:
        # 2D variables: a, c, x, y, z
        flat_idx = row_idx * shape[1] + col_idx
        total = shape[0] * shape[1]
    else:
        return 0.0

    # Normalize to [0, 1], handle edge case of single element
    if total <= 1:
        return 0.0
    return float(flat_idx) / float(total - 1)


def compute_x_aggregated_features(variable_name, row_idx, col_idx, X, y, num_classes):
    """
    Compute observation-specific X-aggregated features for each variable type.

    IMPORTANT: Features are CLASS-AGNOSTIC - they work for any num_classes (2-6).
    Uses summary statistics that aggregate across classes rather than per-class features.

    Args:
        variable_name: 'variable_a', 'variable_b', 'variable_c', 'variable_d'
        row_idx: Row index in variable matrix (feature j for A, class k for C)
        col_idx: Column index (decision node t)
        X: Normalized features array, shape (S, P)
        y: Class labels array, shape (S,)
        num_classes: Number of classes K (2-6)

    Returns:
        List of floats: X-aggregated features (fixed length = X_AGG_DIM = 7)
    """
    x_agg = []

    if variable_name == 'variable_a':
        # A_{j,t}: row_idx = j (feature index)
        # CLASS-AGNOSTIC statistics for feature j
        if row_idx < X.shape[1]:
            feature_j = X[:, row_idx]

            # 1. Compute class-conditional means and aggregate
            class_means = []
            class_stds = []
            for k in range(num_classes):
                class_mask = (y == k)
                if class_mask.sum() > 0:
                    class_means.append(np.mean(feature_j[class_mask]))
                    class_stds.append(np.std(feature_j[class_mask]))

            # Feature 1: Max separation between any two classes (Fisher-like)
            if len(class_means) >= 2:
                max_sep = max(abs(class_means[i] - class_means[j])
                              for i in range(len(class_means))
                              for j in range(i+1, len(class_means)))
                pooled_std = np.std(feature_j) + 1e-8
                x_agg.append(float(max_sep / pooled_std))
            else:
                x_agg.append(0.0)

            # Feature 2: Between-class variance of means
            if len(class_means) >= 2:
                x_agg.append(float(np.var(class_means)))
            else:
                x_agg.append(0.0)

            # Feature 3: Mean of within-class standard deviations
            if len(class_stds) > 0:
                x_agg.append(float(np.mean(class_stds)))
            else:
                x_agg.append(0.0)

            # Feature 4: Range of class means (max - min)
            if len(class_means) >= 2:
                x_agg.append(float(max(class_means) - min(class_means)))
            else:
                x_agg.append(0.0)

            # Feature 5: Coefficient of variation of class means
            if len(class_means) >= 2 and np.mean(class_means) != 0:
                x_agg.append(float(np.std(class_means) / (abs(np.mean(class_means)) + 1e-8)))
            else:
                x_agg.append(0.0)

            # Feature 6: Skewness of feature distribution
            std_j = np.std(feature_j)
            if std_j > 1e-8:
                x_agg.append(float(np.mean((feature_j - np.mean(feature_j))**3) / (std_j**3 + 1e-8)))
            else:
                x_agg.append(0.0)

            # Feature 7: Fraction of samples in majority class for this feature's median split
            median_val = np.median(feature_j)
            above_median = feature_j >= median_val
            if above_median.sum() > 0:
                # Most common class among samples above median
                classes_above = y[above_median]
                majority_frac = max(np.sum(classes_above == k) for k in range(num_classes)) / len(classes_above)
                x_agg.append(float(majority_frac))
            else:
                x_agg.append(0.0)
        else:
            x_agg = [0.0] * X_AGG_DIM

    elif variable_name == 'variable_b':
        # B_t: threshold at decision node t
        # Global quantiles as candidate threshold values (already class-agnostic)
        x_flat = X.flatten()
        x_agg.append(float(np.percentile(x_flat, 10)))
        x_agg.append(float(np.percentile(x_flat, 25)))
        x_agg.append(float(np.percentile(x_flat, 50)))
        x_agg.append(float(np.percentile(x_flat, 75)))
        x_agg.append(float(np.percentile(x_flat, 90)))
        # Feature 6: IQR (interquartile range)
        x_agg.append(float(np.percentile(x_flat, 75) - np.percentile(x_flat, 25)))
        # Feature 7: Range
        x_agg.append(float(np.max(x_flat) - np.min(x_flat)))

    elif variable_name == 'variable_c':
        # C_{k,t}: row_idx = k (class index for this C node)
        # CLASS-AGNOSTIC: statistics about this specific class k
        class_mask = (y == row_idx)

        # Feature 1: Prior probability of this class
        x_agg.append(float(np.mean(class_mask)))

        # Feature 2: Mean of all features for this class
        if class_mask.sum() > 0:
            x_agg.append(float(np.mean(X[class_mask])))
        else:
            x_agg.append(0.0)

        # Feature 3: Std of all features for this class
        if class_mask.sum() > 0:
            x_agg.append(float(np.std(X[class_mask]) + 1e-8))
        else:
            x_agg.append(0.0)

        # Feature 4: How different is this class from overall mean?
        if class_mask.sum() > 0:
            class_mean = np.mean(X[class_mask])
            overall_mean = np.mean(X)
            overall_std = np.std(X) + 1e-8
            x_agg.append(float((class_mean - overall_mean) / overall_std))
        else:
            x_agg.append(0.0)

        # Feature 5: Relative class size (this class vs largest class)
        class_counts = np.array([np.sum(y == k) for k in range(num_classes)])
        if class_counts.max() > 0:
            x_agg.append(float(class_mask.sum() / class_counts.max()))
        else:
            x_agg.append(0.0)

        # Feature 6: Is this the majority class?
        x_agg.append(float(1.0 if class_mask.sum() == class_counts.max() else 0.0))

        # Feature 7: Normalized class index (0 to 1 range)
        x_agg.append(float(row_idx / (num_classes - 1 + 1e-8)))

    elif variable_name == 'variable_d':
        # D_t: leaf indicator at node t
        # Global class distribution statistics (already class-agnostic)
        class_counts = np.array([np.sum(y == k) for k in range(num_classes)])
        class_probs = class_counts / (len(y) + 1e-8)

        # Feature 1: Class entropy (higher = more balanced)
        entropy = -np.sum(class_probs * np.log(class_probs + 1e-8))
        x_agg.append(float(entropy))

        # Feature 2: Normalized entropy (0-1 scale)
        max_entropy = np.log(num_classes)
        x_agg.append(float(entropy / (max_entropy + 1e-8)))

        # Feature 3: Class imbalance ratio (min/max)
        if class_counts.max() > 0:
            x_agg.append(float(class_counts.min() / class_counts.max()))
        else:
            x_agg.append(0.0)

        # Feature 4: Gini impurity
        gini = 1.0 - np.sum(class_probs ** 2)
        x_agg.append(float(gini))

        # Feature 5: Majority class proportion
        x_agg.append(float(class_counts.max() / (len(y) + 1e-8)))

        # Feature 6: Number of classes (normalized, assuming max 6)
        x_agg.append(float(num_classes / 6.0))

        # Feature 7: Sample size indicator (log-scaled)
        x_agg.append(float(np.log(len(y) + 1) / 10.0))  # Normalized log

    else:
        # X, Y, Z, L nodes: zeros (no X-aggregated features needed)
        x_agg = [0.0] * X_AGG_DIM

    # Ensure exactly X_AGG_DIM features
    assert len(x_agg) == X_AGG_DIM, f"Expected {X_AGG_DIM} features, got {len(x_agg)}"
    return x_agg


def generate_variable_elements(variable,
                                        variable_name,
                                        index_counter,
                                        class_feature_mean_k,
                                        class_prior_k=None,
                                        normalized_class_id=None,
                                        feature_stats=None,
                                        linear_features=None,
                                        warm_start_var=None,
                                        final_solution_var=None,
                                        num_classes=None,
                                        n_rows=None, n_cols=None,
                                        X=None,  # NEW: Raw features for X-aggregated features
                                        y=None   # NEW: Class labels for X-aggregated features
                                        ):
    """
    Generate variable elements with LP-based features.
    
    New features computed (for a, b, c, d):
    - distance_to_closest_integer: |LP_solution - MILP_solution|
    - fractional_distance: |LP_value - 0.5| (measures indecisiveness)
    - lp_confidence_score: 1 - 2*|LP_value - 0.5| (normalized confidence)
    - warm_start_value: The warm start solution (CART initial solution)
    - cart_agreement: 1.0 if warm_start == MILP_optimal, 0.0 otherwise
    """
    # variable_name = var_name
    # variable = var_data
    variable_elements = []
    variable_info = extract_variable_info(variable_name)
    shape = variable.shape

    if len(shape) == 1:
        for row_idx in range(shape[0]):
            class_id = row_idx 
            subscripted_name = build_variable_name(variable_name, row_idx, 0)
            depth, is_root, leaf, is_left, is_right = compute_structural_features(variable_name, row_idx, 0, 2)
            decision_node_id = assign_decision_node_id(variable_name, row_idx, 0, 2)
            feature_id = assign_feature_id(variable_name, row_idx, 0)
            match = re.search(r'_k(\d+)_', subscripted_name) if 'variable_c' in subscripted_name else None

            if variable_name == 'variable_a':
                mean_val = float(feature_stats['mean'][row_idx])
                std_val = float(feature_stats['std'][row_idx])
                skew_val = float(feature_stats['skew'][row_idx])
                kurt_val = float(feature_stats['kurt'][row_idx])
            else:
                mean_val = std_val = skew_val = kurt_val = 0.0

            # Extract linear feature value for this element
            linear_value = 0.0
            if linear_features is not None:
                linear_value = float(linear_features[row_idx])

            # Compute new LP-based features
            warm_start_value = 0.0
            distance_to_integer = 0.0
            
            if warm_start_var is not None:
                warm_start_value = float(warm_start_var[row_idx])
            
            if linear_features is not None:
                # distance_to_closest_integer: |LP_value - round(LP_value)|
                # This measures how fractional the LP solution is (0 = integer, 0.5 = max fractional)
                distance_to_integer = abs(linear_value - round(linear_value))

            # Extract sample_value for variable_x and variable_y (1D case)
            # For 1D variables, this is typically b, d, L which don't have sample values
            sample_value = 0.0

            # Compute X-aggregated features (class-agnostic)
            x_agg_features = [0.0] * X_AGG_DIM
            if X is not None and y is not None and num_classes is not None:
                x_agg_features = compute_x_aggregated_features(
                    variable_name, row_idx, 0, X, y, num_classes
                )

            # NEW: Compute variable type encoding and within-type position
            var_type_encoding = compute_variable_type_encoding(variable_name)
            within_type_pos = compute_within_type_position(variable_name, row_idx, 0, shape)

            element_info = {
                'index': index_counter,
                'name': subscripted_name,
                'var_type': variable_info['type'],
                'is_binary': 1 if variable_info['type'] == 'binary' else 0,
                'lower_bound': variable_info['lower_bound'],
                'upper_bound': variable_info['upper_bound'],
                'has_lower_bound': variable_info['has_lower_bound'],
                'has_upper_bound': variable_info['has_upper_bound'],
                'node_type_split': variable_info['node_type_split'],
                'node_type_threshold': variable_info['node_type_threshold'],
                'node_type_leaf': variable_info['node_type_leaf'],
                'node_type_activation': variable_info['node_type_activation'],
                'node_type_prediction': variable_info['node_type_prediction'],
                'node_type_dataset': variable_info['node_type_dataset'],
                'node_type_label': variable_info['node_type_label'],
                'node_type_loss': variable_info['node_type_loss'],
                'depth_level': depth,
                'is_root_node': is_root,
                'leaf_node': leaf,
                'is_left_subtree': is_left,
                'is_right_subtree': is_right,
                'decision_node_id': decision_node_id,
                'feature_id': feature_id,
                'class_prior_k': class_prior_k.get(class_id+1, 0.0) if class_prior_k is not None else 0,
                'class_feature_mean_k': class_feature_mean_k.get(class_id+1, 0.0) if class_feature_mean_k is not None and 'variable_c' in variable_name else 0,
                'normalized_class_id': normalized_class_id.get(class_id+1, 0.0) if normalized_class_id is not None else 0,
                'class_embed_token': class_id if class_id is not None and 'variable_c' in variable_name else 0,
                'feature_mean': mean_val,
                'feature_std': std_val,
                'feature_skewness': skew_val,
                'feature_kurtosis': kurt_val,
                'num_classes': num_classes if num_classes is not None else 0,
                'n_rows': n_rows if n_rows is not None else 0,
                'n_cols': n_cols if n_cols is not None else 0,
                'linear_relaxation': linear_value,
                'distance_to_integer': distance_to_integer,
                'warm_start_value': warm_start_value,
                'sample_value': sample_value,
                'x_agg_features': x_agg_features,  # X-aggregated features
                'var_type_encoding': var_type_encoding,  # NEW: Variable type one-hot
                'within_type_pos': within_type_pos  # NEW: Position within type
            }

            variable_elements.append(element_info)
            index_counter += 1
    # row_idx=col_idx=0
    elif len(shape) == 2:
        for row_idx in range(shape[0]):
            for col_idx in range(shape[1]):
                subscripted_name = build_variable_name(variable_name, row_idx, col_idx)
                depth, is_root, leaf, is_left, is_right = compute_structural_features(variable_name, row_idx, col_idx, 2)
                decision_node_id = assign_decision_node_id(variable_name, row_idx, col_idx, 2)
                feature_id = assign_feature_id(variable_name, row_idx, col_idx)
                match = re.search(r'_k(\d+)_', subscripted_name) if 'variable_c' in subscripted_name else None
                class_id = row_idx
                if variable_name == 'variable_a':
                    mean_val = float(feature_stats['mean'][row_idx])
                    std_val = float(feature_stats['std'][row_idx])
                    skew_val = float(feature_stats['skew'][row_idx])
                    kurt_val = float(feature_stats['kurt'][row_idx])
                else:
                    mean_val = std_val = skew_val = kurt_val = 0.0


                linear_value = 0.0
                if linear_features is not None:
                    linear_value = float(linear_features[row_idx, col_idx])
                
                # Compute new LP-based features
                warm_start_value = 0.0
                distance_to_integer = 0.0
                
                if warm_start_var is not None:
                    warm_start_value = float(warm_start_var[row_idx, col_idx])
                
                if linear_features is not None:
                    # distance_to_closest_integer: |LP_value - round(LP_value)|
                    # This measures how fractional the LP solution is (0 = integer, 0.5 = max fractional)
                    distance_to_integer = abs(linear_value - round(linear_value))

                # Extract sample_value for variable_x and variable_y
                # For variable_x: actual normalized feature value x_{s,p}
                # For variable_y: one-hot label value y_{s,k}
                # For others: 0.0 (not applicable)
                sample_value = 0.0
                if variable_name in ['variable_x', 'variable_y']:
                    sample_value = float(variable[row_idx, col_idx])

                # Compute X-aggregated features (class-agnostic)
                x_agg_features = [0.0] * X_AGG_DIM
                if X is not None and y is not None and num_classes is not None:
                    x_agg_features = compute_x_aggregated_features(
                        variable_name, row_idx, col_idx, X, y, num_classes
                    )

                # NEW: Compute variable type encoding and within-type position
                var_type_encoding = compute_variable_type_encoding(variable_name)
                within_type_pos = compute_within_type_position(variable_name, row_idx, col_idx, shape)

                element_info = {
                    'index': index_counter,
                    'name': subscripted_name,
                    'var_type': variable_info['type'],
                    'is_binary': 1 if variable_info['type'] == 'binary' else 0,
                    'lower_bound': variable_info['lower_bound'],
                    'upper_bound': variable_info['upper_bound'],
                    'has_lower_bound': variable_info['has_lower_bound'],
                    'has_upper_bound': variable_info['has_upper_bound'],
                    'node_type_split': variable_info['node_type_split'],
                    'node_type_threshold': variable_info['node_type_threshold'],
                    'node_type_leaf': variable_info['node_type_leaf'],
                    'node_type_activation': variable_info['node_type_activation'],
                    'node_type_prediction': variable_info['node_type_prediction'],
                    'node_type_dataset': variable_info['node_type_dataset'],
                    'node_type_label': variable_info['node_type_label'],
                    'node_type_loss': variable_info['node_type_loss'],
                    'depth_level': depth,
                    'is_root_node': is_root,
                    'leaf_node': leaf,
                    'is_left_subtree': is_left,
                    'is_right_subtree': is_right,
                    'decision_node_id': decision_node_id,
                    'feature_id': feature_id,
                    'class_prior_k': class_prior_k.get(class_id+1, 0.0) if class_prior_k is not None else 0,
                    'class_feature_mean_k': class_feature_mean_k.get(class_id+1, 0.0) if class_feature_mean_k is not None and 'variable_c' in variable_name else 0,
                    'normalized_class_id': normalized_class_id.get(class_id+1, 0.0) if normalized_class_id is not None else 0,
                    'class_embed_token': class_id if class_id is not None and 'variable_c' in variable_name else 0,
                    'feature_mean': mean_val,
                    'feature_std': std_val,
                    'feature_skewness': skew_val,
                    'feature_kurtosis': kurt_val,
                    'num_classes': num_classes if num_classes is not None else 0,
                    'n_rows': n_rows if n_rows is not None else 0,
                    'n_cols': n_cols if n_cols is not None else 0,
                    'linear_relaxation': linear_value,
                    'distance_to_integer': distance_to_integer,
                    'warm_start_value': warm_start_value,
                    'sample_value': sample_value,
                    'x_agg_features': x_agg_features,  # X-aggregated features
                    'var_type_encoding': var_type_encoding,  # NEW: Variable type one-hot
                    'within_type_pos': within_type_pos  # NEW: Position within type
                }


                variable_elements.append(element_info)
                index_counter += 1

    else:
        raise ValueError(f"Unexpected dimensionality ({len(shape)}) for {variable_name}.")

    return variable_elements, index_counter


# variable_a_elements = [elem for elem in variable_elements if 'variable_a' in elem['name']]




def process_variables4(variables_structure, normalized_test_data, feature_stats, linear_features=None, warm_start=None, final_solution=None):
    """
    Process variables and generate variable features.
    
    Args:
        variables_structure: Dictionary containing variable arrays
        normalized_test_data: Normalized dataset
        feature_stats: Statistical features of the dataset
        linear_features: Optional dict with keys 'variable_a', 'variable_b', 'variable_c', 'variable_d'
                        containing LP relaxation solutions
        warm_start: Optional dict with keys 'variable_a', 'variable_b', 'variable_c', 'variable_d'
                   containing warm start (CART) solutions
        final_solution: Optional dict with keys 'variable_a', 'variable_b', 'variable_c', 'variable_d'
                       containing MILP optimal solutions
    
    Returns:
        variable_features: List of processed variable features
    """
    # 1) Pull first-stage variables structure
    v_a = variables_structure['variable_a']
    v_b = variables_structure['variable_b']
    v_c = variables_structure['variable_c']
    v_d = variables_structure['variable_d']
    v_x = variables_structure['variable_x']  # Normalized data
    v_y = variables_structure['variable_y']  # One-hot encoded labels
    
    # 2) Pull second-stage variables structure
    v_z = variables_structure['variable_z']  # Predicted leaf nodes
    v_L = variables_structure['variable_L']  # Prediction cost array

    X = normalized_test_data[:, :-1]
    y = normalized_test_data[:, -1].astype(int)

    # Compute class features
    class_prior_k, class_feature_mean_k, normalized_class_id, num_classes, n_rows, n_cols = compute_class_features_from_dataset(X, y)

    # 4) Gather them as a dictionary for further processing
    variables = {
        'variable_a': v_a,
        'variable_b': v_b,
        'variable_c': v_c,
        'variable_d': v_d,
        'variable_L': v_L,
        'variable_z': v_z,
        'variable_y': v_y,
        'variable_x': v_x
    }
    
    # 6) Generate the flattened elements
    variable_features = []
    index_counter = 0

    for var_name, var_data in variables.items():
        # var_name='variable_a'
        # var_data = variables[var_name]
        linear_var = None
        if linear_features is not None and var_name in linear_features:
            linear_var = linear_features[var_name]
            
            # Reshape linear_var to match var_data shape if needed
            if linear_var.shape != var_data.shape:
                linear_var = linear_var.reshape(var_data.shape)
        
        # Get warm start value for this variable
        warm_start_var = None
        if warm_start is not None and var_name in warm_start:
            warm_start_var = warm_start[var_name]
            if warm_start_var.shape != var_data.shape:
                warm_start_var = warm_start_var.reshape(var_data.shape)
        
        # Get final (MILP) solution for this variable
        final_solution_var = None
        if final_solution is not None and var_name in final_solution:
            final_solution_var = final_solution[var_name]
            if final_solution_var.shape != var_data.shape:
                final_solution_var = final_solution_var.reshape(var_data.shape)
        
        elements, index_counter = generate_variable_elements(
            var_data, var_name, index_counter, class_feature_mean_k,
            class_prior_k=class_prior_k if var_name == 'variable_c' else None,
            normalized_class_id=normalized_class_id if var_name == 'variable_c' else None,
            feature_stats=feature_stats if var_name == 'variable_a' else None,
            linear_features=linear_var,  # Pass the reshaped linear features
            warm_start_var=warm_start_var,  # Pass the warm start values
            final_solution_var=final_solution_var,  # Pass the MILP solution
            num_classes=num_classes,
            n_rows=n_rows,
            n_cols=n_cols,
            X=X,  # NEW: Pass raw features for X-aggregated features
            y=y   # NEW: Pass class labels for X-aggregated features
        )
        variable_features.extend(elements)

    return variable_features

# ------------- TEST CODE ----------------- #

# variable_features = process_variables4(variables_structure, normalized_test_data, feature_stats, linear_features)
# len(variable_features[0])


# variable_a_elements = [elem for elem in variable_features if 'variable_a' in elem['name']]
# variable_b_elements = [elem for elem in variable_features if 'variable_b' in elem['name']]
# variable_c_elements = [elem for elem in variable_features if 'variable_c' in elem['name']]
# variable_d_elements = [elem for elem in variable_features if 'variable_d' in elem['name']]




























# -------------------- EXTRACT CONSTRAINTS DATA ----------------------- #



# ---------------------------------------------------------------------------
# 1) build  short‑symbol → {'binary' | 'continuous'}
# ---------------------------------------------------------------------------
def build_var_type_map(variable_features: List[Dict]) -> Dict[str, str]:
    rx = re.compile(r"variable_([A-Za-z]+)")      # capture letter(s) after 'variable_'
    m  = {}
    for feat in variable_features:
        sym  = rx.search(feat["name"]).group(1)   # 'a', 'b', 'c', ...
        m[sym] = "binary" if feat["is_binary"] else "continuous"
    return m


# ---------------------------------------------------------------------------
# 2) integer encoders
# ---------------------------------------------------------------------------
OP_ID = {'<': 0,
        '>': 1,
        '==': 2,
        '<=': 3,
        '>=': 4}
COMPLEXITY_ID = {"linear": 0, "quadratic": 1, "logical": 2}

def encode_var_set(var_set: Set[str]) -> int:
    if var_set == {"binary"}:
        return 0
    if var_set == {"continuous"}:
        return 1
    return 2                                   # mixed


# ---------------------------------------------------------------------------
# 3)   low‑level packer  -----------------------------------------------------
# ---------------------------------------------------------------------------
def _pack(op: str,
          complexity: str,
          symbols: List[str],          # only to derive variable‑type code
          num_vars: int,               # explicit count passed in
          has_mult: bool,
          type_map: Dict[str, str]) -> Tuple[int, int, int, int, int]:
    vtypes = {type_map.get(s, "continuous") for s in symbols}
    return (
        OP_ID[op],
        COMPLEXITY_ID[complexity],
        num_vars,
        int(has_mult),
        encode_var_set(vtypes)
    )


# ---------------------------------------------------------------------------
# 4)  constraint‑specific feature builders  ---------------------------------
#     Each receives whatever numeric sizes it needs so we can compute
#     num_variables properly.
# ---------------------------------------------------------------------------
def feat_2b(K: int, type_map):   # ½Σ_k(y+…−2yc) − L_s ≤ 1 − z_st
    return _pack("<=", "quadratic",
                 ["y", "c", "L", "z"],
                 num_vars = 2*K + 2,            # K y_sk + K c_kt + L_s + z_st
                 has_mult = True,
                 type_map = type_map)

def feat_2c(K: int, type_map):   # Σ_k c_kt = 1
    return _pack("==", "linear",
                 ["c"],
                 num_vars = K,
                 has_mult = False,
                 type_map = type_map)

def feat_2d(TL: int, type_map):  # Σ_t z_st = 1
    return _pack("==", "linear",
                 ["z"],
                 num_vars = TL,
                 has_mult = False,
                 type_map = type_map)

def feat_2e(P: int, type_map):   # aᵀx + … ≤ b_m + …(1−z_st)
    return _pack("<=", "linear",
                 ["a", "x", "b", "z"],
                 num_vars = 2*P + 2,            # P a_j + P x_j + b_m + z_st
                 has_mult = False,
                 type_map = type_map)

def feat_2f(P: int, type_map):   # aᵀx ≥ b_m − (1−z_st)
    return _pack(">=", "linear",
                 ["a", "x", "b", "z"],
                 num_vars = 2*P + 2,
                 has_mult = False,
                 type_map = type_map)

def feat_2g(P: int, type_map):   # Σ_j a_jt = d_t
    return _pack("==", "linear",
                 ["a", "d"],
                 num_vars = P + 1,              # P a_jt  +  d_t
                 has_mult = False,
                 type_map = type_map)

def feat_2h(type_map):           # b_t ≤ d_t
    return _pack("<=", "linear",
                 ["b", "d"],
                 num_vars = 2,
                 has_mult = False,
                 type_map = type_map)

def feat_2i(type_map):           # d_t ≤ d_parent
    return _pack("<=", "linear",
                 ["d"],
                 num_vars = 2,                  # d_t  +  d_parent
                 has_mult = False,
                 type_map = type_map)

# def feat_2j(type_map):           # L_s ≤ 1
#     return _pack("<=", "linear",
#                  ["L"],
#                  num_vars = 1,
#                  has_mult = False,
#                  type_map = type_map)


# ---------------------------------------------------------------------------
# 5) (unchanged) helper for binary‑tree navigation  -------------------------
# ---------------------------------------------------------------------------
def node_direct(t: int):
    idx, Ar, Al = t, [], []
    while idx != 1:
        if idx % 2:
            idx //= 2; Ar.append(idx)
        else:
            idx //= 2; Al.append(idx)
    return Al, Ar


# ---------------------------------------------------------------------------
# 6)  Helper functions for constraint tightness computation
# ---------------------------------------------------------------------------
def compute_normalized_slack(lhs: float, rhs: float, comparison_op: str) -> float:
    """
    Compute normalized slack for a constraint.
    For inequality LHS <= RHS: slack = (RHS - LHS) / max(|RHS|, |LHS|, 1)
    For inequality LHS >= RHS: slack = (LHS - RHS) / max(|RHS|, |LHS|, 1)
    For equality LHS == RHS: slack = |RHS - LHS| / max(|RHS|, |LHS|, 1)
    
    Returns value in [0, 1+] where 0 means tight, larger values mean more slack.
    """
    normalizer = max(abs(rhs), abs(lhs), 1.0)
    
    if comparison_op in ['<=', '<']:
        slack = (rhs - lhs) / normalizer
    elif comparison_op in ['>=', '>']:
        slack = (lhs - rhs) / normalizer
    else:  # equality
        slack = abs(rhs - lhs) / normalizer
    
    return max(0.0, slack)  # Ensure non-negative (violated constraints get 0)


def compute_is_tight(lhs: float, rhs: float, epsilon: float = 1e-6) -> int:
    """
    Return 1 if constraint is tight (|LHS - RHS| < epsilon), else 0.
    """
    return 1 if abs(lhs - rhs) < epsilon else 0


def get_parent_node(t: int) -> int:
    """Get parent node index in binary tree (1-indexed)."""
    return t // 2 if t > 1 else 1


def predict_leaf_from_lp(sample: np.ndarray, lp_a: np.ndarray, lp_b: np.ndarray) -> int:
    """
    Predict which leaf node a sample goes to using LP-relaxed a and b.
    Uses the decision tree structure with depth 2.
    
    Args:
        sample: Feature vector for one sample (shape: P,)
        lp_a: LP-relaxed feature selection matrix (shape: P, TD)
        lp_b: LP-relaxed threshold vector (shape: TD,)
    
    Returns:
        Leaf node index (0-indexed, 0 to 3 for depth-2 tree)
    """
    # Get feature indices from LP a (argmax gives the selected feature for each node)
    feature_indices = np.argmax(lp_a, axis=0)
    
    # Navigate tree (depth 2, 3 decision nodes, 4 leaves)
    current_node = 0
    feature_idx = feature_indices[current_node]
    split_value = float(lp_b[current_node]) if lp_b.ndim == 1 else float(lp_b[current_node, 0])
    
    if sample[feature_idx] >= split_value:
        current_node = 2  # Go right
    else:
        current_node = 1  # Go left
    
    # Second level
    feature_idx = feature_indices[current_node]
    split_value = float(lp_b[current_node]) if lp_b.ndim == 1 else float(lp_b[current_node, 0])
    
    if current_node == 1:
        if sample[feature_idx] >= split_value:
            return 1  # Leaf 1
        else:
            return 0  # Leaf 0
    else:  # current_node == 2
        if sample[feature_idx] >= split_value:
            return 3  # Leaf 3
        else:
            return 2  # Leaf 2


def compute_lp_z_matrix(x_data: np.ndarray, lp_a: np.ndarray, lp_b: np.ndarray, num_leaves: int = 4) -> np.ndarray:
    """
    Compute z_st matrix (sample-to-leaf assignment) using LP-relaxed a and b.
    
    Args:
        x_data: Sample features (shape: S, P)
        lp_a: LP feature selection (shape: P, TD)
        lp_b: LP thresholds (shape: TD,)
        num_leaves: Number of leaf nodes (default 4 for depth-2 tree)
    
    Returns:
        z_matrix: One-hot encoded leaf assignments (shape: S, TL)
    """
    S = x_data.shape[0]
    z_matrix = np.zeros((S, num_leaves))
    
    for s in range(S):
        leaf_idx = predict_leaf_from_lp(x_data[s], lp_a, lp_b)
        z_matrix[s, leaf_idx] = 1.0
    
    return z_matrix


def compute_lp_L_vector(z_matrix: np.ndarray, y_matrix: np.ndarray, lp_c: np.ndarray) -> np.ndarray:
    """
    Compute L_s (misclassification indicator) for each sample.
    
    L_s = 1 if the predicted class at the assigned leaf doesn't match true class, else 0.
    
    Args:
        z_matrix: Sample-to-leaf assignment (shape: S, TL)
        y_matrix: True class one-hot (shape: S, K)
        lp_c: LP class-to-leaf assignment (shape: K, TL)
    
    Returns:
        L_vector: Misclassification indicators (shape: S,)
    """
    S = z_matrix.shape[0]
    L_vector = np.zeros(S)
    
    for s in range(S):
        # Find which leaf sample s is assigned to
        leaf_idx = np.argmax(z_matrix[s])
        
        # Get the predicted class for this leaf (argmax of c_kt for this leaf)
        predicted_class = np.argmax(lp_c[:, leaf_idx])
        
        # Get the true class for this sample
        true_class = np.argmax(y_matrix[s])
        
        # L_s = 1 if misclassified, 0 otherwise
        L_vector[s] = 1.0 if predicted_class != true_class else 0.0
    
    return L_vector


# ---------------------------------------------------------------------------
# 6)  main collector --------------------------------------------------------
# ---------------------------------------------------------------------------
def check_constraints_structural(variables_structure: Dict[str, "np.ndarray"],
                                 variable_features: List[Dict],
                                 linear_features: Optional[Dict] = None):
    """
    Generate constraint features including structural features and tightness measures.
    
    Args:
        variables_structure: Dict with variable arrays (variable_y, variable_z, etc.)
        variable_features: List of variable feature dicts
        linear_features: Optional dict with LP-relaxed values for 'variable_a', 'variable_b', 
                        'variable_c', 'variable_d'. If None, tightness features default to 0.5/0.
    """
    # ---- sizes -----
    K   = variables_structure['variable_y'].shape[1]     # classes
    TL  = variables_structure['variable_z'].shape[1]     # leaf nodes
    TD  = len(variables_structure['variable_b'])         # decision nodes
    T   = TL + TD
    S   = len(variables_structure['variable_L'])         # samples
    P   = variables_structure['variable_x'].shape[1]     # features

    # ---- Extract LP values if available -----
    lp_a = linear_features.get('variable_a') if linear_features else None  # shape (P, TD)
    lp_b = linear_features.get('variable_b') if linear_features else None  # shape (TD,)
    lp_c = linear_features.get('variable_c') if linear_features else None  # shape (K, TL)
    lp_d = linear_features.get('variable_d') if linear_features else None  # shape (TD,)
    
    # Get data from variables_structure (always available)
    x_data = variables_structure.get('variable_x')  # shape (S, P) - always available
    y_data = variables_structure.get('variable_y')  # shape (S, K) - always available (true labels)
    
    # ---- Compute LP-based z and L if LP features available -----
    lp_z = None  # shape (S, TL) - computed from LP a, b, x
    lp_L = None  # shape (S,) - computed from z, y, c
    
    if lp_a is not None and lp_b is not None and x_data is not None:
        lp_z = compute_lp_z_matrix(x_data, lp_a, lp_b, num_leaves=TL)
        
        if lp_c is not None and y_data is not None:
            lp_L = compute_lp_L_vector(lp_z, y_data, lp_c)

    # ---- var‑type map -----
    vmap = build_var_type_map(variable_features)

    out, idx = [], 0
    
    # Default tightness values when LP data unavailable
    DEFAULT_SLACK = 0.5
    DEFAULT_TIGHT = 0

    # 2b -----------------------------------------------------------
    # (1/2)Σ_k(y_sk + c_kt - 2y_sk*c_kt) - L_s ≤ 1 - z_st
    # Now we can compute this using: y (from dataset), c (from LP), z (computed), L (computed)
    for s in range(S):
        for t in range(TL):
            f = feat_2b(K, vmap)
            
            norm_slack = DEFAULT_SLACK
            is_tight = DEFAULT_TIGHT
            
            if lp_c is not None and y_data is not None and lp_z is not None and lp_L is not None:
                # Compute LHS: (1/2)Σ_k(y_sk + c_kt - 2*y_sk*c_kt) - L_s
                y_s = y_data[s]  # shape (K,)
                c_t = lp_c[:, t]  # shape (K,)
                
                # Σ_k(y_sk + c_kt - 2*y_sk*c_kt) = Σ_k(y_sk) + Σ_k(c_kt) - 2*Σ_k(y_sk*c_kt)
                sum_term = np.sum(y_s + c_t - 2 * y_s * c_t)
                lhs = 0.5 * sum_term - lp_L[s]
                
                # Compute RHS: 1 - z_st
                rhs = 1.0 - lp_z[s, t]
                
                norm_slack = compute_normalized_slack(lhs, rhs, '<=')
                is_tight = compute_is_tight(lhs, rhs)
            
            out.append({'idx': idx, 'name': f'constraint_2b_s{s}_t{t}',
                        'comparison_op': f[0], 'complexity': f[1],
                        'num_variables': f[2], 'has_multiplication': f[3],
                        'variable_type': f[4],
                        'normalized_slack': norm_slack,
                        'is_tight': is_tight})
            idx += 1

    # 2c -----------------------------------------------------------
    # Σ_k c_kt = 1 (equality constraint)
    for t in range(TL):
        f = feat_2c(K, vmap)
        
        norm_slack = DEFAULT_SLACK
        is_tight = DEFAULT_TIGHT
        if lp_c is not None:
            lhs = float(np.sum(lp_c[:, t]))  # Σ_k c_kt
            rhs = 1.0
            norm_slack = compute_normalized_slack(lhs, rhs, '==')
            is_tight = compute_is_tight(lhs, rhs)
        
        out.append({'idx': idx, 'name': f'constraint_2c_t{t}',
                    'comparison_op': f[0], 'complexity': f[1],
                    'num_variables': f[2], 'has_multiplication': f[3],
                    'variable_type': f[4],
                    'normalized_slack': norm_slack,
                    'is_tight': is_tight})
        idx += 1

    # 2d -----------------------------------------------------------
    # Σ_t z_st = 1 (equality constraint)
    # Now we can compute this using computed lp_z
    for s in range(S):
        f = feat_2d(TL, vmap)
        
        norm_slack = DEFAULT_SLACK
        is_tight = DEFAULT_TIGHT
        
        if lp_z is not None:
            lhs = float(np.sum(lp_z[s, :]))  # Σ_t z_st
            rhs = 1.0
            norm_slack = compute_normalized_slack(lhs, rhs, '==')
            is_tight = compute_is_tight(lhs, rhs)
        
        out.append({'idx': idx, 'name': f'constraint_2d_s{s}',
                    'comparison_op': f[0], 'complexity': f[1],
                    'num_variables': f[2], 'has_multiplication': f[3],
                    'variable_type': f[4],
                    'normalized_slack': norm_slack,
                    'is_tight': is_tight})
        idx += 1

    # 2e -----------------------------------------------------------
    # a_m^T(x_s + ε - ε_min) + ε_min ≤ b_m + (1 + ε_max)(1 - z_st)
    # For normalized data x ∈ [0,1], we use ε_min = 0, ε_max = 1
    # Simplified: a_m^T * x_s ≤ b_m + 2*(1 - z_st)  (since ε terms cancel when ε_min=0)
    # LHS = a_m^T * x_s, RHS = b_m + (1 + ε_max)*(1 - z_st)
    EPSILON_MIN = 0.0
    EPSILON_MAX = 1.0
    
    for s in range(S):
        for t in range(TL, T + 1):           # global indices of leaves (and beyond)
            Al, Ar = node_direct(t)
            leaf_local = t - TL          # 0‑based leaf number
            
            for m in Al:
                f = feat_2e(P, vmap)
                inner_local = m - 1           # 0‑based inner‑node number

                norm_slack = DEFAULT_SLACK
                is_tight = DEFAULT_TIGHT
                if lp_a is not None and lp_b is not None and x_data is not None and inner_local < TD:
                    # LHS = a_m^T * (x_s + ε - ε_min) + ε_min
                    # With ε_min = 0, this simplifies to: a_m^T * x_s (assuming ε ≈ 0 for tightness check)
                    lhs = float(np.dot(lp_a[:, inner_local], x_data[s, :]))
                    
                    # RHS = b_m + (1 + ε_max)*(1 - z_st)
                    b_m = float(lp_b[inner_local]) if lp_b.ndim == 1 else float(lp_b[inner_local, 0])
                    
                    if lp_z is not None and leaf_local < TL:
                        z_st = lp_z[s, leaf_local]
                        # Big-M = (1 + ε_max) = 2 for normalized data
                        big_m = 1.0 + EPSILON_MAX
                        rhs = b_m + big_m * (1.0 - z_st)
                    else:
                        rhs = b_m
                    
                    norm_slack = compute_normalized_slack(lhs, rhs, '<=')
                    is_tight = compute_is_tight(lhs, rhs)

                out.append({
                    'idx' : idx,
                    'name': f'constraint_2e_s{s}_m{inner_local}_t{leaf_local}',
                    'comparison_op'     : f[0],
                    'complexity'        : f[1],
                    'num_variables'     : f[2],
                    'has_multiplication': f[3],
                    'variable_type'     : f[4],
                    'normalized_slack'  : norm_slack,
                    'is_tight'          : is_tight
                })
                idx += 1


    # 2f -----------------------------------------------------------
    # a_m^T * x_s ≥ b_m - (1 - z_st)
    # LHS = a_m^T * x_s, RHS = b_m - (1 - z_st)
    # Big-M coefficient is 1 (not 1e6)
    for s in range(S):
        for t in range(TL, T + 1):
            _, Ar = node_direct(t)            # A_R(t)
            leaf_local = t - TL

            for m in Ar:
                inner_local = m - 1
                f = feat_2f(P, vmap)

                norm_slack = DEFAULT_SLACK
                is_tight = DEFAULT_TIGHT
                if lp_a is not None and lp_b is not None and x_data is not None and inner_local < TD:
                    # LHS = a_m^T * x_s
                    lhs = float(np.dot(lp_a[:, inner_local], x_data[s, :]))
                    
                    # RHS = b_m - (1 - z_st)
                    b_m = float(lp_b[inner_local]) if lp_b.ndim == 1 else float(lp_b[inner_local, 0])
                    
                    if lp_z is not None and leaf_local < TL:
                        z_st = lp_z[s, leaf_local]
                        # Big-M coefficient is 1
                        rhs = b_m - (1.0 - z_st)
                    else:
                        rhs = b_m
                    
                    norm_slack = compute_normalized_slack(lhs, rhs, '>=')
                    is_tight = compute_is_tight(lhs, rhs)

                out.append({
                    'idx'  : idx,
                    'name' : f'constraint_2f_s{s}_m{inner_local}_t{leaf_local}',
                    'comparison_op'     : f[0],
                    'complexity'        : f[1],
                    'num_variables'     : f[2],
                    'has_multiplication': f[3],
                    'variable_type'     : f[4],
                    'normalized_slack'  : norm_slack,
                    'is_tight'          : is_tight
                })
                idx += 1

    # 2g -----------------------------------------------------------
    # Σ_j a_jt = d_t (equality constraint)
    for t in range(TD):
        f = feat_2g(P, vmap)
        
        norm_slack = DEFAULT_SLACK
        is_tight = DEFAULT_TIGHT
        if lp_a is not None and lp_d is not None:
            lhs = float(np.sum(lp_a[:, t]))  # Σ_j a_jt
            rhs = float(lp_d[t]) if lp_d.ndim == 1 else float(lp_d[t, 0])
            norm_slack = compute_normalized_slack(lhs, rhs, '==')
            is_tight = compute_is_tight(lhs, rhs)
        
        out.append({'idx': idx, 'name': f'constraint_2g_t{t}',
                    'comparison_op': f[0], 'complexity': f[1],
                    'num_variables': f[2], 'has_multiplication': f[3],
                    'variable_type': f[4],
                    'normalized_slack': norm_slack,
                    'is_tight': is_tight})
        idx += 1

    # 2h -----------------------------------------------------------
    # b_t ≤ d_t (inequality constraint)
    for t in range(TD):
        f = feat_2h(vmap)
        
        norm_slack = DEFAULT_SLACK
        is_tight = DEFAULT_TIGHT
        if lp_b is not None and lp_d is not None:
            lhs = float(lp_b[t]) if lp_b.ndim == 1 else float(lp_b[t, 0])
            rhs = float(lp_d[t]) if lp_d.ndim == 1 else float(lp_d[t, 0])
            norm_slack = compute_normalized_slack(lhs, rhs, '<=')
            is_tight = compute_is_tight(lhs, rhs)
        
        out.append({'idx': idx, 'name': f'constraint_2h_t{t}',
                    'comparison_op': f[0], 'complexity': f[1],
                    'num_variables': f[2], 'has_multiplication': f[3],
                    'variable_type': f[4],
                    'normalized_slack': norm_slack,
                    'is_tight': is_tight})
        idx += 1

    # 2i -----------------------------------------------------------
    # d_t ≤ d_{p(t)} (inequality constraint - depth ordering)
    for t in range(TD):
        f = feat_2i(vmap)
        
        norm_slack = DEFAULT_SLACK
        is_tight = DEFAULT_TIGHT
        if lp_d is not None:
            lhs = float(lp_d[t]) if lp_d.ndim == 1 else float(lp_d[t, 0])
            # Get parent index (1-indexed tree structure, so node t+1 has parent (t+1)//2)
            parent_idx = get_parent_node(t + 1) - 1  # Convert to 0-indexed
            if parent_idx >= 0 and parent_idx < TD:
                rhs = float(lp_d[parent_idx]) if lp_d.ndim == 1 else float(lp_d[parent_idx, 0])
                norm_slack = compute_normalized_slack(lhs, rhs, '<=')
                is_tight = compute_is_tight(lhs, rhs)
            elif t == 0:  # Root node - d_root ≤ 1
                rhs = 1.0
                norm_slack = compute_normalized_slack(lhs, rhs, '<=')
                is_tight = compute_is_tight(lhs, rhs)
        
        out.append({'idx': idx, 'name': f'constraint_2i_t{t}',
                    'comparison_op': f[0], 'complexity': f[1],
                    'num_variables': f[2], 'has_multiplication': f[3],
                    'variable_type': f[4],
                    'normalized_slack': norm_slack,
                    'is_tight': is_tight})
        idx += 1

    return out





# constraint_features = check_constraints_structural(variables_structure, variable_features)
# len(constraint_features)






# ------------- TEST CODE ----------------- #

# len(constraint_features)
# constraints_2b = [feat for feat in constraint_features if feat["name"].startswith("constraint_2b")]
# constraints_2f = [feat for feat in constraint_features if feat["name"].startswith("constraint_2b_2f")]

















# --------------------- BUILD ADJACENCY MATRIX -------------------------- #



# Pre-compile these regex patterns once at module level, so they are
# not re-compiled for every call or every variable/constraint.
var_pattern = re.compile(
    r"variable_(?P<varname>[a-zA-Z]+)"
    r"(?:_s(?P<sindex>\d+))?"
    r"(?:_k(?P<kindex>\d+))?"
    r"(?:_m(?P<mindex>\d+))?"
    r"(?:_j(?P<jindex>\d+))?"
    r"(?:_t(?P<tindex>\d+))?"
)

con_pattern = re.compile(
    r"constraint_(?P<label>2[a-z])"
    r"(?:_s(?P<sindex>\d+))?"
    r"(?:_m(?P<mindex>\d+))?"
    r"(?:_t(?P<tindex>\d+))?"
    r"(?:_k(?P<kindex>\d+))?"
)

def parse_variable_name(vname):
    match = var_pattern.match(vname)
    return match.groupdict() if match else {}

def parse_constraint_name(cname):
    match = con_pattern.match(cname)
    return match.groupdict() if match else {}

def build_adjacency_matrix(stage_vars, stage_constraints, num_threads=-1):
    """
    Build a (n+m) x (n+m) adjacency matrix. Faster version:
      1) Parse variable names once, store them in dictionaries keyed by subindices.
      2) Parse constraint names, then do dictionary lookups instead of scanning all variables.
      3) Optionally parallelize constraint processing with multiple threads.
    """
    n = len(stage_vars)         # number of variables
    m = len(stage_constraints)  # number of constraints
    N = n + m
    
    # Initialize adjacency matrix to 0
    adj_matrix = np.zeros((N, N), dtype=int)
    
    # Create index maps: 
    #   rows/cols [0..n-1] -> variables
    #   rows/cols [n..n+m-1] -> constraints
    var_idx_map = {}
    for i, var in enumerate(stage_vars):
        var_idx_map[var['index']] = i
        
    con_idx_map = {}
    for j, con in enumerate(stage_constraints):
        con_idx_map[con['idx']] = n + j
        
    # --------------------------------------------------------------
    # 1. Pre-parse variables exactly once, store them in dictionaries
    # --------------------------------------------------------------
    
    # For example, separate dictionaries for each kind of variable label 
    # so we can do quick lookups. Each dictionary is keyed by the relevant sub-indices.
    # Example for 'z_{s,t}': z_map[(s, t)] = var_index
    # Example for 'y_{s,k}': y_map[(s, k)] = var_index
    # etc.
    z_map = {}
    y_map = {}
    c_map = {}
    L_map = {}
    b_map = {}
    d_map = {}
    a_map = {}
    x_map = {}
    # If you have other variable labels, add more as needed.

    for var in stage_vars:
        v_idx  = var_idx_map[var['index']]
        vinfo  = parse_variable_name(var['name'])

        label  = vinfo.get('varname','') 
        vs_str = vinfo.get('sindex', None)
        vt_str = vinfo.get('tindex', None)
        vk_str = vinfo.get('kindex', None)
        vm_str = vinfo.get('mindex', None)
        vj_str = vinfo.get('jindex', None)
        
        # Convert to integer if not None
        vs = int(vs_str) if vs_str else None
        vt = int(vt_str) if vt_str else None
        vk = int(vk_str) if vk_str else None
        vm = int(vm_str) if vm_str else None
        vj = int(vj_str) if vj_str else None

        # Store in appropriate dictionary (adapt if your naming differs):
        if label == 'z':
            z_map[(vs, vt)] = v_idx
        elif label == 'y':
            y_map[(vs, vk)] = v_idx
        elif label == 'c':
            c_map[(vk, vt)] = v_idx
        elif label == 'L':
            # Possibly keyed by (s) alone
            L_map[vs] = v_idx
        elif label == 'b':
            # Possibly keyed by (t) alone
            b_map[vt] = v_idx
        elif label == 'd':
            # Possibly keyed by (t) alone
            d_map[vt] = v_idx
        elif label == 'a':
            # Possibly keyed by (j, t) or (j, s), depending on your usage.
            # The original example used j for 'mindex'; if so, store as:
            a_map[(vj, vt)] = v_idx
        elif label == 'x':
            x_map[vs] = v_idx
        # etc... for other labels

    # --------------------------------------------------------------
    # 2. A function that processes each constraint (or a batch of them)
    # --------------------------------------------------------------
    def process_constraints(constraints_subset):
        local_edges = []  # store edges as (vindex, cindex) pairs, to fill in later
        for con in constraints_subset:
            c_idx  = con_idx_map[con['idx']]
            cinfo  = parse_constraint_name(con['name'])
            c_label = cinfo.get('label','')
            cs_str  = cinfo.get('sindex', None)
            ct_str  = cinfo.get('tindex', None)
            ck_str  = cinfo.get('kindex', None)
            cm_str  = cinfo.get('mindex', None)

            cs = int(cs_str) if cs_str else None
            ct = int(ct_str) if ct_str else None
            ck = int(ck_str) if ck_str else None
            cm = int(cm_str) if cm_str else None

            if c_label == '2b':
                # Uses y_{s,k}, c_{k,t}, L_s, z_{s,t}
                # If a dictionary lookup fails (key not found), skip it.
                if cs is not None:
                    # y_{s,k} for all k
                    # If this constraint references *all* k in some set, you might
                    # iterate over possible k’s. But if your stage_vars only define 
                    # certain y_{s,k}, we can do partial checks. For example:
                    # look up y_map[(cs, ck)] only if ck is not None. 
                    if ck is not None and (cs, ck) in y_map:
                        local_edges.append((y_map[(cs, ck)], c_idx))

                    # c_{k,t}
                    if ck is not None and ct is not None and (ck, ct) in c_map:
                        local_edges.append((c_map[(ck, ct)], c_idx))

                    # L_s
                    if cs in L_map:
                        local_edges.append((L_map[cs], c_idx))
                    
                    # z_{s,t}
                    if ct is not None and (cs, ct) in z_map:
                        local_edges.append((z_map[(cs, ct)], c_idx))

            elif c_label == '2c':
                # sum_{k in K} c_{k,t} = 1 for each t
                # Means c_{k,ct} for all k. 
                # If you only have c_{(k, t)} from your dictionary, you can iterate.
                if ct is not None:
                    # Quick approach: find all (k, ct) in c_map. 
                    # Because c_map is keyed by (k, t), we can do:
                    #   for (k_, t_) in c_map if t_ == ct: ...
                    # but that’s not super-efficient if many c_ variables exist. 
                    # For large K, store a separate dict-of-dicts: c_map[t][k] → v_idx.
                    # For clarity, we do a direct iteration below:
                    for (k_, t_) in c_map:
                        if t_ == ct:
                            local_edges.append((c_map[(k_, t_)], c_idx))

            elif c_label == '2d':
                # sum_{t in T_L} z_{s,t} = 1 for each s
                # So for the given s, all z_{s, t} appear. 
                if cs is not None:
                    for (s_, t_) in z_map:
                        if s_ == cs:
                            local_edges.append((z_map[(s_, t_)], c_idx))

            elif c_label == '2e':
                # a^T_m x_{s} + ε_min <= b_m + (1 - z_{s,t})
                # This involves a_{j,t}, x_{s}, b_{t}, z_{s,t}.
                # Now we only check the column t for a(j,t):
                if ct is not None:
                    # For every (j_, t_) in a_map, if t_ == ct, we add an edge.
                    for (j_, t_) in a_map:
                        if t_ == ct:
                            local_edges.append((a_map[(j_, t_)], c_idx))
                if cs is not None and cs in x_map:
                    local_edges.append((x_map[cs], c_idx))
                if ct is not None and ct in b_map:
                    local_edges.append((b_map[ct], c_idx))
                if cs is not None and ct is not None and (cs, ct) in z_map:
                    local_edges.append((z_map[(cs, ct)], c_idx))

            elif c_label == '2f':
                # a^T_m x_{s} >= b_m - (1 - z_{s,t})
                # Similarly involves a_{j,t}, x_{s}, b_{t}, z_{s,t}.
                # Again, we only check the column t for a(j,t):
                if ct is not None:
                    for (j_, t_) in a_map:
                        if t_ == ct:
                            local_edges.append((a_map[(j_, t_)], c_idx))
                if cs is not None and cs in x_map:
                    local_edges.append((x_map[cs], c_idx))
                if ct is not None and ct in b_map:
                    local_edges.append((b_map[ct], c_idx))
                if cs is not None and ct is not None and (cs, ct) in z_map:
                    local_edges.append((z_map[(cs, ct)], c_idx))

            elif c_label == '2g':
                # sum_{j=1..P} a_{j,t} = d_t
                # => a_{j, t}, d_t
                if ct is not None and ct in d_map:
                    local_edges.append((d_map[ct], c_idx))
                # Because we store a_map keyed by (j, t), 
                # you could iterate over all j for which (j, ct) is in a_map:
                for (j_, t_) in a_map:
                    if t_ == ct:
                        local_edges.append((a_map[(j_, t_)], c_idx))

            elif c_label == '2h':
                # 0 <= b_t <= d_t => both b_{t} and d_{t} appear
                if ct is not None and ct in b_map:
                    local_edges.append((b_map[ct], c_idx))
                if ct is not None and ct in d_map:
                    local_edges.append((d_map[ct], c_idx))

            elif c_label == '2i':
                # d_t <= d_p(t) or something similar => d_t
                if ct is not None and ct in d_map:
                    local_edges.append((d_map[ct], c_idx))

            # elif c_label == '2j':
            #     # 0 <= L_s <= 1 => L_s
            #     if cs is not None and cs in L_map:
            #         local_edges.append((L_map[cs], c_idx))

        return local_edges

    # --------------------------------------------------------------
    # 3. Optionally process constraints in parallel
    # --------------------------------------------------------------
    # If num_threads=1, this just runs in a single thread.
    # If you have many constraints, you can increase num_threads
    # to parallelize the lookups. But you must collect the edges
    # and fill the adjacency matrix carefully to avoid race conditions.
    # One approach is to gather all edges in a list of lists, then
    # write them to adj_matrix at the end.
    
    # Simple chunking of constraints
    if num_threads <= 1:
        edges_all = process_constraints(stage_constraints)
    else:
        chunk_size = max(1, len(stage_constraints)//num_threads)
        futures = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(stage_constraints), chunk_size):
                subset = stage_constraints[i:i+chunk_size]
                futures.append(executor.submit(process_constraints, subset))
        
        edges_all = []
        for f in futures:
            edges_all.extend(f.result())

    # Now fill the adjacency matrix for all edges
    for v_idx, c_idx in edges_all:
        adj_matrix[v_idx, c_idx] = 1

    # Fill diagonal with 1
    np.fill_diagonal(adj_matrix, 1)
    
    return adj_matrix




# ----------------------- TEST CODE --------------------------- #

# stage_vars=first_stage_vars
# stage_constraints=first_stage_constraints
# indices = connections[connections[:, 0] == 0]
# real_indices = indices[1:10, 1] - len(stage_vars) 
# filtered_indices = set(real_indices)

# filtered_constraints = [constraint for constraint in stage_constraints if constraint['idx'] in filtered_indices]
# filtered_constraints



# variable_y = [feat for feat in variable_features if feat["name"].startswith("variable_y")]
# len(variable_y)













# stage_vars  = [var for var in variable_features if var['name'].startswith(('variable_a', 'variable_b', 'variable_c', 'variable_d', 'variable_x', 'variable_y'))]
# stage_constraints = [con for con in constraint_features if any(stage in con['name'] for stage in ['2b', '2c', '2e', '2f', '2g', '2h', '2i'])]
# stage_vars = first_stage_vars
def extract_features(variable_features, variable_shapes, constraint_features):
    def process_stage(stage_vars, stage_constraints, stage_type):
        """
        Internal function to process a single stage's variables and constraints.
        """
        # Populate adjacency matrix
        adj_matrix = build_adjacency_matrix(stage_vars, stage_constraints, num_threads=available_cpus) 
        rows, cols = np.where(adj_matrix == 1)
        connections = np.array(pd.DataFrame(np.stack((rows, cols), axis=1), columns=["Variable_Index", "Constraint_Index"]))
        edge_features = np.ones(len(connections), dtype=int)


        # All possible constraint prefixes you care about
        constraint_groups = [
            "constraint_2b", "constraint_2c", "constraint_2d", "constraint_2e",
            "constraint_2f", "constraint_2g", "constraint_2h", "constraint_2i"
        ]

        # Initialize counters for each prefix
        group_counts = {prefix: 0 for prefix in constraint_groups}

        # Regex to extract the main constraint group from names like "constraint_2f_s12_m0_t2"
        constraint_pattern = re.compile(r"(constraint_2[a-z])")

        # Populate the counts
        for cdict in stage_constraints:
            full_name = cdict["name"]  # e.g., "constraint_2f_s12_m0_t2"
            
            match = constraint_pattern.match(full_name)
            if match:
                prefix_part = match.group(1)  # Extract "constraint_2f"
                if prefix_part in group_counts:
                    group_counts[prefix_part] += 1

        # Collect the counts into a list (as strings)
        #constraints_count_list = [f"{prefix}: {group_counts[prefix]}" for prefix in constraint_groups]

        # Convert counts into a numerical list
        constraint_shapes = [group_counts[prefix] for prefix in constraint_groups]


        #integer_indexes = extract_indexes(stage_vars, 'integer')
        #binary_indexes = extract_indexes(stage_vars, 'binary')
        # Add if() conditional so it returns the corresponding values to the stage being analyzed
        # stage_type = 'first_stage'
        first_stage_variable_indices  = ([var['index'] for var in variable_features if var['name'].startswith(('variable_a', 'variable_b', 'variable_c', 'variable_d'))] if stage_type == 'first_stage' else [])
        second_stage_variable_indices  = ([var['index'] for var in variable_features if var['name'].startswith(('variable_z'))] if stage_type == 'second_stage' else [])
        first_stage_constraint_indices  = ([cons['idx'] for cons in constraint_features if cons['name'].startswith(('constraint_2c', 'constraint_2g', 'constraint_2h', 'constraint_2i'))] if stage_type == 'first_stage' else [])
        second_stage_constraint_indices  = ([cons['idx'] for cons in constraint_features if cons['name'].startswith(('constraint_2d'))] if stage_type == 'second_stage' else [])
        shared_constraint_indices  = [cons['idx'] for cons in constraint_features if cons['name'].startswith(('constraint_2b', 'constraint_2e', 'constraint_2f'))]

        # Extract solution values, lower bounds, and upper bounds
        def extract_values(var_features, key):
            return np.array([var[key] for var in var_features])

        #solution_values = extract_values(stage_vars, 'solution_value')
        lower_bounds = extract_values(stage_vars, 'lower_bound')
        upper_bounds = extract_values(stage_vars, 'upper_bound')


        return {
            'variable_features': stage_vars,
            'constraint_features': stage_constraints,
            #'variables': variables,
            'variable_shapes': variable_shapes,
            'constraint_shapes': constraint_shapes,
            'edge_indices': connections,
            'edge_features': edge_features,
            #'variable_nodes': variable_nodes,
            #'all_integer_variable_indices': integer_indexes,
            #'binary_variable_indices': binary_indexes,
            'first_stage_variable_indices': first_stage_variable_indices,
            'second_stage_variable_indices': second_stage_variable_indices,
            'first_stage_constraint_indices': first_stage_constraint_indices,
            'second_stage_constraint_indices': second_stage_constraint_indices,
            'shared_constraint_indices': shared_constraint_indices,
            'model_maximize': False,
            #'best_solution_labels': solution_values,
            'variable_lbs': lower_bounds,
            'variable_ups': upper_bounds
        }

    # Filter variable features into first-stage and second-stage
    first_stage_vars = [var for var in variable_features if var['name'].startswith(('variable_a', 'variable_b', 'variable_c', 'variable_d', 'variable_x', 'variable_y'))]
    second_stage_vars = [var for var in variable_features if var['name'].startswith(('variable_z'))] 

    # Reset indexes for variables
    for new_idx, var in enumerate(first_stage_vars):
        var['index'] = new_idx  # Restart index for first-stage vars

    for new_idx, var in enumerate(second_stage_vars):
        var['index'] = new_idx 


    # Filter constraint features into first-stage and second-stage
    first_stage_constraints = [con for con in constraint_features if any(stage in con['name'] for stage in ['2b', '2c', '2e', '2f', '2g', '2h', '2i'])]
    second_stage_constraints = [con for con in constraint_features if any(stage in con['name'] for stage in ['2b', '2d', '2e', '2f'])]

    # Reset indexes for constraints
    for new_idx, con in enumerate(first_stage_constraints):
        con['idx'] = new_idx  

    for new_idx, con in enumerate(second_stage_constraints):
        con['idx'] = new_idx 



    # Process first-stage and second-stage data using the internal function
    # stage_vars = first_stage_vars
    # stage_constraints = first_stage_constraints
    first_stage_data = process_stage(first_stage_vars, first_stage_constraints, stage_type='first_stage')
    second_stage_data = process_stage(second_stage_vars, second_stage_constraints,  stage_type='second_stage')

    # Return all extracted information
    return {
        'first_stage': first_stage_data,
        'second_stage': second_stage_data
    }


# extracted_features = extract_features(variable_features, variable_shapes, constraint_features)

# first_extracted_features = extracted_features['first_stage']['first_stage_variable_indices']
# variable_y = [feat for feat in variable_features if feat["name"].startswith("variable_y")]
# len(variable_y)




# stage_data = extracted_features['first_stage']

# ---------------- CREATE GRAPH REPRESENTATION DATA OF ALL FEATURES -----------------#

def create_graph_representation_two_stage(extracted_features):
    # Basis status and equality bound mappings

    def handle_infinity(value):
        """Handle infinity values in bounds."""
        if value == float('inf'):
            return 1e10  # Large positive number
        elif value == -float('inf'):
            return -1e10  # Large negative number
        return value

    def process_stage_data(stage_data, stage_type):
        """Process features, constraints, and other stage-specific data."""
        # Process variable features
        variable_features = []
        # Positional counter for second-stage (z) nodes
        z_pos_counter = 0
        # entry = stage_data['variable_features'][85]
        for entry in stage_data['variable_features']:
            index = entry['index']
            is_binary = 1 if entry['is_binary'] else 0
            has_lower_bound = 1 if entry['lower_bound'] is not None else 0
            has_upper_bound = 1 if entry['upper_bound'] is not None else 0
            lower_bound = handle_infinity(float(entry['lower_bound']) if entry['lower_bound'] is not None else 0.0)
            upper_bound = handle_infinity(float(entry['upper_bound']) if entry['upper_bound'] is not None else 0.0)
            node_type_split = entry['node_type_split']
            node_type_threshold = entry['node_type_threshold']
            node_type_leaf = entry['node_type_leaf']
            node_type_activation = entry['node_type_activation']
            node_type_prediction = entry['node_type_prediction']
            node_type_dataset = entry['node_type_dataset']
            node_type_label = entry['node_type_label']
            node_type_loss = entry['node_type_loss']
            depth_level = entry['depth_level']
            is_root_node = entry['is_root_node']
            leaf_node = entry['leaf_node']
            is_left_subtree = entry['is_left_subtree']
            is_right_subtree = entry['is_right_subtree']
            decision_node_id = entry['decision_node_id']
            feature_id = entry['feature_id']
            class_prior_k = entry['class_prior_k']
            class_feature_mean_k= entry['class_feature_mean_k']
            normalized_class_id= entry['normalized_class_id']
            class_embed_token= entry['class_embed_token']
            feature_mean = entry['feature_mean']
            feature_std = entry['feature_std']
            feature_skewness= entry['feature_skewness']
            feature_kurtosis= entry['feature_kurtosis']
            num_classes = entry['num_classes']
            n_cols = entry['n_cols']
            n_rows = entry['n_rows']
            # COMMENTED OUT: Removing LP/warm-start features to focus on X-aggregated features
            # linear_relaxation = entry['linear_relaxation']
            # distance_to_integer = entry['distance_to_integer']
            # warm_start_value = entry['warm_start_value']
            sample_value = entry.get('sample_value', 0.0)  # New feature: actual sample value for x, y, z
            x_agg_features = entry.get('x_agg_features', [0.0] * X_AGG_DIM)  # X-aggregated features
            # NEW: Variable type encoding features
            var_type_encoding = entry.get('var_type_encoding', [0.0, 0.0, 0.0, 0.0])  # [is_a, is_b, is_c, is_d]
            within_type_pos = entry.get('within_type_pos', 0.0)  # Normalized position within type

            # solution_is_at_lower_bound = 1 if solution_value == lower_bound else 0
            # solution_is_at_upper_bound = 1 if solution_value == upper_bound else 0

            # Build small positional features for z-nodes only (else zeros)
            if stage_type == 'second_stage':
                # small numeric positional signal: [idx, idx/10, idx/100, idx/1000]
                pos_feats = [float(z_pos_counter), float(z_pos_counter) / 10.0, float(z_pos_counter) / 100.0, float(z_pos_counter) / 1000.0]
                z_pos_counter += 1
            else:
                pos_feats = [0.0] * Z_POS_DIM

            variable_features.append([
                # index,
                # is_binary,
                #solution_value,
                # has_lower_bound,
                # has_upper_bound,
                # lower_bound,
                # upper_bound,
                node_type_split,
                # node_type_threshold,
                node_type_leaf,
                # node_type_activation,
                node_type_prediction,
                # node_type_dataset,
                # node_type_label,
                # node_type_loss,
                depth_level,
                is_root_node,
                leaf_node, #
                is_left_subtree,
                is_right_subtree,
                decision_node_id, #
                feature_id, #
                class_prior_k,
                class_feature_mean_k,
                normalized_class_id,
                class_embed_token,
                feature_mean,
                feature_std,
                feature_skewness,
                feature_kurtosis,
                num_classes,
                n_cols,
                n_rows,
                # COMMENTED OUT: LP/warm-start features removed (were indices 21-23)
                # linear_relaxation,
                # distance_to_integer,
                # warm_start_value,
                sample_value, # New feature: actual dataset value for x_{s,p}, y_{s,k} (now index 21)
                ] + pos_feats + x_agg_features + var_type_encoding + [within_type_pos])  # MODIFIED: append var_type features (5 features)

        # Process constraint features
        constraint_features = []
        #entry = stage_data['constraint_features'][1]
        for entry in stage_data['constraint_features']:
            try:
                num_variables = handle_infinity(entry['num_variables'] if entry['num_variables'] is not None else 0.0)
                complexity = handle_infinity(entry['complexity'] if entry['complexity'] is not None else 3)
                comparison_op = entry['variable_type'] if entry['variable_type'] is not None else 5
                has_multiplication = entry['has_multiplication'] if entry['has_multiplication'] is not None else 2
                variables_type =  entry['variable_type'] if entry['variable_type'] is not None else 3
                # New tightness features
                normalized_slack = entry.get('normalized_slack', 0.5)
                is_tight = entry.get('is_tight', 0)

                constraint_features.append([
                    comparison_op,
                    complexity,
                    num_variables,
                    has_multiplication,
                    variables_type,
                    normalized_slack,
                    is_tight
                ])
            except Exception as e:
                print(f"Error processing constraint feature: {e}")
                break

        # Extract model maximize
        model_maximize = 1 if extracted_features.get('model_maximize', False) else 0

        # Process other stage-specific data
        edge_indices = stage_data['edge_indices']
        edge_features = stage_data['edge_features']
        #integer_variable_indices = stage_data['all_integer_variable_indices']
       # binary_variable_indices = stage_data['binary_variable_indices']
        model_maximize_tensor = torch.tensor(model_maximize, dtype=torch.int32)
        #best_solution_labels = stage_data['best_solution_labels']
        variable_lbs = stage_data['variable_lbs']
        variable_ups = stage_data['variable_ups']
        first_stage_variable_indices = (stage_data['first_stage_variable_indices'] if stage_type == 'first_stage' else [])
        second_stage_variable_indices = (stage_data['second_stage_variable_indices'] if stage_type == 'second_stage' else [])
        first_stage_constraint_indices = (stage_data['first_stage_constraint_indices'] if stage_type == 'first_stage' else [])
        second_stage_constraint_indices = (stage_data['second_stage_constraint_indices'] if stage_type == 'second_stage' else [])
        shared_constraint_indices = stage_data['shared_constraint_indices']
        variable_shapes = stage_data['variable_shapes']
        constraint_shapes = stage_data['constraint_shapes']


        # Convert data to PyTorch tensors
        return {
            'variable_features': torch.tensor(variable_features, dtype=torch.float32),
            'constraint_features': torch.tensor(constraint_features, dtype=torch.float32),
            #'variables': variables,
            'variable_shapes': variable_shapes,
            'constraint_shapes': torch.tensor(constraint_shapes,  dtype=torch.int32),
            'edge_indices': torch.tensor(edge_indices, dtype=torch.int32),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32),
            #'all_integer_variable_indices': torch.tensor(integer_variable_indices, dtype=torch.int32),
            #'binary_variable_indices': torch.tensor(binary_variable_indices, dtype=torch.int32),
            'first_stage_variable_indices': torch.tensor(first_stage_variable_indices, dtype=torch.int32),
            'second_stage_variable_indices':  torch.tensor(second_stage_variable_indices,  dtype=torch.int32),
            'first_stage_constraint_indices': torch.tensor(first_stage_constraint_indices, dtype=torch.int32),
            'second_stage_constraint_indices': torch.tensor(second_stage_constraint_indices, dtype=torch.int32),
            'shared_constraint_indices': torch.tensor(shared_constraint_indices, dtype=torch.int32),
            'model_maximize': model_maximize_tensor,
            #'best_solution_labels': torch.tensor(best_solution_labels, dtype=torch.float32),
            'variable_lbs': torch.tensor([handle_infinity(x) for x in variable_lbs], dtype=torch.float32),
            'variable_ups': torch.tensor([handle_infinity(x) for x in variable_ups], dtype=torch.float32)

        }

    # Process first-stage and second-stage data
    # stage_data = extracted_features['first_stage']
    first_stage = process_stage_data(extracted_features['first_stage'], stage_type='first_stage')
    second_stage = process_stage_data(extracted_features['second_stage'], stage_type='second_stage')

    return {
        'first_stage': first_stage,
        'second_stage': second_stage
    }


# extracted_features = extract_features(variable_features, variable_shapes, constraint_features)
# state = create_graph_representation_two_stage(extracted_features)
# state['first_stage']['variable_features'][21:25]