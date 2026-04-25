# GPS transformer network with Explicit Tree Routing
# Simplified version: Removes complex decoder, uses explicit tree routing for Z and L
#
# Key changes from two_stage_gps_skipped5.py:
# - decoder_forward() uses explicit tree routing with Straight-Through Estimators (STE)
# - STE eliminates train-test discrepancy: hard decisions in forward, soft gradients in backward
# - Z computed directly from A, B, X using HARD routing decisions (via STE)
# - L computed from Z-C-Y alignment (misclassification count)
# - Removes second-stage GPS layers, constraint merging, graph building
#
# STE Application:
# - A (feature selection): one-hot via straight_through_softmax
# - B (thresholds): continuous, no STE
# - C (class assignment): one-hot via straight_through_softmax
# - D (node activation): binary via straight_through_sigmoid
# - Z (leaf assignment): one-hot (computed from hard routing)
# - L (loss): continuous (computed from hard Z and C)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GCNConv, GATConv
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from two_stage_load_features2 import Z_POS_DIM
import math
import logging


# ------------------------------------------------------- #
# ---------- STRAIGHT-THROUGH ESTIMATOR (STE) ----------- #
# ------------------------------------------------------- #
# STE enables hard (discrete) decisions in the forward pass
# while allowing gradients to flow through soft approximations
# in the backward pass. This eliminates train-test discrepancy.

# Global counter for minimal diagnostic logging
_STE_DIAG_COUNTER = 0
_STE_DIAG_INTERVAL = 500  # Log every N forward passes (minimal overhead)


def _log_ste_routing_summary(z_logits: torch.Tensor, b_values: torch.Tensor):
    """Log minimal STE routing diagnostics periodically."""
    global _STE_DIAG_COUNTER
    _STE_DIAG_COUNTER += 1

    if _STE_DIAG_COUNTER % _STE_DIAG_INTERVAL != 1:
        return

    with torch.no_grad():
        # Check leaf distribution (imbalance causes gradient issues)
        z_pred = z_logits.argmax(dim=1)
        leaf_counts = [(z_pred == i).sum().item() for i in range(4)]
        total = sum(leaf_counts)
        max_pct = 100.0 * max(leaf_counts) / total if total > 0 else 0
        logging.info(
            f"[STE] leaf_dist={leaf_counts} b=[{b_values.min():.3f},{b_values.max():.3f}] "
            f"imbalance={max_pct:.0f}%"
        )



def straight_through_softmax(logits: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Straight-Through Estimator for softmax.

    Forward: Returns hard one-hot at argmax position
    Backward: Gradients flow through soft softmax
    """
    soft = F.softmax(logits, dim=dim)
    hard = torch.zeros_like(soft)
    indices = soft.argmax(dim=dim, keepdim=True)
    hard.scatter_(dim, indices, 1.0)
    # STE trick: hard - soft.detach() + soft
    return hard - soft.detach() + soft


def straight_through_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for sigmoid (binary decision).

    Forward: Returns hard 0/1 at threshold 0.5
    Backward: Gradients flow through soft sigmoid
    """
    soft = torch.sigmoid(x)
    hard = (soft > 0.5).float()
    return hard - soft.detach() + soft


def temperature_softmax(logits: torch.Tensor, temperature: float = 1.0, dim: int = 0) -> torch.Tensor:
    """
    Temperature-scaled softmax for categorical variables (A, C).
    High temp -> soft/uniform. Low temp -> sharp/one-hot. temp->0 -> argmax.
    Replaces STE: continuous gradients flow naturally through softmax.
    """
    return F.softmax(logits / max(temperature, 1e-8), dim=dim)


def temperature_sigmoid(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Temperature-scaled sigmoid for binary routing decisions.
    High temp -> soft/0.5. Low temp -> sharp/step function. temp->0 -> hard threshold.
    Replaces STE: continuous gradients flow naturally through sigmoid.
    """
    return torch.sigmoid(x / max(temperature, 1e-8))


# ------------------------------------------------------- #
# ------------------- FUNCTIONS ------------------------- #
# ------------------------------------------------------- #

def get_adjacency_matrix(graph):
    upper = torch.stack([graph.senders, graph.receivers], dim=1)
    lower = torch.stack([graph.receivers, graph.senders], dim=1)
    indices = torch.cat([upper, lower], dim=0).long()
    values = torch.cat([graph.edge_attr.squeeze(), graph.edge_attr.squeeze()], dim=0)

    num_nodes = graph.x.size(0)
    return SparseTensor(row=indices[:, 0], col=indices[:, 1], value=values, sparse_sizes=(num_nodes, num_nodes))


def get_model(**params):
    return NewTwoStageGCN(**params)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


torch.autograd.set_detect_anomaly(True)


def convert_adjacency_for_gps(adj_matrix):
    """
    Convert SparseTensor adjacency to edge_index format for GPS
    GPS expects edge_index in COO format [2, num_edges]
    """
    row, col, _ = adj_matrix.coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


# ------------------------------------------------------- #
# -------------------- CLASSES -------------------------- #
# ------------------------------------------------------- #


class GPSLayer(nn.Module):
    """
    GPS Layer with ResNet-style skip connections (Transformer-style)
    Combines local message passing with global attention + residual connection
    """
    def __init__(self,
                 channels: int,
                 conv_type: str = 'GCN',
                 heads: int = 4,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 norm: str = 'batch',
                 use_skip: bool = True):
        super(GPSLayer, self).__init__()

        self.use_skip = use_skip
        self.dropout_rate = dropout

        # Diagnostics
        self.last_attention_weights = None
        self.capture_attention = False
        self._attn_hook_handle = None

        # Choose local message passing layer
        if conv_type == 'GCN':
            self.local_conv = GCNConv(channels, channels, add_self_loops=False)
        elif conv_type == 'GAT':
            self.local_conv = GATConv(channels, channels // heads, heads=heads,
                                     concat=True, add_self_loops=False)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        # GPS Conv layer
        norm_arg = 'batch_norm' if norm == 'batch' else (None if norm == 'none' else norm)
        self.gps = GPSConv(
            channels=channels,
            conv=self.local_conv,
            heads=heads,
            dropout=dropout,
            act=act,
            norm=norm_arg,
            attn_type='multihead',
            attn_kwargs=None
        )

        self.skip_dropout = nn.Dropout(dropout) if use_skip else None
        self.layer_norm = nn.LayerNorm(channels) if use_skip else None

    def forward(self, x, edge_index, batch=None):
        if self.use_skip:
            residual = x
            x = torch.clamp(x, min=-1e4, max=1e4)
            out = self.gps(x, edge_index, batch=batch)

            if torch.isnan(out).any():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

            if self.skip_dropout is not None:
                out = self.skip_dropout(out)

            out = out + residual

            if self.layer_norm is not None:
                out = self.layer_norm(out)

            return out
        else:
            x = torch.clamp(x, min=-1e4, max=1e4)
            return self.gps(x, edge_index, batch=batch)

    def enable_attention_capture(self):
        self.capture_attention = True
        self.last_attention_weights = None
        if hasattr(self.gps, 'attn') and self.gps.attn is not None:
            if isinstance(self.gps.attn, torch.nn.MultiheadAttention):
                self._attn_hook_handle = self.gps.attn.register_forward_hook(self._attention_hook)

    def disable_attention_capture(self):
        self.capture_attention = False
        self.last_attention_weights = None
        if self._attn_hook_handle is not None:
            self._attn_hook_handle.remove()
            self._attn_hook_handle = None

    def _attention_hook(self, module, input, output):
        if self.capture_attention:
            if isinstance(output, tuple) and len(output) >= 2:
                self.last_attention_weights = output[1].detach() if output[1] is not None else None
            else:
                self.last_attention_weights = None


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_outputs):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs)
        )

    def forward(self, x):
        return self.mlp(x)


class PositionalEncodingLayer(nn.Module):
    """
    Joint Position-Type Encoding Layer.
    Combines tree positional features with variable type features.
    """
    def __init__(self, pos_dim: int = 5, var_type_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        combined_dim = pos_dim + var_type_dim
        output_dim = combined_dim

        self.pos_enc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.pos_dim = pos_dim
        self.var_type_dim = var_type_dim

    def forward(self, pos_features, var_type_features):
        combined = torch.cat([pos_features, var_type_features], dim=1)
        return self.pos_enc(combined)


class GroupedFeatureAttention(nn.Module):
    """
    Learnable attention over feature groups.
    """
    def __init__(self, feature_dim: int = 38, num_groups: int = 9):
        super().__init__()

        self.groups = {
            'node_types': [0, 1, 2],
            'tree_position': [3, 4, 5, 6, 7, 8, 9],
            'class_stats': [10, 11, 12, 13],
            'feature_stats': [14, 15, 16, 17],
            'dataset_meta': [18, 19, 20],
            'sample_data': [21],
            'positional': [22, 23, 24, 25],
            'x_aggregated': [26, 27, 28, 29, 30, 31, 32],
            'variable_type': [33, 34, 35, 36, 37]
        }

        self.group_logits = nn.Parameter(torch.zeros(num_groups))
        self.register_buffer('group_mask', self._build_group_mask(feature_dim))

    def _build_group_mask(self, feature_dim):
        mask = torch.zeros(len(self.groups), feature_dim)
        for i, (name, indices) in enumerate(self.groups.items()):
            for idx in indices:
                if idx < feature_dim:
                    mask[i, idx] = 1.0
        return mask

    def forward(self, x):
        group_weights = F.softmax(self.group_logits, dim=0)
        feature_weights = torch.matmul(group_weights, self.group_mask)
        return x * feature_weights.unsqueeze(0)

    def get_group_weights(self):
        with torch.no_grad():
            weights = F.softmax(self.group_logits, dim=0)
            return {name: weights[i].item()
                    for i, name in enumerate(self.groups.keys())}


# --------------------------------------------------------------- #
# ------------------ MAIN MODEL CLASS -------------------------- #
# --------------------------------------------------------------- #

class NewTwoStageGCN(nn.Module):
    """
    Simplified GPS model with explicit tree routing.

    Key changes:
    - decoder_forward() uses explicit tree routing (path probabilities from A, B, X)
    - L computed from Z-C-Y alignment (misclassification)
    - No second-stage GPS layers needed
    """
    def __init__(self, n_layers, node_model_hidden_sizes,
                 output_model_hidden_sizes, dropout=0.0,
                 gps_heads=4, conv_type='GCN'):
        super(NewTwoStageGCN, self).__init__()

        # Temperature for routing decisions (managed by training loop, not a learned parameter)
        # High temp = soft predictions (good gradients), low temp = near-hard (tree-like)
        self.routing_temperature = 1.0

        self.n_layers = n_layers
        self.dropout = dropout
        self.debug = False
        self._debug_prefix = "[Z-DEBUG]"

        self._layer_embeddings = {}
        self._capture_diagnostics = False

        self.log_loss_weights = nn.Parameter(
            torch.tensor([math.log(10.0), math.log(5.0), math.log(3.0)])
        )

        hidden_dim = node_model_hidden_sizes[1]

        X_AGG_DIM = 7
        VAR_TYPE_DIM = 5
        self.input_feature_dim = int(node_model_hidden_sizes[0]) + int(Z_POS_DIM) + X_AGG_DIM + VAR_TYPE_DIM

        # Joint Position-Type Encoding
        self.pos_encoding = PositionalEncodingLayer(pos_dim=5, var_type_dim=VAR_TYPE_DIM, hidden_dim=64)

        # Grouped Feature Attention
        self.feature_attention = GroupedFeatureAttention(feature_dim=self.input_feature_dim, num_groups=9)

        # First-stage preprocessing
        self.first_linear = nn.Sequential(
            nn.Linear(self.input_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Constraint embedding
        self.constraint_linear = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # GPS layers for first stage (encoder)
        self.first_layers = nn.ModuleList([
            GPSLayer(channels=hidden_dim,
                    conv_type=conv_type,
                    heads=gps_heads,
                    dropout=dropout,
                    act='relu',
                    norm='batch',
                    use_skip=True)
            for _ in range(n_layers)
        ])

        # Output heads for A, B, C, D (reused from encoder)
        self.a_head = self._make_output_head(output_model_hidden_sizes)
        self.b_head = self._make_output_head(output_model_hidden_sizes)
        self.c_head = self._make_output_head(output_model_hidden_sizes)
        self.d_head = self._make_output_head(output_model_hidden_sizes)

        # NOTE: Removed second_layers, second_linear, mlp_logits_one, etc.
        # These are no longer needed with explicit tree routing

    def _make_output_head(self, sizes):
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.Linear(sizes[-1], 1))
        return nn.Sequential(*layers)

    def enable_diagnostics(self):
        self._capture_diagnostics = True
        self._layer_embeddings = {}
        for layer in self.first_layers:
            layer.enable_attention_capture()

    def disable_diagnostics(self):
        self._capture_diagnostics = False
        self._layer_embeddings = {}
        for layer in self.first_layers:
            layer.disable_attention_capture()

    def get_layer_embeddings(self) -> dict:
        return self._layer_embeddings.copy()

    def get_attention_weights(self) -> dict:
        """
        Return attention weights from all GPS layers.
        Note: Attention weights may be None if GPSConv doesn't expose them.
        """
        weights = {}
        for i, layer in enumerate(self.first_layers):
            if layer.last_attention_weights is not None:
                weights[f'first_layer_{i}'] = layer.last_attention_weights
        return weights

    def encode_first_graph(self, graph, is_training: bool, first_stage_variable_indices=None) -> torch.Tensor:
        """
        Encode first graph using GPS layers with separate variable/constraint pathways.
        """
        x = graph.x

        if first_stage_variable_indices is not None:
            if hasattr(graph, 'ptr') and graph.ptr is not None:
                n_vars = first_stage_variable_indices[0][-1].item() + 1 if len(first_stage_variable_indices) > 0 else x.size(0)
            else:
                n_vars = first_stage_variable_indices[-1].item() + 1 if first_stage_variable_indices.numel() > 0 else x.size(0)
        else:
            n_vars = x.size(0)

        x_vars = x[:n_vars]
        x_cons = x[n_vars:]

        # Variable pathway
        pos_indexes = [3, 4, 5, 6, 7]
        var_type_indexes = [33, 34, 35, 36, 37]

        pos_feats = x_vars[:, pos_indexes]
        var_type_feats = x_vars[:, var_type_indexes]

        joint_encoded = self.pos_encoding(pos_feats, var_type_feats)

        x_vars_updated = x_vars.clone()
        x_vars_updated[:, pos_indexes] = joint_encoded[:, :5]
        x_vars_updated[:, var_type_indexes] = joint_encoded[:, 5:]

        x_vars_weighted = self.feature_attention(x_vars_updated)
        x_vars_projected = self.first_linear(x_vars_weighted)

        # Constraint pathway
        if x_cons.size(0) > 0:
            x_cons_real = x_cons[:, :7]
            x_cons_projected = self.constraint_linear(x_cons_real)
        else:
            x_cons_projected = x_vars_projected.new_zeros((0, x_vars_projected.size(1)))

        nodes = torch.cat([x_vars_projected, x_cons_projected], dim=0)

        batch = graph.batch if hasattr(graph, 'batch') else None
        edge_index = graph.edge_index
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        for layer_idx, layer in enumerate(self.first_layers):
            nodes = layer(nodes, edge_index, batch=batch)
            if self._capture_diagnostics:
                self._layer_embeddings[f'first_layer_{layer_idx}'] = nodes.detach().clone()

        return nodes

    def encoder_forward(
        self,
        first_stage_graph,
        first_stage_variable_indices,
        is_training,
        variable_shapes,
        device
    ):
        """
        GPS-based encoder forward pass with separate variable/constraint pathways.
        """
        graph_ptrs = first_stage_graph.ptr.to(device)

        nodes = self.encode_first_graph(first_stage_graph, is_training, first_stage_variable_indices)

        offset_indices = []
        for i, indices in enumerate(first_stage_variable_indices):
            offset = graph_ptrs[i].item()
            offset_indices.append(indices.to(device) + offset)

        all_indices = torch.cat(offset_indices, dim=0)
        first_stage_nodes = nodes[all_indices.long()]

        split_sizes = []
        for shapes in variable_shapes:
            sizes = [int(torch.prod(torch.tensor(s)).item()) for s in shapes[:4]]
            split_sizes.extend(sizes)

        split_points = torch.cumsum(torch.tensor([0] + split_sizes), dim=0).to(device)

        nodes_a = []
        nodes_b = []
        nodes_c = []
        nodes_d = []

        num_graphs = len(variable_shapes)
        for i in range(num_graphs):
            offset = i * 4
            nodes_a.append(first_stage_nodes[split_points[offset + 0]:split_points[offset + 1]])
            nodes_b.append(first_stage_nodes[split_points[offset + 1]:split_points[offset + 2]])
            nodes_c.append(first_stage_nodes[split_points[offset + 2]:split_points[offset + 3]])
            nodes_d.append(first_stage_nodes[split_points[offset + 3]:split_points[offset + 4]])

        nodes_a = torch.cat(nodes_a, dim=0)
        nodes_b = torch.cat(nodes_b, dim=0)
        nodes_c = torch.cat(nodes_c, dim=0)
        nodes_d = torch.cat(nodes_d, dim=0)

        pred_a_logits = self.a_head(nodes_a)
        pred_b = self.b_head(nodes_b).squeeze(-1)
        pred_c_logits = self.c_head(nodes_c)
        pred_d = self.d_head(nodes_d).squeeze(-1)

        pred_a_per_graph = []
        pred_c_per_graph = []

        a_offset = 0
        c_offset = 0

        for shapes in variable_shapes:
            P, Td = shapes[0]
            K, Tl = shapes[2]

            a_size = P * Td
            c_size = K * Tl

            a_graph = pred_a_logits[a_offset:a_offset + a_size].view(P, Td)
            c_graph = pred_c_logits[c_offset:c_offset + c_size].view(K, Tl)

            pred_a_per_graph.append(a_graph)
            pred_c_per_graph.append(c_graph)

            a_offset += a_size
            c_offset += c_size

        fs_logits = [pred_a_per_graph, pred_b, pred_c_per_graph, pred_d]

        return fs_logits, nodes

    def decoder_forward(
        self,
        fs_logits,
        original_X_list,
        original_Y_list,
        variable_shapes,
        device
    ):
        """
        Simplified decoder with explicit tree routing using Straight-Through Estimators.

        Uses STE for discrete decisions (A, C, D, Z) to eliminate train-test discrepancy:
        - Forward pass: hard (discrete) decisions matching inference behavior
        - Backward pass: soft gradients for learning

        Z is computed directly from A, B, X using hard routing decisions.
        L is computed from Z, C, Y alignment (misclassification count).

        Args:
            fs_logits: Tuple of (pred_a_list, pred_b, pred_c_list, pred_d) from encoder
            original_X_list: List of [S_i, P] tensors - ORIGINAL feature values
            original_Y_list: List of [S_i, K] tensors - ORIGINAL one-hot labels
            variable_shapes: List of variable shapes per batch
            device: torch device
        """
        pred_a_list, pred_b, pred_c_list, pred_d = fs_logits
        batch_size = len(pred_a_list)

        all_z_logits = []
        all_L_predictions = []

        # Track offset into pred_b (which is concatenated across batch)
        b_offset = 0

        for i in range(batch_size):
            # --- Get predictions from encoder (already computed) ---
            a_logits = pred_a_list[i]  # [P, T_D]
            a_probs = temperature_softmax(a_logits, temperature=self.routing_temperature, dim=0)

            T_D = a_logits.size(1)
            b_values = pred_b[b_offset:b_offset + T_D]
            b_offset += T_D

            c_logits = pred_c_list[i]  # [K, T_L]
            c_probs = temperature_softmax(c_logits, temperature=self.routing_temperature, dim=0)

            x_original = original_X_list[i].to(device)  # [S, P]
            y_original = original_Y_list[i].to(device)  # [S, K]
            num_samples = x_original.size(0)

            # --- Explicit tree routing with temperature annealing ---
            # Node 0 (root)
            weighted_feat_0 = torch.einsum('sp,p->s', x_original, a_probs[:, 0])
            p_left_0 = temperature_sigmoid(b_values[0] - weighted_feat_0, temperature=self.routing_temperature)
            p_right_0 = 1 - p_left_0

            # Node 1 (left child)
            weighted_feat_1 = torch.einsum('sp,p->s', x_original, a_probs[:, 1])
            p_left_1 = temperature_sigmoid(b_values[1] - weighted_feat_1, temperature=self.routing_temperature)
            p_right_1 = 1 - p_left_1

            # Node 2 (right child)
            weighted_feat_2 = torch.einsum('sp,p->s', x_original, a_probs[:, 2])
            p_left_2 = temperature_sigmoid(b_values[2] - weighted_feat_2, temperature=self.routing_temperature)
            p_right_2 = 1 - p_left_2

            # --- Compute leaf assignment (Z) ---
            # Each sample goes to exactly ONE leaf (binary path via STE)
            z_leaf_0 = p_left_0 * p_left_1      # Leaf 0: left at root, left at node 1
            z_leaf_1 = p_left_0 * p_right_1     # Leaf 1: left at root, right at node 1
            z_leaf_2 = p_right_0 * p_left_2     # Leaf 2: right at root, left at node 2
            z_leaf_3 = p_right_0 * p_right_2    # Leaf 3: right at root, right at node 2

            z_probs = torch.stack([z_leaf_0, z_leaf_1, z_leaf_2, z_leaf_3], dim=1)  # [S, 4]
            # CRITICAL FIX: Clamp z_probs before log to prevent gradient explosion
            # Without clamp: d/dz[log(z+1e-8)] = 1/(z+1e-8) = 1e8 when z=0 (from STE hard routing)
            # With clamp(min=0.01): max gradient = 1/0.01 = 100 (10000x reduction)
            z_probs_clamped = z_probs.clamp(min=0.01)
            z_logits_batch = torch.log(z_probs_clamped)  # [S, 4]
            all_z_logits.append(z_logits_batch)

            # --- Compute L from Z-C-Y alignment ---
            predicted_class = torch.einsum('sl,kl->sk', z_probs, c_probs)  # [S, K]
            correct_prob = (predicted_class * y_original).sum(dim=1)  # [S]
            L_per_sample = 1 - correct_prob
            L_batch = L_per_sample.sum()
            all_L_predictions.append(L_batch.unsqueeze(0))

        # Concatenate results
        z_logits_reshaped = torch.cat(all_z_logits, dim=0)  # [total_N, 4]
        L_logits = torch.cat(all_L_predictions, dim=0)       # [batch_size]

        # Minimal diagnostic: log leaf distribution periodically
        _log_ste_routing_summary(z_logits_reshaped.detach(), pred_b.detach())

        return z_logits_reshaped, L_logits, L_logits

    def forward(self,
                first_stage_graph,
                is_training,
                first_stage_variable_indices,
                variable_shapes,
                first_stage_states,  # NEW: states containing original X/Y
                device,
                # Unused parameters kept for interface compatibility
                second_stage_constraint_features=None,
                second_stage_variable_features=None,
                second_stage_edge_indices=None,
                first_stage_constraint_shapes=None,
                second_stage_constraint_shapes=None):
        """
        Main forward pass with GPS layers and explicit tree routing.

        Args:
            first_stage_graph: Batched PyG graph for encoder
            is_training: Whether in training mode
            first_stage_variable_indices: Variable indices per graph
            variable_shapes: Variable shapes per graph
            first_stage_states: List of state dicts containing original X/Y data
            device: torch device
        """

        # === FIRST STAGE (encoder) ===
        fs_logits, nodes = self.encoder_forward(
            first_stage_graph,
            first_stage_variable_indices,
            is_training,
            variable_shapes,
            device
        )

        # === SECOND STAGE (explicit tree routing) ===
        # Extract original X from normalized_X (features only, no class label)
        original_X_list = [s['normalized_X'] for s in first_stage_states]

        # Extract Y from sample_true_labels (integer class labels) and convert to one-hot
        # sample_true_labels: [S] integer labels, need to convert to [S, K] one-hot
        original_Y_list = []
        for i, s in enumerate(first_stage_states):
            labels = s['sample_true_labels']  # [S] integer class labels
            num_classes = variable_shapes[i][2][0]  # K from c_shape [K, T_L]
            # Convert integer labels to one-hot
            # Labels may be 1-indexed, so adjust if needed
            min_label = labels.min().long().item()
            adjusted_labels = (labels - min_label).long()
            y_one_hot = F.one_hot(adjusted_labels, num_classes=num_classes).float()
            original_Y_list.append(y_one_hot)

        z_logits_reshaped, L_logits, all_L_predictions = self.decoder_forward(
            fs_logits=fs_logits,
            original_X_list=original_X_list,
            original_Y_list=original_Y_list,
            variable_shapes=variable_shapes,
            device=device
        )

        return fs_logits, z_logits_reshaped, L_logits, all_L_predictions
