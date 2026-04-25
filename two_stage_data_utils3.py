# two_stage_data_utils_updated

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, Batch
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from typing import Any, Dict, NamedTuple, Optional
from two_stage_load_features2 import check_constraints_structural, process_variables4, extract_features, create_graph_representation_two_stage
from read_solution_final2 import generate_problem_data
from typing import Any, Dict, NamedTuple, Optional ,List
import multiprocessing
import os
import numpy as np

BIAS_FEATURE_INDEX = 2 # Change this!!!!!!!!!!!!!!!!!!
SOLUTION_FEATURE_INDEX = 1
BINARY_FEATURE_INDEX = 0

# Number of variable features without incumbent features.
NUM_ROOT_VARIABLE_FEATURES = 6 # Double check this

# Number of past incumbents to include in features.
NUM_PAST_INCUMBENTS = 3 # Change this!!!!!!!!!!!!!!!!!!

# Total number of variable features.
NUM_VARIABLE_FEATURES = NUM_ROOT_VARIABLE_FEATURES + 2 * NUM_PAST_INCUMBENTS + 1
NUM_VARIABLE_FEATURES = NUM_ROOT_VARIABLE_FEATURES 

_INDICATOR_DIM = 1
_CON_FEATURE_DIM = 4 # Change this based on the number of available features

ORDER_TO_FEATURE_INDEX = {
    'coefficient': 6,
    'fractionality': 11,
}

# Effective per-node variable feature dimension
# UPDATED: Removed LP/warm_start features (3 features: linear_relaxation, distance_to_integer, warm_start_value)
# 22 base features + 4 Z_POS_DIM + 7 X_AGG_DIM + 5 VAR_TYPE_DIM = 38 total
# VAR_TYPE_DIM = 5: variable type one-hot (4) + within-type position (1)
VARIABLE_FEATURE_DIM = 22 + 4 + 7 + 5  # 22 base + 4 positional + 7 x_aggregated + 5 var_type = 38

available_cpus =  int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())) # Gets all logical CPUs on the node


# --------------- EXTRACT DATASET FEATURE METADATA --------------------- #



class DatasetTuple(NamedTuple):
    state: Dict[str, Any]
    graphs_tuple: Data
    #labels: torch.Tensor
    #integer_labels: torch.Tensor
    #integer_node_indices: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    second_stage_state: Optional[Dict[str, Any]] 

# edges_subgraph=final_edges
# second_stage_variable_nodes=z_nodes_s
# second_stage_constraint_nodes=constraint_nodes
# node_depth=None
# second_stage_constraint_nodes = merged_constraints
# second_stage_variable_nodes = final_variable_tensor[0]
def build_graph_second_gcn(
    edges_subgraph: torch.Tensor,
    second_stage_variable_nodes: torch.Tensor,
    second_stage_constraint_nodes: torch.Tensor,
    node_depth: Optional[torch.Tensor] = None,
) -> Data:
    # --- Handle NaNs in variable node features ---
    variable_features = torch.where(
        torch.isnan(second_stage_variable_nodes),
        torch.zeros_like(second_stage_variable_nodes),
        second_stage_variable_nodes,
    )
    n_variables = variable_features.size(0)
    variable_feature_dim = variable_features.size(1)

    constraint_features = second_stage_constraint_nodes
    n_constraints = constraint_features.size(0)
    constraint_feature_dim = constraint_features.size(1)

    # --- Determine target embedding dim dynamically from input tensors ---
    # Use the larger of the two input dimensions as target
    target_dim = max(variable_feature_dim, constraint_feature_dim)
    
    # --- Pad both node types to match embedding dim ---
    def add_padding(tensor, current_dim, name, desired_dim):
        if current_dim < desired_dim:
            return F.pad(tensor, (0, desired_dim - current_dim), "constant")
        elif current_dim > desired_dim:
            raise ValueError(f"{name} exceeds {desired_dim} features with {current_dim}.")
        return tensor

    padded_variables   = add_padding(variable_features, variable_feature_dim, "variables", target_dim)
    padded_constraints = add_padding(constraint_features, constraint_feature_dim, "constraints", target_dim)

    # --- Final node matrix ---
    graph_nodes = torch.cat([padded_variables, padded_constraints], dim=0)  # [n_nodes, target_dim]

    # --- Edge index and attributes ---
    edge_index = edges_subgraph.long().t().contiguous()  # [2, E] as torch.long
    edge_attr  = torch.ones(edge_index.size(1), 1, dtype=torch.float32, device=edge_index.device)  # [E, 1]

    # --- PyG Data object ---
    return Data(
        x         = graph_nodes.float(),  # [n_nodes, target_dim]
        edge_index= edge_index,           # [2, E]
        edge_attr = edge_attr,            # [E, 1]
        y         = node_depth.float() if node_depth is not None else None,
        n_nodes   = n_variables + n_constraints
    )






def bnb_model_inputs_updated(
    stage_state: Dict[str, Any],
    stage_type: str,  # "first_stage" or "second_stage"
    second_stage_predicted_nodes=None,
    second_stage_solution_loss=None,
    device=None,
    node_depth: Optional[int] = None,
) -> Data:
    """
    Unified function to handle graph data input creation for both first and second stages.

    Args:
        stage_state (Dict[str, Any]): Input state dictionary containing variable and constraint features.
        stage_type (str): Specify "first_stage" or "second_stage".
        first_stage_logits (torch.Tensor, optional): Logits from the first stage (used only in second stage).
        first_stage_loss (torch.Tensor, optional): Loss from the first stage (used only in second stage).
        device (torch.device, optional): Device to move tensors to (e.g., GPU or CPU).
        node_depth (Optional[int]): Node depth, if applicable.

    Returns:
        Data: PyTorch Geometric Data object.
    """

    # Validate feature dimensions and apply padding to ensure VARIABLE_FEATURE_DIM features
    def add_padding(tensor, current_dim, name, desired_dim=VARIABLE_FEATURE_DIM):
        if current_dim < desired_dim:
            return F.pad(tensor, (0, desired_dim - current_dim), "constant")
        elif current_dim > desired_dim:
            raise ValueError(f"{name} exceeds desired feature dimensions with {current_dim} features.")
        return tensor


    if (stage_type != "second_stage") or (stage_type == "second_stage" and second_stage_predicted_nodes is None and second_stage_solution_loss is None):

        # Extract variable features and handle NaN values
        variable_features = torch.where(
            torch.isnan(stage_state["variable_features"]),
            torch.zeros_like(stage_state["variable_features"]),
            stage_state["variable_features"],
        )

        # Squeeze first dimension if needed (specific to second-stage inputs)
        variable_features = variable_features.squeeze(0) if stage_type == "second_stage" else variable_features
        n_variables = variable_features.size(0)
        variable_feature_dim = variable_features.size(1)

        # Extract constraint features
        constraint_features = stage_state["constraint_features"]
        constraint_features = constraint_features.squeeze(0) if stage_type == "second_stage" else constraint_features
        n_constraints = constraint_features.size(0)
        constraint_feature_dim = constraint_features.size(1)


        padded_variables = add_padding(variable_features, variable_feature_dim, "padded_variables", VARIABLE_FEATURE_DIM)
        padded_constraints = add_padding(constraint_features, constraint_feature_dim, "padded_constraints", VARIABLE_FEATURE_DIM)
        nodes = torch.cat([padded_variables, padded_constraints], dim=0)


    # Add logits and loss for second stage
    elif stage_type == "second_stage" and second_stage_predicted_nodes is not None and second_stage_solution_loss is not None:
        
        second_stage_variable_indices = stage_state["second_stage_variable_indices"]
        second_stage_predicted_variable_features = second_stage_predicted_nodes[second_stage_variable_indices].float().squeeze(0)
        

        initial_variable_features =  torch.where(
            torch.isnan(stage_state["variable_features"]),
            torch.zeros_like(stage_state["variable_features"]),
            stage_state["variable_features"],
        )
        
        predicted_variable_features = initial_variable_features.clone().squeeze(0)
        predicted_variable_features = predicted_variable_features.to(device)
        
        predicted_variable_features[second_stage_variable_indices.squeeze(0)] = second_stage_predicted_variable_features # Replace rows at the specified indices

        # Squeeze first dimension if needed (specific to second-stage inputs)
        n_variables = predicted_variable_features.size(0)
        variable_feature_dim = predicted_variable_features.size(1)

        # Extract constraint features
        constraint_features = stage_state["constraint_features"]
        constraint_features = constraint_features.squeeze(0) if stage_type == "second_stage" else constraint_features
        n_constraints = constraint_features.size(0)
        constraint_feature_dim = constraint_features.size(1)

        # Validate feature dimensions and apply padding to ensure VARIABLE_FEATURE_DIM features
        def add_padding(tensor, current_dim, name, desired_dim=VARIABLE_FEATURE_DIM):
            if current_dim < desired_dim:
                return F.pad(tensor, (0, desired_dim - current_dim), "constant")
            elif current_dim > desired_dim:
                raise ValueError(f"{name} exceeds desired feature dimensions with {current_dim} features.")
            return tensor

        padded_variables = add_padding(predicted_variable_features, variable_feature_dim, "padded_variables", VARIABLE_FEATURE_DIM)
        padded_constraints = add_padding(constraint_features, constraint_feature_dim, "padded_constraints", VARIABLE_FEATURE_DIM)

        # Move tensors to device
        padded_constraints = padded_constraints.to(device)

        # Ensure loss is reshaped and padded correctly
        # reshaped_loss = second_stage_solution_loss.view(1, 1)  # Ensure reshaped loss has 2D shape
        # padded_loss = add_padding(reshaped_loss, reshaped_loss.size(1), "padded_loss", 16)
        # padded_loss = padded_loss.to(device)

        # Concatenate features into nodes
        #nodes = torch.cat([padded_variables, padded_constraints, padded_loss], dim=0)
        nodes = torch.cat([padded_variables, padded_constraints], dim=0)

    
    # Process edge indices
    edge_indices = torch.cat(
        [
            stage_state["edge_indices"][:, :1],
            stage_state["edge_indices"][:, 1:],
        ],
        dim=1,
    )
    edge_indices = edge_indices.squeeze(0) if stage_type == "second_stage" else edge_indices

    # Extract senders and receivers for edges
    senders = edge_indices[:, 0]
    receivers = edge_indices[:, 1]

    edge_features = stage_state["edge_features"]

    # Create PyTorch Geometric Data object
    graph_tuple = Data(
        x=nodes.view(-1, VARIABLE_FEATURE_DIM).float(),  # Reshape to ensure compatibility with the model (including Z positional dims)
        edge_attr=edge_features.float(),
        edge_index=edge_indices.t().contiguous(),
        senders=senders,
        receivers=receivers,
        y=node_depth.float() if node_depth is not None else None,
        batch=None,
        n_nodes=n_variables + n_constraints,
    )
    return graph_tuple






def convert_to_minimization(gt: Data, state: Dict[str, Any]) -> Data:
    nodes = gt.x
    if state['model_maximize'].item():
        num_vars = state['variable_features'].size(0)
        feature_idx = ORDER_TO_FEATURE_INDEX['coefficient']
        indices = torch.stack([
            torch.arange(num_vars),
            torch.tensor([feature_idx] * num_vars)
        ], dim=0)
        sign_change = torch.ones_like(nodes)
        sign_change[indices[0], indices[1]] = -1.0
        nodes = nodes * sign_change

    return Data(x=nodes, edge_index=gt.edge_index, edge_attr=gt.edge_attr, senders = gt.senders, receivers = gt.receivers, y=gt.y, batch=None, n_nodes = gt.n_nodes)


# Updated_function
def get_graphs_tuple_updated(
    stage_state: Dict[str, Any], 
    stage_type: str,  # "first_stage" or "second_stage"
    device: torch.device = None, 
    second_stage_predicted_nodes: torch.Tensor = None, 
    second_stage_solution_loss: torch.Tensor = None
) -> Data:
    """
    Unified function to handle graphs tuple creation for both first and second stages.

    Args:
        stage_state (Dict[str, Any]): Input state containing variable and constraint features.
        stage_type (str): Specify "first_stage" or "second_stage".
        device (torch.device, optional): Device to move tensors to (required for second stage).
        second_stage_predicted_nodes (torch.Tensor, optional): Logits from the first stage (used only in second stage).
        second_stage_solution_loss (torch.Tensor, optional): Loss from the first stage (used only in second stage).

    Returns:
        Data: A PyTorch Geometric Data object representing the graph structure.
    """
    # Add bounds (lbs and ups) to variable features
    state_with_bounds = stage_state.copy()
    # state_with_bounds['variable_features'] = torch.cat([
    #     stage_state['variable_features'],
    #     stage_state['variable_lbs'].unsqueeze(-1),
    #     stage_state['variable_ups'].unsqueeze(-1)
    # ], dim=-1)

    # Call appropriate function based on stage type
    if stage_type == "second_stage" and second_stage_predicted_nodes is not None and second_stage_solution_loss is not None:
        graphs_tuple = bnb_model_inputs_updated(
            stage_state=state_with_bounds,
            stage_type=stage_type,
            node_depth=torch.tensor(1.0),
            second_stage_predicted_nodes=second_stage_predicted_nodes,
            second_stage_solution_loss=second_stage_solution_loss,
            device=device
        )
    elif stage_type == "second_stage" and second_stage_predicted_nodes is None and second_stage_solution_loss is None:
        graphs_tuple = bnb_model_inputs_updated(
            stage_state=state_with_bounds,
            stage_type=stage_type,
            node_depth=torch.tensor(1.0)
        )
    elif stage_type == "first_stage":
        graphs_tuple = bnb_model_inputs_updated(
            state_with_bounds,
            stage_type=stage_type,
            node_depth=torch.tensor(1.0)
        )
        
    else:
        raise ValueError("Invalid `stage_type`. Must be 'first_stage' or 'second_stage'.")

    # Convert to minimization if required
    graphs_tuple = convert_to_minimization(graphs_tuple, state_with_bounds)

    return graphs_tuple



#gt_result = get_graphs_tuple(stage_state=stage_state)



def apply_feature_scaling(state, labels):

    SOLUTION_FEATURE_INDEX = 1  # Assuming these indices are defined elsewhere
    BINARY_FEATURE_INDEX = 0    # You should set these to appropriate values
    BIAS_FEATURE_INDEX = 2      # 


    sol = state['variable_features'][:, SOLUTION_FEATURE_INDEX]
    is_binary = state['variable_features'][:, BINARY_FEATURE_INDEX].bool()
    is_non_integer = ~is_binary
    continuous_sol = sol[is_non_integer]
    norm = torch.norm(continuous_sol)
    
    lbs = state['variable_lbs']
    ubs = state['variable_ups']
    state['variable_lbs'] = torch.where(is_non_integer, lbs / norm, lbs)
    state['variable_ubs'] = torch.where(is_non_integer, ubs / norm, ubs)
    
    scaled_sol = torch.where(is_non_integer, sol / norm, sol)
    variable_features = torch.cat(
        [state['variable_features'][:, :SOLUTION_FEATURE_INDEX],
         scaled_sol.unsqueeze(-1),
         state['variable_features'][:, SOLUTION_FEATURE_INDEX + 1:]],
        dim=1
    )
    state['variable_features'] = variable_features
    
    senders = state['edge_indices'][:, 0].long()
    is_integer_edge = is_non_integer[senders]
    edges = state['edge_features'].squeeze()
    scaled_edges = torch.where(is_integer_edge, edges / norm, edges)
    state['edge_features'] = scaled_edges.unsqueeze(-1)
    
    biases = state['constraint_features'][:, BIAS_FEATURE_INDEX]
    scaled_biases = biases / norm
    state['constraint_features'] = torch.cat([
        state['constraint_features'][:, :BIAS_FEATURE_INDEX],
        scaled_biases.unsqueeze(-1),
        state['constraint_features'][:, BIAS_FEATURE_INDEX + 1:],
    ], dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_non_integer = is_non_integer.to(device)
    norm = norm.to(device)
    is_non_integer = is_non_integer.unsqueeze(-1)
    scaled_labels = torch.where(is_non_integer, labels / norm, labels)
    
    return state, scaled_labels




def extract_stage_data(
    state: Dict[str, Any],
    stage_type: str,  # "first_stage" or "second_stage"
    second_stage_predicted_nodes: torch.Tensor = None,
    second_stage_solution_loss: torch.Tensor = None,
    device: torch.device = None,
    scale_features: bool = False,
    meta_info: Optional[Dict[str, Any]] = None
) -> DatasetTuple:
    """
    Unified function to extract graph and dataset data for both first and second stages.

    Args:
        stage_state (Dict[str, Any]): Input state containing variable and constraint features.
        stage_type (str): Specify "first_stage" or "second_stage".
        second_stage_predicted_nodes (torch.Tensor, optional): Logits from the first stage (used only for second stage).
        second_stage_solution_loss (torch.Tensor, optional): Loss from the first stage (used only for second stage).
        device (torch.device, optional): Device to move tensors to (used only for second stage).
        scale_features (bool, optional): Whether to apply feature scaling.

    Returns:
        DatasetTuple: Processed data for the specified stage.
    """
    # Extract labels
    stage_state = state['first_stage'] if stage_type == 'first_stage' else state
    second_stage_state = state['second_stage'] if stage_type == 'first_stage' else None

    
    # Apply feature scaling if required
    #scale_features=False
    # if scale_features:
    #     stage_state, labels = apply_feature_scaling(stage_state, labels)

    # Get graphs tuple based on stage type
    if stage_type == "second_stage" and second_stage_predicted_nodes is not None and second_stage_solution_loss is not None:
        graphs_tuple = get_graphs_tuple_updated(
            stage_state=stage_state,
            stage_type="second_stage",
            second_stage_predicted_nodes=second_stage_predicted_nodes,
            second_stage_solution_loss=second_stage_solution_loss,
            device=device
        )
    elif stage_type == "second_stage" and second_stage_predicted_nodes is  None and second_stage_solution_loss is  None:
        graphs_tuple = get_graphs_tuple_updated(
            stage_state=stage_state,
            stage_type="second_stage"
        )
    elif stage_type == "first_stage":
        graphs_tuple = get_graphs_tuple_updated(
            stage_state=stage_state,
            stage_type="first_stage"
        )
    else:
        raise ValueError("Invalid `stage_type`. Must be 'first_stage' or 'second_stage'.")

    # Extract edge indices, senders, and receivers
    edge_indices = stage_state['edge_indices']
    if stage_type == "second_stage":
        edge_indices = edge_indices.squeeze(0)
    senders = edge_indices[:, 0]
    receivers = edge_indices[:, 1]

    # Return dataset tuple
    return DatasetTuple(
        state=stage_state,
        graphs_tuple=graphs_tuple,
        senders=senders,
        receivers=receivers,
        second_stage_state=second_stage_state
    )


# extracted_data = extract_stage_data(state=state, stage_type='first_stage', scale_features= False)
# state_data = extracted_data.state
# variable_features = state_data['variable_features']
# extracted_data[1]



class MIPDataset(Dataset):
    """
    If outputs_paths is None we work in 'test' mode (no ground‑truth labels).
    """
    def __init__(self, dataset_paths, outputs_paths=None, linear_features_path=None, scale_features=False, group_size=1):
        self.dataset_paths  = [str(p) for p in dataset_paths]
        self.outputs_paths  = [str(p) for p in outputs_paths] if outputs_paths else None
        self.linear_features_path = [str(p) for p in linear_features_path] if linear_features_path else None
        self.scale_features = scale_features
        self.group_size = group_size
        self.data_triples   = list(zip(
            self.dataset_paths,
            self.outputs_paths or [None]*len(self.dataset_paths),
            self.linear_features_path or [None]*len(self.dataset_paths)
        ))

    def __len__(self):
        return len(self.data_triples)

    def __getitem__(self, idx):
        #idx=0
        dataset_path, outputs_path, linear_features_path = self.data_triples[idx]

        if outputs_path is not None and linear_features_path is not None:
            # dataset_path = test_data_path
            # outputs_path = problem_path
            feature_stats, normalized_test_data, solution_data, variable_shapes, variables_structure, linear_features = generate_problem_data(outputs_path, dataset_path, linear_features_path)
        else:
            # Test mode: set solution_data and linear features to None or dummy
            feature_stats, normalized_test_data, _, variable_shapes, variables_structure, linear_features = generate_problem_data(None, dataset_path, None)
            solution_data = {}
        
        # Skip corrupted files that returned empty solution_data
        if isinstance(solution_data, torch.Tensor) and solution_data.numel() == 0:
            # Return None to signal DataLoader to skip this sample
            return None

        # Extract warm_start from solution_data if available (training/validation mode)
        # For testing, warm_start will be None (CART solution not available for unseen data)
        warm_start = None
        if isinstance(solution_data, dict) and 'warm_start' in solution_data:
            warm_start = solution_data['warm_start']

        # Pass warm_start to process_variables4 for embedding warm_start_value features
        variable_features = process_variables4(
            variables_structure,
            normalized_test_data,
            feature_stats,
            linear_features=linear_features,
            warm_start=warm_start
        )

        constraint_features = check_constraints_structural(variables_structure,
                                                           variable_features,
                                                           linear_features)
        features = extract_features(variable_features,
                                    variable_shapes,
                                    constraint_features)

        state = create_graph_representation_two_stage(features)

        # ------------------------------------------------------------------
        # 3.  Attach extras
        # ------------------------------------------------------------------

        def to_torch_recursive(obj):
            if isinstance(obj, np.ndarray):
                return torch.tensor(obj, dtype=torch.float32)
            elif isinstance(obj, dict):
                return {k: to_torch_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_torch_recursive(v) for v in obj]
            else:
                return obj
            
        if solution_data is not None:
            state['first_stage']['solution_data'] = to_torch_recursive(solution_data)

        # Add normalized_X for tree routing (needed for GRPO feasibility check)
        # normalized_test_data is [N, P+1] where last column is class label
        # We want just the features [N, P]
        state['first_stage']['normalized_X'] = torch.tensor(
            normalized_test_data[:, :-1], dtype=torch.float32
        )
        
        # Add sample_true_labels for end-to-end accuracy (GRPO reward computation)
        # The last column of normalized_test_data contains the class labels
        state['first_stage']['sample_true_labels'] = torch.tensor(
            normalized_test_data[:, -1], dtype=torch.long
        )

        if outputs_path is not None and linear_features_path is not None:
            state['first_stage']['meta_info'] = {
                "idx"         : idx,
                "dataset_path": dataset_path,
                "outputs_path": outputs_path,
                'linear_features_path': linear_features_path
            }
        else:
            state['first_stage']['meta_info'] = {
                "idx"         : idx,
                "dataset_path": dataset_path,
                "outputs_path": '',
                'linear_features_path': ''
            }

        # ------------------------------------------------------------------
        # 4.  Produce one Sample (DatasetTuple)
        # ------------------------------------------------------------------
        sample = extract_stage_data(state,
                                    stage_type='first_stage',
                                    scale_features=self.scale_features)
        
        if sample is None:
            print(f"[DEBUG] Extracted sample is None at index {idx}, file: {dataset_path}")
            raise ValueError(f"extract_stage_data returned None for dataset {dataset_path}")

        return sample




def custom_collate_fn(batch):
    # Filter out None items (corrupted/incomplete files)
    batch = [item for item in batch if item is not None]
    
    # If entire batch is None, return None to signal skip
    if len(batch) == 0:
        return None
    
    graphs = [item.graphs_tuple for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    return DatasetTuple(
        state=[item.state for item in batch],
        graphs_tuple=batched_graph,
        senders=[item.senders for item in batch],
        receivers=[item.receivers for item in batch],
        second_stage_state=[item.second_stage_state for item in batch],
    )


# Updated function
# dataset_paths = test_data_path
# outputs_paths = problem_path
# linear_features_path = linear_features_path

# dataset_paths = train_problems_datasets[0][0]
# outputs_paths = train_problems_outputs[0][0]
# linear_features_path = train_problems_linear_feats[0][0]
# group_size=2
# scale_features=False

def get_dataset(dataset_paths,
                outputs_paths=None,
                linear_features_path=None,
                scale_features=False,
                batch_size=1,
                shuffle=True,
                group_size=1,
                num_workers=4):
    """
    Wrapper to build the evaluation/training DataLoader.
    num_workers is exposed so evaluation can set it to 0 to avoid
    multiprocess overhead and NFS temp-file contention on small runs.
    """
    dataset = MIPDataset(dataset_paths, outputs_paths, linear_features_path, scale_features, group_size=group_size)

    prefetch = 4 if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch,
        collate_fn=custom_collate_fn  # Now references the top-level function
    )

# dataset = get_dataset(dataset_paths, outputs_paths, linear_features_path, scale_features=False, batch_size=1, shuffle=False)
# train_loader = next(iter(dataset))  # Get 1 DataLoader
# batch = next(iter(train_loader))
# batch2 = next(iter(train_loader))
# train_loader[0]['meta_info']
# valid_loader = next(iter(valid_data_loaders))  # Get 1 DataLoader