# --- TRAIN 
import re
import numpy as np
import math
from scipy.stats import skew, kurtosis
import torch
from typing import Union, Optional

# Extract matrices and vectors from the log file
def extract_matrices_and_vectors(problem_path):

    with open(problem_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    log_text = ''.join(lines)

    dimension_pattern = r'dimension: (\d+), class: (\d+)'
    dimension_match = re.search(dimension_pattern, log_text)
    if dimension_match:
        num_rows = int(dimension_match.group(1))
        num_cols = int(dimension_match.group(2))
    else:
        # Return empty dicts for corrupted/incomplete files instead of crashing
        return {}, {}

    matrix_pattern = r'(\d+)×(\d+) Matrix{Float64}:([\s\S]+?)(?=\n\d|\ncomparison|\n\d+-element)'
    vector_pattern = r'(\d+)-element Vector{Float64}:([\s\S]+?)(?=\n\d|\ncomparison|\n7×|\n3×)'

    matrices = re.findall(matrix_pattern, log_text)
    vectors = re.findall(vector_pattern, log_text)

    def parse_matrix(matrix_data):
        rows, cols, values = matrix_data
        rows, cols = int(rows), int(cols)
        values = np.fromstring(values.replace('\n', ' ').replace('  ', ' '), sep=' ')
        return values.reshape((rows, cols))

    def parse_vector(vector_data):
        size, values = vector_data
        values = np.fromstring(values.replace('\n', ' '), sep=' ')
        return values

    matrices_extracted = [parse_matrix(matrix) for matrix in matrices]
    vectors_extracted = [parse_vector(vector) for vector in vectors]

    variable_names = ['variable_a', 'variable_b', 'variable_c', 'variable_d']
    
    warm_start = {}
    final_solution = {}
    
    if len(matrices_extracted) > 0 and len(vectors_extracted) > 0:
        warm_start[variable_names[0]] = matrices_extracted[0] if len(matrices_extracted) > 0 else None
        warm_start[variable_names[1]] = vectors_extracted[0] if len(vectors_extracted) > 0 else None
        warm_start[variable_names[2]] = matrices_extracted[1] if len(matrices_extracted) > 1 else None
        warm_start[variable_names[3]] = vectors_extracted[1] if len(vectors_extracted) > 1 else None
    
    if len(matrices_extracted) > 2 and len(vectors_extracted) > 2:
        final_solution[variable_names[0]] = matrices_extracted[2] if len(matrices_extracted) > 2 else None
        final_solution[variable_names[1]] = vectors_extracted[2] if len(vectors_extracted) > 2 else None
        final_solution[variable_names[2]] = matrices_extracted[3] if len(matrices_extracted) > 3 else None
        final_solution[variable_names[3]] = vectors_extracted[3] if len(vectors_extracted) > 3 else None

    return warm_start, final_solution

# Extract time, ub, lb, and gap from the log file
def extract_values_from_segment(segment):
    lines = segment.strip().split("\n")
    values_line = lines[1]  
    values = re.split(r'\s+', values_line)
    time = float(values[1])  
    ub = float(values[2])    
    lb = float(values[3])    
    gap = float(values[4])   
    return {'time': time, 'ub': ub, 'lb': lb, 'gap': gap}

# Process the file and extract solution data
def process_solution_file(problem_path):
    with open(problem_path, 'r',  encoding='utf-8', errors='replace') as file:
        file_content = file.read()
    
    lines = file_content.strip().split("\n")
    extracted_info = None  # Will store only the last segment
    
    for i in range(len(lines)):
        if "Dataname" in lines[i]:
            # Overwrite extracted_info with the current segment (last one wins)
            segment = "\n".join(lines[i:i+3])
            extracted_info = extract_values_from_segment(segment)
    
    if extracted_info is None:
        raise ValueError(f"No 'Dataname' segment found in {problem_path}")
    
    return [extracted_info]


# Normalize the training data
def normalize_data(training_data):
    data_min = np.min(training_data[:, :-1], axis=0)
    data_max = np.max(training_data[:, :-1], axis=0)
    range_data = data_max - data_min
    range_data[range_data == 0] = 1
    normalized_data = (training_data[:, :-1] - data_min) / range_data
    return normalized_data


# Predict leaf nodes for a sample
def predict(sample, feature_indices, variable_b):
    current_node = 0  
    feature_idx = feature_indices[current_node]
    split_value = variable_b[current_node]  

    if sample[feature_idx] >= split_value:
        current_node = 2
    else:
        current_node = 1

    if current_node == 1:  
        if sample[feature_indices[1]] >= variable_b[1]:
            final_leaf_node = 1  
        else:
            final_leaf_node = 0  
    else:  
        if sample[feature_indices[2]] >= variable_b[2]:
            final_leaf_node = 3  
        else:
            final_leaf_node = 2  

    return final_leaf_node + 1


# One-hot encoding of the predicted leaf nodes
def leaf_node_encoding(leaf_node, num_leaves=4):
    encoding = np.zeros(num_leaves)
    encoding[leaf_node - 1] = 1  
    return encoding


# Predict all samples
def predict_all_samples(normalized_data, feature_indices, variable_b):
    predicted_leaf_nodes = []
    for sample in normalized_data:
        leaf_node = predict(sample, feature_indices, variable_b)
        one_hot_leaf_node = leaf_node_encoding(leaf_node)
        predicted_leaf_nodes.append(one_hot_leaf_node)
    return np.array(predicted_leaf_nodes)


# Convert real leaf values to one-hot encoding
def convert_real_leaf_values_to_one_hot(real_leaf_values):
    # Determine the number of leaves by finding the max value in real_leaf_values
    min_label = int(np.min(real_leaf_values))
    max_label = int(np.max(real_leaf_values))
    num_leaves = max_label - min_label + 1
    
    # Initialize the one-hot encoded matrix
    one_hot_real_leafs = np.zeros((real_leaf_values.shape[0], num_leaves))
    
    # Populate the one-hot encoding based on real_leaf_values
    for i, leaf in enumerate(real_leaf_values):
        one_hot_real_leafs[i, int(leaf) - min_label] = 1  # Adjust for potential non-zero-based labels
    
    return one_hot_real_leafs



# Compute the total cost L
# Compute the prediction cost per sample (1 if prediction is incorrect, 0 if correct)
def compute_prediction_cost(predicted_leaf_nodes, real_leaf_values_one_hot, variable_c):
    # Extract the leaf node labels from the last four columns of variable_c
    leaf_labels = np.argmax(variable_c[:, -4:], axis=0) + 1  # Add 1 to get actual labels (1-based indexing)

    # Determine the predicted label for each sample based on predicted_leaf_nodes
    # Each row in predicted_leaf_nodes corresponds to an observation and each column corresponds to a leaf node
    predicted_labels = np.argmax(predicted_leaf_nodes, axis=1)  # Get the leaf node index (0-based)

    # Map the leaf node index to the actual label using the extracted leaf_labels
    predicted_classifications = leaf_labels[predicted_labels]

    # Convert the real one-hot labels back to class labels
    real_classifications = np.argmax(real_leaf_values_one_hot, axis=1) + 1  # Add 1 to get 1-based labels

    # Compare predicted_classifications with real_classifications to determine cost (0 if correct, 1 if wrong)
    sample_costs = (predicted_classifications != real_classifications).astype(int)

    return sample_costs



def extract_time_from_file(file_path):
    """
    Extracts the `time` value from the last 'Dataname' segment in the text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        float: Extracted time value from the last segment.
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
    
    # Search from the end of the file for the last "Dataname" occurrence
    for i in range(len(lines) - 1, -1, -1):  # Iterate backwards
        if lines[i].startswith("Dataname"):
            # The next line contains the data
            next_line = lines[i + 1].strip()
            # Split and extract the time value (2nd column)
            time_value = float(next_line.split("\t")[1])
            return time_value
    
    raise ValueError(f"No 'Dataname' segment found in {file_path}")




def extract_linear_features(linear_features_path):
    """
    Extracts relaxed LP solution for a, b, c, d variables from solution file.
    
    Args:
        linear_features_path (str): Path to the solution file.
    
    Returns:
        dict: Dictionary with keys 'variable_a', 'variable_b', 'variable_c', 'variable_d' containing numpy arrays.
    """
    with open(linear_features_path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
    
    result = {}
    
    # Extract each variable
    for var_name in ['a', 'b', 'c', 'd']:
        pattern = rf'{var_name}: \[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError(f"Variable '{var_name}' not found in {linear_features_path}")
        
        # Get the content inside brackets
        var_str = match.group(1)
        
        # Handle matrix format (with semicolons)
        if ';' in var_str:
            # Split by semicolons for rows
            rows = [row.strip() for row in var_str.split(';')]
            matrix = []
            for row in rows:
                # Convert each row to floats
                values = [float(x) for x in row.split() if x]
                if values:  # Skip empty rows
                    matrix.append(values)
            result[f'variable_{var_name}'] = np.array(matrix)
        else:
            # Handle vector format (comma-separated)
            values = [float(x.strip()) for x in var_str.split(',')]
            result[f'variable_{var_name}'] = np.array(values)

    result['variable_c'] = result['variable_c'][:, 3:]  # Extract the last 4 columns of variable_c as variable_c
    
    return result




# Extract information from the dataset
def analyze_dataset(test_data):

    # Number of total columns
    total_columns = test_data.shape[1]

    # Features are all but the last column
    num_features = total_columns - 1

    # The last column is the target variable y
    y = test_data[:, -1]

    # Number of unique classes in y
    num_classes = len(np.unique(y))

    # Number of samples
    num_samples = test_data.shape[0]

    return num_features, num_classes, num_samples



def compute_variable_shapes(num_features, num_classes, num_samples, D=2):
    # Compute total nodes in the tree
    T = 2**(D + 1) - 1
    Td = math.floor(T / 2)
    Tl = T - Td

    # Compute shapes
    v_a_shape = [num_features, Td]
    v_b_shape = [Td]
    v_c_shape = [num_classes, Tl]
    v_d_shape = [Td]
    v_z_shape = [num_samples, Tl]
    v_L_shape = [num_samples]
    v_y_shape = [num_samples, num_classes]
    v_x_shape = [num_samples, num_features]

    variable_shapes = [
        v_a_shape,
        v_b_shape,
        v_c_shape,
        v_d_shape,
        v_L_shape,
        v_z_shape,
        v_y_shape,
        v_x_shape
    ]

    return variable_shapes


def build_variable_matrices(variable_shapes):
    # Unpack shapes
    v_a_shape, v_b_shape, v_c_shape, v_d_shape, v_L_shape, v_z_shape, v_y_shape, v_x_shape = variable_shapes

    # Create zero matrices/vectors with appropriate shapes
    v_a = np.zeros(v_a_shape)
    v_b = np.zeros((v_b_shape[0], 1))           # convert [3] to [3,1]
    v_c = np.zeros(v_c_shape)
    v_d = np.zeros((v_d_shape[0], 1))           # convert [3] to [3,1]
    v_L = np.zeros((v_L_shape[0], 1))           # convert [206] to [206,1]
    v_z = np.zeros(v_z_shape)
    v_y = np.zeros(v_y_shape)
    v_x = np.zeros(v_x_shape)

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

    return variables


# Function to read file and process data
def process_file_to_numpy(file_path):
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            data = file.read()

        # Split the data into rows and convert to list of lists
        data_rows = [list(map(float, line.split(','))) for line in data.strip().split('\n')]

        # Convert list of lists into a NumPy array
        test_data = np.array(data_rows)

        return test_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Main function that calls all the others and organizes the results
# test_data_path='C:\\Users\\josen\\OneDrive\\Documentos\\two_stage_gcn_hpc\\hpc_datasets\\train\\seeds\\training\\seeds_372.seeds'
# problem_path='C:\\Users\\josen\\OneDrive\\Documentos\\two_stage_gcn_hpc\\hpc_datasets\\train\\seeds\\outputs\\info-seeds_372.seeds-sd1-2-CMS-6.out'
# linear_features_path= 'C:\\Users\\josen\\OneDrive\\Documentos\\two_stage_gcn_hpc\\hpc_datasets\\train\\seeds\\linear_feats\\info-seeds_372.seeds-sd1-2-CMS-6.out'

# test_data_path='C:\\Users\\navarrodelacruz\\Documents\\GitHub\\two_stage_gcn_hpc\\hpc_datasets\\train\\glass\\training\\glass_1.glass'
# problem_path='C:\\Users\\navarrodelacruz\\Documents\\GitHub\\two_stage_gcn_hpc\\hpc_datasets\\train\\glass\\outputs\\info-glass_1.glass-sd1-2-CMS-6.out'

def generate_problem_data(problem_path: Optional[str], test_data_path: str, linear_features_path: Optional[str]):
    """
    Training call:
        problem_path != None  ->  returns full solution_data dict
        linear_features_path != None ->  adds extracted a, b, c, d to solution_data
    Testing  call:
        problem_path is None  or file does not exist
                                -> returns linear_features_path = None
    """

    # ------------------------------------------------------------------
    # 1. always load the *dataset* file (features)
    # ------------------------------------------------------------------
    test_data = process_file_to_numpy(test_data_path)
    if test_data is None:
        raise ValueError(f"Failed to process test data from {test_data_path}")

    num_features, num_classes, num_samples = analyze_dataset(test_data)
    variable_shapes            = compute_variable_shapes(num_features,
                                                          num_classes,
                                                          num_samples)
    variables_structure        = build_variable_matrices(variable_shapes)
    
    linear_features = None
    # ------------------------------------------------------------------
    # 2. put x into the structure (same for train / test)
    # ------------------------------------------------------------------
    feature_means = test_data[:, :-1].mean(axis=0)
    feature_stds = test_data[:, :-1].std(axis=0)
    feature_skews = skew(test_data[:, :-1], axis=0)
    feature_kurtoses = kurtosis(test_data[:, :-1], axis=0)

    feature_stats = {
        'mean': feature_means,
        'std': feature_stds,
        'skew': feature_skews,
        'kurt': feature_kurtoses
    }

    variable_x = normalize_data(test_data)
    variables_structure['variable_x'] = variable_x
    
    class_column = test_data[:, -1].reshape(-1, 1)
    normalized_test_data = np.concatenate((variable_x, class_column), axis=1)

    # ------------------------------------------------------------------
    # 3. If we *do* have an outputs file → build full solution_data
    # ------------------------------------------------------------------
    if problem_path is not None:
        try:
            warm_start, final_solution = extract_matrices_and_vectors(problem_path)
            
            # Check if extraction was successful (non-empty dicts)
            if warm_start and final_solution:
                warm_start['variable_c'] = warm_start['variable_c'][:, 3:]
                final_solution['variable_c'] = final_solution['variable_c'][:, 3:]

                solution_data = {
                    'warm_start'    : warm_start,
                    'final_solution': final_solution,
                    'extracted_info': process_solution_file(problem_path),
                    'variable_x'    : variable_x,
                    'time'          : extract_time_from_file(problem_path)
                }

                # Add extracted linear relaxation variables if solution_path is provided
                if linear_features_path is not None:
                    linear_features = extract_linear_features(linear_features_path)

                # --- secondary labels derived from final_solution ----------------
                feature_indices = np.argmax(final_solution['variable_a'], axis=0)
                pred_leaf_nodes = predict_all_samples(variable_x,
                                                      feature_indices,
                                                      final_solution['variable_b'])
                solution_data['variable_z'] = pred_leaf_nodes

                real_leaf_vals = test_data[:, -1]
                real_one_hot   = convert_real_leaf_values_to_one_hot(real_leaf_vals)
                if problem_path is None:
                    variables_structure['variable_y'] =  torch.empty(0)
                variables_structure['variable_y'] = real_one_hot

                total_cost = compute_prediction_cost(pred_leaf_nodes,
                                                     real_one_hot,
                                                     final_solution['variable_c'])
                solution_data['variable_L'] = total_cost
            else:
                # Corrupted/incomplete file - return empty solution_data
                solution_data = torch.empty(0)
        except Exception as e:
            # Any other error during solution extraction - skip this file
            solution_data = torch.empty(0)

    else:
        # inference / test – no ground‑truth available
        solution_data =  torch.empty(0)

    return feature_stats, normalized_test_data, solution_data, variable_shapes, variables_structure, linear_features


# feature_stats, normalized_test_data, solution_data, variable_shapes, variables_structure, linear_features = generate_problem_data(problem_path, test_data_path, linear_features_path)
