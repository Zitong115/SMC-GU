import os
import errno
import numpy as np
import pandas as pd
import networkx as nx
import torch
import json
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from scipy.sparse.linalg import spsolve
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import gc
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor
import psutil

EXTRA_INFO_PATH = "/raid/zitong/scalable_GNN_unlearning/extra_info"
IMPORTANCE_THRESHOLD_FOR_METHOD_SELECTION = 99
REMAINED_GRAPH_PATH = "/home/zitong/graph_unlearning/scalable_GNN_unlearning/remained_graph"

def get_true_indices(mask):
    """
    Get indices of True values from a mask, handling different types and devices
    
    Args:
        mask: Can be numpy array, torch.Tensor on CPU or CUDA
    
    Returns:
        numpy array of indices where mask is True
    """
    # Case 1: If input is numpy array
    if isinstance(mask, np.ndarray):
        return np.where(mask)[0]
    
    # Case 2: If input is torch tensor
    elif torch.is_tensor(mask):
        # Check if tensor is on CUDA
        if mask.is_cuda:
            # Get indices while handling CUDA tensor
            return torch.where(mask)[0].cpu().numpy()
        else:
            # Get indices from CPU tensor
            return torch.where(mask)[0].numpy()
    
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

def randomly_flip_ones_to_zeros(binary_vector, num_to_flip):
    """
    Randomly turn a specific number of 1s to 0s in a binary vector
    
    Args:
        binary_vector: A tensor or numpy array with 0-1 values
        num_to_flip: Number of 1s to flip to 0s
        
    Returns:
        Modified binary vector with num_to_flip 1s changed to 0s
    """
    # Convert to numpy if tensor
    is_tensor = torch.is_tensor(binary_vector)
    if is_tensor:
        device = binary_vector.device
        binary_vector = binary_vector.cpu().numpy()
    
    # Find indices of 1s
    one_indices = np.where(binary_vector == 1)[0]
    
    # Check if we have enough 1s to flip
    num_ones = len(one_indices)
    if num_ones == 0:
        print("Warning: No 1s to flip in the binary vector")
        return torch.tensor(binary_vector, device=device) if is_tensor else binary_vector
    
    # Adjust num_to_flip if it's larger than available 1s
    num_to_flip = min(num_to_flip, num_ones)
    
    # Randomly select indices to flip
    indices_to_flip = np.random.choice(one_indices, size=num_to_flip, replace=False)
    
    # Create a copy and flip the selected indices
    result = binary_vector.copy()
    result[indices_to_flip] = 0
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        result = torch.tensor(result, device=device)
    
    return result

def get_remain_data_file_name(dataset_name, unlearn_task, unlearn_ratio):
    return '_'.join([dataset_name, unlearn_task, str(unlearn_ratio)])

def convert_tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array, handling different types and devices

    Args:
        tensor: Can be torch.Tensor on CPU or CUDA

    Returns:
        numpy array
    """
    # Case 1: If tensor is on CUDA, get value and convert to CPU
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    # Case 2: If tensor is on CPU, get value directly
    else:
        return tensor.numpy()

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph

def find_nodes_within_path_length(edge_index, num_nodes, source_nodes, threshold):
    """
    Find nodes that have shortest path length < threshold from any node in source_nodes
    
    Args:
        edge_index: Edge list of shape (2, num_edges)
        num_nodes: Total number of nodes in the graph
        source_nodes: Set of source nodes (set A)
        threshold: Maximum path length threshold
    
    Returns:
        result_nodes: Set of nodes (set B) that are within threshold distance from any node in A
    """
    
    # Convert edge_index to adjacency list for faster neighbor lookup
    adj_list = [[] for _ in range(num_nodes)]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # Remove if directed graph
    
    # Initialize visited set and result set
    visited = set()
    result_nodes = set()
    
    # BFS from each source node
    for source in source_nodes:
        if source in visited:
            continue
            
        queue = deque([(source, 0)])  # (node, distance)
        level_visited = {source}  # Track visited nodes for this BFS
        
        while queue:
            node, dist = queue.popleft()
            
            # If we've exceeded threshold, skip this node
            if dist >= threshold:
                continue
            
            # Add neighbors to queue
            for neighbor in adj_list[node]:
                if neighbor not in level_visited:
                    level_visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    result_nodes.add(neighbor)
        
        visited.update(level_visited)
    
    # Remove source nodes from result if needed
    result_nodes -= set(source_nodes)
    
    return result_nodes

def calculate_katz_index_direct(edge_index, num_nodes, alpha=0.1, use_sparse=True):
    """
    Calculate Katz centrality index for large graphs
    
    Args:
        edge_index: PyG edge_index tensor or numpy array of shape (2, num_edges)
        num_nodes: Number of nodes in the graph
        alpha: Attenuation factor (should be less than 1/largest_eigenvalue)
        batch_size: Batch size for processing large matrices
        sparse: Whether to use sparse matrix operations (recommended for large graphs)
    
    Returns:
        katz_index: numpy array of Katz centrality scores
    """
    # Convert edge_index to numpy if it's a tensor
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    # Create sparse adjacency matrix
    adj = sparse.coo_matrix(
        (np.ones(edge_index.shape[1]), 
         (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Make symmetric for undirected graph
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    if use_sparse:
        print("start using sparse vector to calculate katz index...")
        # Sparse implementation (memory efficient)
        identity = sparse.eye(num_nodes)
        # Solve (I - αA)x = 1
        katz_index = spsolve(identity - alpha * adj, np.ones(num_nodes))
        katz_index = katz_index - identity
    else:
        # Dense implementation (faster for small graphs)
        adj_dense = adj.todense()
        identity = np.eye(num_nodes)
        katz_index = np.linalg.solve(identity - alpha * adj_dense, np.ones(num_nodes))
        katz_index = katz_index - identity
    
    # Normalize scores
    katz_index = katz_index / np.linalg.norm(katz_index)
    
    return katz_index

def estimate_largest_eigenvalue(edge_index, num_nodes, max_iter=100):
    """
    Estimate largest eigenvalue using power iteration
    Used to determine safe beta value
    """
    rows, cols = edge_index
    data = np.ones_like(rows)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    x = np.random.rand(num_nodes)
    
    for _ in range(max_iter):
        x_next = A.dot(x)
        norm = np.linalg.norm(x_next)
        x = x_next / norm
        
    return float(norm)

def sparse_eye(n, device=None, dtype=None):
    """
    Create a sparse identity matrix
    
    Args:
        n: Size of the matrix (n x n)
        device: torch device (default: None)
        dtype: torch dtype (default: None)
    
    Returns:
        Sparse tensor of size (n x n) with ones on the diagonal
    """
    indices = torch.arange(n, device=device)
    indices = torch.stack([indices, indices])
    values = torch.ones(n, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n))# .to_sparse_csr()

def incremental_sparse_construction(result_indices, result_values, size):
    """Build sparse tensor incrementally"""
    device = result_values[0].device
    
    # Initialize with first element
    current_indices = result_indices[0]
    current_values = result_values[0]
    
    # Clear memory
    result_indices[0] = None
    result_values[0] = None
    
    # Create initial sparse tensor
    current_tensor = torch.sparse_coo_tensor(
        current_indices, current_values, size, device=device
    ).coalesce()
    
    # Add remaining elements incrementally
    for i in range(1, len(result_values)):
        # Create temporary tensor
        temp_tensor = torch.sparse_coo_tensor(
            result_indices[i], result_values[i], size, device=device
        ).coalesce()
        
        # Add to current tensor
        current_tensor = (current_tensor + temp_tensor).coalesce()
        
        # Clear memory
        result_indices[i] = None
        result_values[i] = None
        torch.cuda.empty_cache()
    
    return current_tensor


def chunk_sparse_mm(A, B, beta = -1, chunk_size=512):
    """
    Perform sparse matrix multiplication in chunks
    A: sparse matrix (M x N)
    B: sparse matrix (N x K)
    chunk_size: number of columns to process at once
    """
    device = A.device
    M, N = A.size()
    _, K = B.size()
    
    print("A.size(), B.size():", A.size(), B.size())
    
    # Ensure matrices are coalesced
    A = A.coalesce()
    B = B.coalesce()
    
    # Initialize result list to store chunks
    result_indices = []
    result_values = []
    
    # Process columns in chunks
    for chunk_start in tqdm(range(0, K, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, K)
        
        # Get indices and values for columns in current chunk
        B_indices = B._indices()
        B_values = B._values()
        
        # Find elements in the current column chunk
        col_mask = (B_indices[1] >= chunk_start) & (B_indices[1] < chunk_end)
        chunk_indices = B_indices[:, col_mask]
        chunk_values = B_values[col_mask]
        
        if chunk_indices.size(1) > 0:  # Only process if chunk has non-zero elements
            # Adjust column indices for the chunk
            chunk_indices = chunk_indices.clone()
            chunk_indices[1] -= chunk_start
            
            # Create chunk sparse tensor
            B_chunk = torch.sparse_coo_tensor(
                indices=chunk_indices,
                values=chunk_values,
                size=(N, chunk_end - chunk_start),
                device=device
            ).coalesce()
            
            # Perform multiplication for this chunk
            chunk_result = torch.sparse.mm(A, B_chunk)
            
            # If chunk result has non-zero elements, add to results
            if chunk_result._nnz() > 0:
                chunk_result = chunk_result.coalesce()
                chunk_result_indices = chunk_result._indices()
                
                # Adjust column indices back to original position
                chunk_result_indices = chunk_result_indices.clone()
                chunk_result_indices[1] += chunk_start
                
                result_indices.append(chunk_result_indices)
                result_values.append(chunk_result._values())
        
        # Clear memory
        torch.cuda.empty_cache()
    
    # Combine all chunks
    if len(result_indices) > 0:
        
        """
        indices = torch.cat(result_indices, dim=1)
        values = torch.cat(result_values)
        print("indices.shape, values.shape:", indices.shape, values.shape)
        # scale by beta
        if(beta > 0):
            values = values * beta
        
        result = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(M, K),
            device=device
        ).coalesce()
        """
        result = incremental_sparse_construction(result_indices = result_indices, result_values = result_values, size = (M, K))
    else:
        # Return empty sparse matrix if no non-zero elements
        result = torch.sparse_coo_tensor(
            indices=torch.empty(2, 0, device=device, dtype=torch.long),
            values=torch.empty(0, device=device, dtype=A.dtype),
            size=(M, K)
        )
    
    return result

def calculate_katz_index_chunked(edge_index, num_nodes, beta=0.1, max_iter=100,tolerance=1e-6, chunk_size=512, sparsity_threshold = 10):
    """
    Calculate Katz centrality using chunked sparse matrix multiplication
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sparse adjacency matrix
    rows, cols = edge_index
    data = torch.ones(len(rows), dtype=torch.float32, device=device)
    indices = torch.stack([rows.long(), cols.long()], dim=0)
    A = torch.sparse_coo_tensor(indices, data, size=(num_nodes, num_nodes), 
                              device=device).coalesce()
    
    # Initialize
    X = sparse_eye(num_nodes, device=device, dtype=torch.float32)
    # current_term = sparse_eye(num_nodes, device=device, dtype=torch.float32)
    
    # βA
    current_term = torch.sparse_coo_tensor(
            indices=indices,
            values=data * beta,
            size=(num_nodes, num_nodes),
            device=device
        ).coalesce()
    
    X = (X + current_term).coalesce()
    
    print("chunk_size: %d, max_iter: %d" % (chunk_size, max_iter))
    
    try:
        for i in range(max_iter):
            # Compute next term using chunked multiplication
            new_term = chunk_sparse_mm(A, current_term, chunk_size=chunk_size, beta = beta)
            new_term = new_term.coalesce()
            
            """
            # Scale by beta
            new_term = torch.sparse_coo_tensor(
                indices=new_term._indices(),
                values=new_term._values() * beta,
                size=new_term.size(),
                device=device
            ).coalesce()
            """
            # Check sparsity
            nnz = new_term._nnz()
            sparsity = nnz / (num_nodes * num_nodes)
            
            print(f"Iteration {i}, Sparsity: {sparsity:.4f}")
            if sparsity > sparsity_threshold:
                print(f"Stopping at iteration {i} due to density: {sparsity:.4f}")
                break
        
            # Update current term
            current_term = new_term
            
            # Add to result
            X = (X + current_term).coalesce()
            
            # Check convergence
            if torch.sqrt(torch.sum(current_term._values() ** 2)) < tolerance:
                break
            
            # Clear cache
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        print(f"Error during computation: {e}")
        # Try with smaller chunk size or fall back to CPU
        if chunk_size > 128:
            print(f"Retrying with smaller chunk size...")
            return calculate_katz_index_chunked(
                edge_index, num_nodes, beta, max_iter, 
                tolerance, chunk_size=chunk_size//2
            )
        else:
            print("Falling back to CPU computation...")
            return calculate_katz_index_cpu(edge_index, num_nodes, beta, max_iter, tolerance)
    
    return X

def calculate_katz_index_fullbatch(edge_index, num_nodes, beta=0.1, max_iter=100, tolerance=1e-6):
    """
    Calculate Katz centrality for large graphs using sparse matrices
    (I + βA + β²A² + β³A³ + ...) * 1

    Args:
        edge_index: Graph edges as a 2xN array of source and target nodes
        num_nodes: Total number of nodes in the graph
        beta: Attenuation factor (should be less than 1/largest eigenvalue)
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        Katz centrality scores for each node
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sparse adjacency matrix
    #rows, cols = edge_index
    #data = np.ones_like(rows)
    
    # Create sparse adjacency matrix
    rows, cols = edge_index
    data = torch.ones(len(rows), dtype=torch.float32, device=device)
    
    if isinstance(rows, np.ndarray):
        rows = rows.astype(int)
        cols = cols.astype(int)
    elif isinstance(rows, torch.Tensor):
        rows = rows.long()
        cols = cols.long()
        
    indices = torch.stack([rows, cols], dim=0)
    A = torch.sparse_coo_tensor(indices, data, size=(num_nodes, num_nodes), device=device)# .to_sparse_csr()
    
    #A = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    #A = torch.sparse_coo_tensor(indices = (rows, cols), data = data, size=(num_nodes, num_nodes), device=device).to_sparse_csr()
    
    # Initialize centrality scores
    # Initialize batch vectors
    # X = torch.zeros((num_nodes, num_nodes), dtype=torch.float16, device=device)
    # for i, node in enumerate(batch_nodes):
    #     X[i, node] = 1.0

    # Power iteration for batch
    # scores = X.clone()
    # current_terms = X.clone()
        
    X = sparse_eye(num_nodes, device=device, dtype=torch.float32)
    current_term = sparse_eye(num_nodes, device=device, dtype=torch.float32)
    # X = sparse.eye(num_nodes)
    # current_term = sparse.eye(num_nodes)
    
    # Power series expansion: I + βA + β²A² + β³A³ + ...
    for i in range(max_iter):
        # Compute next term in series
        # current_term = beta * A.dot(current_term)
        current_term = beta * torch.sparse.mm(A, current_term)
        X = X + current_term
        
        # print("type(current_term): ", type(current_term))
        # Check convergence
        if torch.sqrt(torch.sum(current_term.coalesce().values() ** 2)) < tolerance:
            break
        #if torch.linalg.norm(current_term) < tolerance:
        #    break
    
    return X.toarray()

def calculate_katz_for_node_pairs_gpu_fullbatch(edge_index, num_nodes, deleted_nodes, beta=0.1, max_iter=10, tolerance=1e-6):
    """
    Calculate Katz index between deleted nodes and remaining nodes
    
    Args:
        edge_index: Edge list of shape (2, num_edges)
        num_nodes: Total number of nodes
        deleted_nodes: Array of nodes to be deleted
        beta: Damping factor
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        katz_matrix: shape (len(deleted_nodes), len(remained_nodes))
        remained_nodes: array of remaining node indices
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create sparse adjacency matrix
    rows, cols = edge_index
    data = torch.ones(len(rows), dtype=torch.float32, device=device)
    
    print(type(rows), type(cols))
    if isinstance(rows, np.ndarray):
        rows = rows.astype(int)
        cols = cols.astype(int)
    elif isinstance(rows, torch.Tensor):
        rows = rows.long()
        cols = cols.long()
        
    indices = torch.stack([rows, cols], dim=0)
    A = torch.sparse_coo_tensor(indices, data, size=(num_nodes, num_nodes), device=device).to_sparse_csr()

    # Get remaining nodes
    remained_nodes = np.setdiff1d(np.arange(num_nodes), deleted_nodes)

    # Initialize result matrix
    katz_matrix = torch.zeros((len(deleted_nodes), len(remained_nodes)), dtype=torch.float16, device=device)

    # For each deleted node
    for i, source in enumerate(deleted_nodes):
        # Initialize vector with 1 at source node
        x = torch.zeros((num_nodes), dtype=torch.float32, device=device)
        x[source] = 1.0

        # Power iteration
        score = x.clone()  # Initialize with identity contribution
        current_term = x.clone()

        for _ in range(max_iter):
            # Perform multiplication in CSR format
            current_term = beta * torch.sparse.mm(A, current_term)
            score = score + current_term  # Addition preserves CSR format

            # Check convergence using norm
            if torch.norm(current_term) < tolerance:
                break

        # Store only the values for remaining nodes
        katz_matrix[i] = score[0, remained_nodes]

    return katz_matrix.cpu().numpy()

def select_nodes_by_katz_threshold_no_matrix(data, source_nodes, target_nodes, threshold, beta=0.1, max_path_length=100, batch_size=100):
    """
    Select nodes from target_nodes whose Katz scores with source_nodes are below threshold
    
    Args:
        edge_index: Edge list as (2, E) tensor or array
        num_nodes: Number of nodes in graph
        source_nodes: List of source nodes
        target_nodes: List of target nodes to filter
        threshold: Maximum Katz score threshold
        beta: Attenuation factor
        max_path_length: Maximum path length to consider
        batch_size: Size of batches for processing
    
    Returns:
        filtered_targets: List of target nodes that meet the threshold criterion
        scores: Dictionary of Katz scores for filtered nodes
    """
     # Convert inputs to numpy arrays for faster processing
    num_nodes = data.num_nodes
    rows = data.edge_index[0].cpu().numpy()
    cols = data.edge_index[1].cpu().numpy()
    
    if torch.is_tensor(source_nodes):
        source_nodes = source_nodes.cpu().numpy()
    if torch.is_tensor(target_nodes):
        target_nodes = target_nodes.cpu().numpy()

    # Create adjacency list using dictionary for faster lookup
    adj_list = defaultdict(list)
    for i in range(len(rows)):
        adj_list[rows[i]].append(cols[i])
    
    # Precompute beta powers
    beta_powers = np.array([beta ** i for i in range(max_path_length + 1)])

    def compute_paths_batch(sources, targets, threshold):
        """
        Compute Katz scores for a batch of source-target pairs
        Returns scores and a mask indicating which pairs are below threshold
        """
        scores = np.zeros(len(targets))
        paths_dict = [{s: 1} for s in sources]
        
        # Add direct connections (length 0)
        for i, (s, t) in enumerate(zip(sources, targets)):
            if s == t:
                scores[i] = 1
        
        # Early return for nodes that already exceed threshold
        mask = scores <= threshold
        if not np.any(mask):
            return scores, mask

        # Iterate through path lengths
        for length in range(1, max_path_length + 1):
            new_paths = []
            for paths in paths_dict:
                next_paths = defaultdict(int)
                for node, count in paths.items():
                    for neighbor in adj_list[node]:
                        next_paths[neighbor] += count
                new_paths.append(next_paths)
            
            # Update scores for current length
            for i, (paths, t) in enumerate(zip(new_paths, targets)):
                if mask[i]:  # Only process pairs still below threshold
                    contribution = beta_powers[length] * paths.get(t, 0)
                    scores[i] += contribution
                    if scores[i] > threshold:
                        mask[i] = False
            
            # Early stopping if all pairs exceed threshold
            if not np.any(mask):
                break
                
            paths_dict = new_paths

        return scores, mask

    filtered_targets = []
    scores_dict = {}

    # Process target nodes in batches
    for i in tqdm(range(0, len(target_nodes), batch_size)):
        batch_targets = target_nodes[i:i + batch_size]
        batch_sources = np.repeat(source_nodes, len(batch_targets))
        batch_targets_rep = np.tile(batch_targets, len(source_nodes))
        
        # Compute scores for all source-target pairs in batch
        batch_scores, mask = compute_paths_batch(batch_sources, batch_targets_rep, threshold)
        
        # Reshape scores to matrix form (sources × targets)
        scores_matrix = batch_scores.reshape(len(source_nodes), len(batch_targets))
        
        # Find maximum score for each target across all sources
        max_scores = np.max(scores_matrix, axis=0)
        below_threshold = max_scores <= threshold
        
        # Store results for filtered nodes
        filtered_indices = np.where(below_threshold)[0]
        filtered_targets.extend(batch_targets[filtered_indices])
        scores_dict.update({batch_targets[i]: max_scores[i] for i in filtered_indices})

        # Clear some memory
        del batch_scores, scores_matrix
        gc.collect()

    return filtered_targets, scores_dict

def find_nodes_below_katz_threshold_batched(edge_index, num_nodes, source_nodes, threshold, 
                                          beta=0.1, max_iter=10, tolerance=1e-6, batch_size=100):
    """
    Find nodes whose Katz indices are smaller than the given threshold for multiple source nodes
    
    Args:
        edge_index: Edge list of shape (2, num_edges)
        num_nodes: Total number of nodes in the graph
        source_nodes: List or array of source nodes to calculate Katz indices from
        threshold: Threshold value for filtering nodes
        beta: Damping factor (default: 0.1)
        max_iter: Maximum number of iterations (default: 10)
        tolerance: Convergence tolerance (default: 1e-6)
        batch_size: Number of source nodes to process at once (default: 100)
    
    Returns:
        filtered_nodes_dict: Dictionary mapping source nodes to their filtered nodes
        katz_scores_dict: Dictionary mapping source nodes to dictionaries of Katz scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sparse adjacency matrix
    rows, cols = edge_index
    data = torch.ones(len(rows), dtype=torch.float16, device=device)
    
    if isinstance(rows, np.ndarray):
        rows = rows.astype(int)
        cols = cols.astype(int)
    elif isinstance(rows, torch.Tensor):
        rows = rows.long()
        cols = cols.long()
    
    indices = torch.stack([rows, cols], dim=0)
    A = torch.sparse_coo_tensor(indices, data, size=(num_nodes, num_nodes), device=device)

    filtered_nodes_dict = {}
    katz_scores_dict = {}
    
    # Convert source_nodes to numpy array if it's not already
    if torch.is_tensor(source_nodes):
        source_nodes = source_nodes.cpu().numpy()
    elif not isinstance(source_nodes, np.ndarray):
        source_nodes = np.array(source_nodes)

    # Process in batches
    for batch_start in tqdm(range(0, len(source_nodes), batch_size)):
        batch_end = min(batch_start + batch_size, len(source_nodes))
        batch_source_nodes = source_nodes[batch_start:batch_end]
        current_batch_size = len(batch_source_nodes)

        # Initialize batch vectors
        X = torch.zeros((current_batch_size, num_nodes), dtype=torch.float16, device=device)
        for i, node in enumerate(batch_source_nodes):
            X[i, node] = 1.0

        # Power iteration for batch
        scores = X.clone()
        current_term = X.clone()

        for _ in range(max_iter):
            current_term = beta * torch.sparse.mm(A, current_term.t()).t()
            scores = scores + current_term

            if torch.norm(current_term) < tolerance:
                break

        # Process results for each source node in batch
        scores_np = scores.cpu().numpy()
        
        for i, source_node in enumerate(batch_source_nodes):
            node_scores = scores_np[i]
            filtered_nodes = np.where(node_scores < threshold)[0]
            
            # Store results
            filtered_nodes_dict[int(source_node)] = filtered_nodes.tolist()
            katz_scores_dict[int(source_node)] = {
                int(node): float(node_scores[node]) 
                for node in filtered_nodes
            }

        # Clear GPU memory
        del scores, current_term, X
        torch.cuda.empty_cache()

    return filtered_nodes_dict, katz_scores_dict
    

def select_nodes_by_katz_threshold(edge_index, num_nodes, deleted_nodes, remained_nodes, threshold, beta=0.1, max_iter=10, tolerance=1e-6, batch_size=100):
    """
    !! This code may not produce the expected results. Please skip this code. 2025.3.30.
    
    Select nodes from remained_nodes whose minimum Katz scores are:
    1. Lower than threshold
    2. Not zero
    Removes filtered nodes from remained_nodes after each batch to reduce computation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to set for faster operations
    if torch.is_tensor(remained_nodes):
        remained_nodes = set(remained_nodes.cpu().numpy())
    else:
        remained_nodes = set(remained_nodes)
        
    # Store all filtered nodes across batches
    filtered_nodes = set()
    
    rows, cols = edge_index
    data = torch.ones(len(rows), dtype=torch.float32, device=device)
    
    for batch_idx in tqdm(range(0, len(deleted_nodes), batch_size)):
        if len(remained_nodes) == 0:
            break
            
        # Get current batch of deleted nodes
        batch_end = min(batch_idx + batch_size, len(deleted_nodes))
        batch_nodes = deleted_nodes[batch_idx:batch_end]
        
        # Filter edges to only include current remained nodes
        mask = np.isin(cols.cpu().numpy(), list(remained_nodes))
        current_rows = rows[mask]
        current_cols = cols[mask]
        current_data = data[mask]
        
        # Create sparse matrix for current remained nodes
        indices = torch.stack([current_rows, current_cols], dim=0)
        print("sum(current_rows), sum(current_data), indices.shape:", sum(current_rows), sum(current_data), indices.shape)
        A = torch.sparse_coo_tensor(indices, current_data, 
                                  size=(len(current_rows), len(current_cols)), 
                                  device=device)

        # Initialize vectors for batch nodes
        X = torch.zeros((len(batch_nodes), len(current_cols)), 
                       dtype=torch.float32, device=device)
        for i, node in enumerate(batch_nodes):
            X[i, node] = 1.0

        # Power iteration
        scores = X.clone()
        current_terms = X.clone()
        print("current_terms.shape, A.shape:", current_terms.shape, A.shape)
        for _ in range(max_iter):
            current_terms = beta * torch.sparse.mm(A, current_terms.t()).t()
            scores = scores + current_terms

            if torch.norm(current_terms) < tolerance:
                break
        
        # Get scores for remained nodes and filter
        remained_list = list(remained_nodes)
        remained_idx = torch.tensor(remained_list, device=device)
        batch_scores = scores[:, remained_idx]
        
        # Find nodes with min scores lower than threshold and not zero
        
        # Replace zeros with infinity
        batch_scores_no_zeros = torch.where(batch_scores == 0, torch.tensor(float('inf'), dtype=batch_scores.dtype, device=device), batch_scores)
        # print("batch_score.shape:", batch_scores.shape, "min(batch_scores_no_zeros):", min(batch_scores_no_zeros))
        min_scores = torch.min(batch_scores_no_zeros, dim=0)[0]
        
        mask = min_scores < threshold
        # min_scores = torch.min(batch_scores, dim=1)[0]
        
        # Add filtered nodes from this batch
        current_filtered = set(np.array(remained_list)[mask.cpu().numpy()])
        filtered_nodes.update(current_filtered)
        
        # Remove filtered nodes from remained_nodes
        remained_nodes -= current_filtered
        
        print(f"Batch {batch_idx//batch_size + 1}: "
              f"Filtered nodes in batch: {len(current_filtered)}, "
              f"Total filtered nodes: {len(filtered_nodes)}, "
              f"Remaining nodes: {len(remained_nodes)}")
        
        # Clear GPU memory
        del scores, current_terms, A, X, batch_scores
        torch.cuda.empty_cache()

    return list(filtered_nodes)

# Helper function to get specific rows from sparse matrix
def get_sparse_rows(sparse_matrix, row_indices):
    """
    Efficiently select specific rows from sparse matrix
    """
    sparse_matrix = sparse_matrix.coalesce()
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()
    
    # Convert row_indices to tensor if it's not already
    if not torch.is_tensor(row_indices):
        row_indices = torch.tensor(row_indices, device=indices.device)
    
    # Ensure long dtype for indexing
    row_indices = row_indices.long()
    
    # Find elements in the specified rows
    row_mask = torch.isin(indices[0], row_indices)
    selected_indices = indices[:, row_mask]
    selected_values = values[row_mask]
    
    # Adjust row indices to new positions
    row_map = {idx.item(): new_idx for new_idx, idx in enumerate(row_indices)}
    selected_indices[0] = torch.tensor(
        [row_map[idx.item()] for idx in selected_indices[0]], 
        device=indices.device
    )
    
    # Create new sparse tensor
    return torch.sparse_coo_tensor(
        indices=selected_indices,
        values=selected_values,
        size=(len(row_indices), sparse_matrix.size(1)),
        device=sparse_matrix.device
    ).coalesce()
    
def calculate_katz_scores_gpu_batched(edge_index, num_nodes, deleted_nodes, beta=0.1, max_iter=10, tolerance=1e-6, chunk_size=512):
    """
    Calculate Katz centrality for large graphs using sparse matrices
    (I + βA + β²A² + β³A³ + ...) * 1
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rows, cols = edge_index
    data = torch.ones(len(rows), dtype=torch.float32, device=device)
    
    # Get remaining nodes
    remained_nodes = np.setdiff1d(np.arange(num_nodes), deleted_nodes)
    
    if isinstance(rows, np.ndarray):
        rows = rows.astype(int)
        cols = cols.astype(int)
    elif isinstance(rows, torch.Tensor):
        rows = rows.long()
        cols = cols.long()
        
    indices = torch.stack([rows, cols], dim=0)
    # Keep as COO format instead of converting to CSR
    A = torch.sparse_coo_tensor(indices, data, size=(num_nodes, num_nodes), device=device)

    # Initialize X with rows from A corresponding to deleted nodes
    X = get_sparse_rows(A, deleted_nodes)  # This gets the rows we need
    current_term = X.clone()  # Start with these rows
    
    # Create mapping from original node indices to row positions in X
    node_to_row = {node.item(): idx for idx, node in enumerate(deleted_nodes)}
    
    try:
        for i in range(max_iter):
            # Compute next term using chunked multiplication
            new_term = chunk_sparse_mm(current_term, A, chunk_size=chunk_size, beta = beta)
            print(A.shape, current_term.shape, new_term.shape, X.shape)
            new_term = new_term.coalesce()
            
            """
            # Scale by beta
            new_term = torch.sparse_coo_tensor(
                indices=new_term._indices(),
                values=new_term._values() * beta,
                size=new_term.size(),
                device=device
            ).coalesce()
            """
            # Update current term
            current_term = new_term
            
            # Add to result
            X = (X + current_term).coalesce()
            
            # Check convergence
            if torch.sqrt(torch.sum(current_term._values() ** 2)) < tolerance:
                break
            
            # Clear cache
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        print(f"Error during computation: {e}")
        # Try with smaller chunk size or fall back to CPU
        if chunk_size > 128:
            print(f"Retrying with smaller chunk size...")
            return calculate_katz_scores_gpu_batched(
                A, deleted_nodes, beta, max_iter, 
                tolerance, chunk_size=chunk_size//2
            )
        else:
            raise e
    
    # Convert to dense only for the final result
    # scores = X.to_dense()
    
    return X, node_to_row

def filter_sparse_columns(sparse_tensor, threshold, exclude_zeros=True):
    """
    Filter columns of a sparse tensor that contain values below threshold but not zero
    
    Args:
        sparse_tensor: coalesced sparse tensor
        threshold: maximum value threshold
        exclude_zeros: whether to exclude zero values
    
    Returns:
        filtered_col_indices: indices of columns meeting criteria
    """
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    # Get column indices and their corresponding values
    col_indices = indices[1]
    
    if exclude_zeros:
        # Find values that are non-zero and below threshold
        mask = (values != 0) & (values < threshold)
    else:
        # Find values below threshold
        mask = values < threshold
    
    # Get unique columns that have values meeting our criteria
    filtered_cols = torch.unique(col_indices[mask])
    
    return filtered_cols

def sample_quantile(values, percentile, sample_size=100000):
    """
    Calculate approximate quantile using random sampling
    """
    if len(values) <= sample_size:
        return torch.quantile(values, percentile)
    
    # Random sampling
    indices = torch.randperm(len(values))[:sample_size]
    sampled_values = values[indices]
    
    return torch.quantile(sampled_values, percentile)

def filter_sparse_columns_by_percentile(sparse_tensor, percentile, exclude_zeros=True):
    """
    Filter columns of a sparse tensor using percentile threshold
    
    Args:
        sparse_tensor: coalesced sparse tensor
        percentile: percentile value (0-100) to use as threshold
        exclude_zeros: whether to exclude zero values
    
    Returns:
        filtered_col_indices: indices of columns meeting criteria
        threshold: the actual threshold value used
    """
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    # Get non-zero values if excluding zeros
    if exclude_zeros:
        values = values[values != 0]
    
    # Calculate threshold using percentile
    threshold = sample_quantile(values, percentile/100.0)
    
    # Get column indices and their corresponding values
    col_indices = indices[1]
    
    if exclude_zeros:
        # Find values that are non-zero and below threshold
        mask = (sparse_tensor._values() != 0) & (sparse_tensor._values() < threshold)
    else:
        # Find values below threshold
        mask = sparse_tensor._values() < threshold
    
    # Get unique columns that have values meeting our criteria
    filtered_cols = torch.unique(col_indices[mask])
    
    return filtered_cols, threshold

def save_sparse_tensor(sparse_tensor, filepath):
    """Save sparse tensor in COO format"""
    # Ensure the tensor is coalesced
    sparse_tensor = sparse_tensor.coalesce()
    
    # Save indices, values, and size separately
    torch.save({
        'indices': sparse_tensor._indices(),
        'values': sparse_tensor._values(),
        'size': sparse_tensor.size()
    }, filepath)

def load_sparse_tensor(filepath, device='cuda'):
    """Load sparse tensor from saved COO components"""
    data = torch.load(filepath, map_location=device)
    return torch.sparse_coo_tensor(
        indices=data['indices'],
        values=data['values'],
        size=data['size'],
        device=device
    ).coalesce()


def calculate_katz_index(edge_index, num_nodes, beta=0.1, batch_size=1000, max_iter=100, tolerance=1e-6):
    """
    Calculate Katz centrality for large graphs using sparse matrices
    (I + βA + β²A² + β³A³ + ...) * 1

    Args:
        edge_index: Graph edges as a 2xN array of source and target nodes
        num_nodes: Total number of nodes in the graph
        beta: Attenuation factor (should be less than 1/largest eigenvalue)
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        Katz centrality scores for each node
    """
    rows, cols = edge_index
    data = np.ones_like(rows)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    # Initialize result matrix
    katz_matrix = np.eye(num_nodes)
    
    # Process in batches
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        batch_indices = np.arange(start, end)
        
        # Initialize current batch
        X_batch = sparse.eye(num_nodes, format='csr')[:, batch_indices].toarray()
        current_term = sparse.eye(num_nodes, format='csr')[:, batch_indices].toarray()
        
        # Power series for this batch
        for _ in range(max_iter):
            current_term = beta * A.dot(current_term)
            X_batch += current_term
            
            if np.linalg.norm(current_term) < tolerance:
                break
        
        # Update the corresponding columns in the result matrix
        katz_matrix[:, start:end] = X_batch
    
    return katz_matrix

def get_katz_index(dataset_name, edge_index, num_nodes, beta = 0.1, use_chunk=True, use_sparse = True, chunk_size = 64, max_iter = 10, sparsity_threshold = 0.1):
    """
    Get Katz centrality index for large graphs or pre-computed index

    Args:
        edge_index: PyG edge_index tensor or numpy array of shape (2, num_edges)
        num_nodes: Number of nodes in the graph
        beta: Attenuation factor (should be less than 1/largest_eigenvalue)
        batch_size: Batch size for processing large matrices
        sparse: Whether to use sparse matrix operations (recommended for large graphs)

    Returns:
        katz_index: numpy array of Katz centrality scores
    """
    # Check if pre-computed index exists
    index_file = os.path.join(EXTRA_INFO_PATH, f"{dataset_name}_katz_index_{beta}.npy")
    
    if os.path.exists(index_file):
        # Load pre-computed index
        katz_index = np.load(index_file)
    else:
        # Calculate and save index
        if(beta == -1):
            beta = estimate_alpha(edge_index, num_nodes)
            print(f"Estimated alpha: {beta}")
        
        if(use_chunk == True):
            # Estimate safe beta value
            # largest_eigenvalue = estimate_largest_eigenvalue(edge_index, num_nodes)
            # beta = 0.9 / largest_eigenvalue  # Use smaller value than 1/largest_eigenvalue

            #katz_index = calculate_katz_index(edge_index = edge_index, num_nodes = num_nodes, beta = alpha)
            katz_index = calculate_katz_index_chunked(edge_index=edge_index, num_nodes = num_nodes, beta=beta, max_iter=max_iter, tolerance=1e-6, 
                                                      chunk_size = chunk_size, sparsity_threshold = sparsity_threshold)
        else:
            katz_index = calculate_katz_index_direct(edge_index, num_nodes, beta, use_sparse = use_sparse)
            
        save_sparse_tensor(katz_index, index_file)

    return katz_index

def calculate_katz_for_node_pairs(edge_index, num_nodes, deleted_nodes, beta=0.1, max_iter=10, tolerance=1e-6):
    """
    Calculate Katz index between deleted nodes and remaining nodes
    
    Args:
        edge_index: Edge list of shape (2, num_edges)
        num_nodes: Total number of nodes
        deleted_nodes: Array of nodes to be deleted
        beta: Damping factor
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        katz_matrix: shape (len(deleted_nodes), len(remained_nodes))
        remained_nodes: array of remaining node indices
    """
    # Create sparse adjacency matrix
    rows, cols = edge_index
    data = np.ones_like(rows)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    # Ensure A is in CSR format for efficient multiplication
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    elif not isinstance(A, sparse.csr_matrix):
        A = A.tocsr()
    
    # Get remaining nodes
    remained_nodes = np.setdiff1d(np.arange(num_nodes), deleted_nodes)
    
    # Initialize result matrix
    katz_matrix = sparse.lil_matrix((len(deleted_nodes), len(remained_nodes)))
    
    # For each deleted node
    for i, source in enumerate(deleted_nodes):
        # Initialize vector with 1 at source node as LIL
        x = sparse.lil_matrix((1, num_nodes))
        x[0, source] = 1.0
        x = x.tocsr()  # Convert to CSR once for multiplications

        # Power iteration
        score = x.copy()  # Initialize with identity contribution
        current_term = x.copy()
        
        for _ in range(max_iter):
            # Perform multiplication in CSR format
            current_term = beta * current_term.dot(A)
            score = score + current_term  # Addition preserves CSR format
            
            # Check convergence using sparse norm
            if sparse.linalg.norm(current_term) < tolerance:
                break
        
        # Store only the values for remaining nodes
        katz_matrix[i] = score[0, remained_nodes]
    
    katz_matrix = katz_matrix.tocsr()
    
    return katz_matrix

def select_influence_range_by_katz(katz_index, deleted_nodes, thres = -1):
    """
    Select influence range by Katz centrality index
    """
    
    if(thres >= 1):
        # remove zeros
        katz_index_flatten = katz_index.flatten()
        katz_index_flatten = katz_index_flatten[katz_index_flatten > 0]
        
        mythres = np.percentile(katz_index_flatten, thres)
    else:
        mythres = thres
    
    # Find indices where Katz values are below threshold
    affected_indices = np.nonzero(np.any(katz_index < mythres, axis=0))[0]

    return affected_indices, mythres

def save_katz_index(edge_index, num_nodes, output_file, alpha=0.1):
    """
    Calculate and save Katz index to a .npy file
    
    Args:
        edge_index: PyG edge_index tensor or numpy array
        num_nodes: Number of nodes in the graph
        output_file: Path to save the .npy file
        alpha: Attenuation factor
    """
    # Calculate Katz index
    katz_scores = calculate_katz_index(edge_index, num_nodes, alpha)
    
    # Save to file
    np.save(output_file, katz_scores)
    
    return katz_scores

def load_katz_index(file_path):
    """
    Load pre-computed Katz index from .npy file
    """
    return np.load(file_path)

def estimate_alpha(edge_index, num_nodes, method='power_iteration', max_iter=100):
    """
    Estimate appropriate alpha value based on largest eigenvalue
    """
    # Convert edge_index to numpy if it's a tensor
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    # Create sparse adjacency matrix
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), 
         (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Make symmetric for undirected graph
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    if method == 'power_iteration':
        # Power iteration method to estimate largest eigenvalue
        x = np.random.rand(num_nodes)
        for _ in range(max_iter):
            x_new = adj.dot(x)
            x_new = x_new / np.linalg.norm(x_new)
            x = x_new
        
        largest_eigenvalue = abs(x.dot(adj.dot(x)))
        
    else:
        # Use scipy's eigsh for small graphs
        from scipy.sparse.linalg import eigsh
        largest_eigenvalue = eigsh(adj, k=1, return_eigenvectors=False)[0]
    
    # Return alpha slightly less than 1/largest_eigenvalue
    return 0.9 / largest_eigenvalue

# For very large graphs, use batched processing:
def calculate_katz_large_graph(edge_index, num_nodes, alpha=0.1, batch_size=1000):
    """
    Calculate Katz index for very large graphs using batched processing
    """
    # Convert edge_index to numpy if it's a tensor
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    # Create sparse adjacency matrix
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), 
         (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Make symmetric for undirected graph
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Initialize result vector
    katz_index = np.zeros(num_nodes)
    
    # Process in batches
    for i in range(0, num_nodes, batch_size):
        end_idx = min(i + batch_size, num_nodes)
        batch_size_current = end_idx - i
        
        # Solve for this batch
        identity = sp.eye(batch_size_current)
        batch_adj = adj[i:end_idx, i:end_idx]
        batch_solution = spsolve(
            identity - alpha * batch_adj,
            np.ones(batch_size_current)
        )
        
        katz_index[i:end_idx] = batch_solution
    
    # Normalize scores
    katz_index = katz_index / np.linalg.norm(katz_index)
    
    return katz_index

# Usage with progress tracking:
def calculate_and_save_katz_with_progress(edge_index, num_nodes, output_file, alpha=None):
    """
    Calculate and save Katz index with progress tracking
    """
    from tqdm import tqdm
    
    print("Estimating alpha value...")
    if alpha is None:
        alpha = estimate_alpha(edge_index, num_nodes)
    print(f"Using alpha = {alpha}")
    
    print("Calculating Katz index...")
    with tqdm(total=100) as pbar:
        katz_scores = calculate_katz_index(edge_index, num_nodes, alpha)
        pbar.update(50)
        
        print("Saving results...")
        np.save(output_file, katz_scores)
        pbar.update(50)
    
    print(f"Katz index saved to {output_file}")
    return katz_scores


def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index) + 1
    feature_count = max(feature_index) + 1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    return features


def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"]).reshape(-1, 1)
    return target


def make_adjacency(graph, max_degree, sel=None):
    all_nodes = np.array(graph.nodes())

    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes + 1, max_degree)) + n_nodes).astype(int)

    if sel is not None:
        all_nodes = all_nodes[sel]

    for node in tqdm(all_nodes):
        neibs = np.array(list(graph.neighbors(node)))

        if sel is not None:
            neibs = neibs[sel[neibs]]

        if len(neibs) > 0:
            if len(neibs) > max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=False)
            elif len(neibs) < max_degree:
                extra = np.random.choice(neibs, max_degree - neibs.shape[0], replace=True)
                neibs = np.concatenate([neibs, extra])
            adj[node, :] = neibs

    return adj


def connected_component_subgraphs(graph):
    """
    Find all connected subgraphs in a networkx Graph

    Args:
        graph (Graph): A networkx Graph

    Yields:
        generator: A subgraph generator
    """
    for c in nx.connected_components(graph):
        yield graph.subgraph(c)


def check_exist(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise


def filter_edge_index(edge_index, node_indices, reindex=True):
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index


def pyg_to_nx(data):
    """
    Convert a torch geometric Data to networkx Graph.

    Args:
        data (Data): A torch geometric Data.

    Returns:
        Graph: A networkx Graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(data.num_nodes))
    edge_index = data.edge_index.numpy()

    for u, v in np.transpose(edge_index):
        graph.add_edge(u, v)

    return graph


def edge_index_to_nx(edge_index, num_nodes):
    """
    Convert a torch geometric Data to networkx Graph by edge_index.
    Args:
        edge_index (Data.edge_index): A torch geometric Data.
        num_nodes (int): Number of nodes in a graph.
    Returns:
        Graph: networkx Graph
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(num_nodes))
    edge_index = edge_index.numpy()

    for u, v in np.transpose(edge_index):
        graph.add_edge(u, v)

    return graph


def filter_edge_index_1(data, node_indices):
    """
    Remove unnecessary edges from a torch geometric Data, only keep the edges between node_indices.
    Args:
        data (Data): A torch geometric Data.
        node_indices (list): A list of nodes to be deleted from data.

    Returns:
        data.edge_index: The new edge_index after removing the node_indices.
    """
    if isinstance(data.edge_index, torch.Tensor):
        data.edge_index = data.edge_index.cpu()

    edge_index = data.edge_index
    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = data.edge_index[:, col_index]

    return np.searchsorted(node_indices, edge_index)

def random_walk_influence(edge_index, num_nodes, deleted_nodes, 
                         num_walks=10, walk_length=3, p=0.2):
    """
    Perform random walks from deleted nodes to find influenced node range
    
    Args:
        edge_index: Edge list of shape (2, num_edges)
        num_nodes: Total number of nodes
        deleted_nodes: List/array of nodes to be deleted
        num_walks: Number of random walks per node
        walk_length: Length of each random walk
        p: Probability of returning to previous node
        
    Returns:
        influenced_nodes: Set of nodes influenced by deleted nodes
        visit_counts: Dictionary of visit counts for each node
    """
    # Convert edge_index to adjacency list for faster neighbor lookup
    adj_list = [[] for _ in range(num_nodes)]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # Remove if directed graph
    
    # Convert deleted_nodes to set for O(1) lookup
    deleted_set = set(deleted_nodes)
    influenced_nodes = set()
    visit_counts = defaultdict(int)
    
    def single_random_walk(start_node):
        """Perform single random walk"""
        path = [start_node]
        current = start_node
        prev = None
        
        for _ in range(walk_length):
            neighbors = adj_list[current]
            if not neighbors:
                break
                
            # Probability of returning to previous node
            if prev is not None and np.random.random() < p:
                next_node = prev
            else:
                next_node = np.random.choice(neighbors)
            
            path.append(next_node)
            prev = current
            current = next_node
            
        return path
    
    # Perform random walks from each deleted node
    for node in tqdm(deleted_nodes, desc="Random walks"):
        for _ in range(num_walks):
            path = single_random_walk(node)
            
            # Update influenced nodes and visit counts
            for visited_node in path:
                if visited_node not in deleted_set:
                    influenced_nodes.add(visited_node)
                    visit_counts[visited_node] += 1
    
    return influenced_nodes, visit_counts

def analyze_influence_distribution(visit_counts, percentile_thresholds=[25, 50, 75, 90]):
    """
    Analyze the distribution of influence based on visit counts
    """
    counts = np.array(list(visit_counts.values()))
    
    stats = {
        'mean': np.mean(counts),
        'median': np.median(counts),
        'std': np.std(counts),
        'min': np.min(counts),
        'max': np.max(counts),
        'percentiles': {
            p: np.percentile(counts, p) 
            for p in percentile_thresholds
        }
    }
    
    return stats

def filter_by_visit_threshold(visit_counts, threshold_percentile=75):
    """
    Filter nodes based on visit count threshold
    """
    counts = np.array(list(visit_counts.values()))
    threshold = np.percentile(counts, threshold_percentile)
    
    filtered_nodes = {
        node for node, count in visit_counts.items() 
        if count >= threshold
    }
    
    return filtered_nodes, threshold

def parallel_random_walks(edge_index, num_nodes, deleted_nodes, 
                         num_walks=10, walk_length=3, p=0.2, 
                         num_workers=4):
    """
    Parallel implementation of random walks
    """
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    
    # Split deleted nodes into chunks for parallel processing
    chunk_size = len(deleted_nodes) // num_workers
    node_chunks = [
        deleted_nodes[i:i + chunk_size] 
        for i in range(0, len(deleted_nodes), chunk_size)
    ]
    
    # Prepare worker function
    worker_fn = partial(
        random_walk_influence, 
        edge_index, num_nodes,
        num_walks=num_walks,
        walk_length=walk_length,
        p=p
    )
    
    # Run parallel walks
    all_influenced = set()
    all_counts = defaultdict(int)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker_fn, node_chunks))
        
        # Combine results
        for influenced, counts in results:
            all_influenced.update(influenced)
            for node, count in counts.items():
                all_counts[node] += count
    
    return all_influenced, all_counts

def visualize_influence(G, deleted_nodes, influenced_nodes, visit_counts):
    """
    Visualize influence spread using networkx
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create position layout
    pos = nx.spring_layout(G)
    
    # Draw network
    plt.figure(figsize=(12, 8))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Draw nodes with different colors and sizes
    node_colors = []
    node_sizes = []
    
    max_count = max(visit_counts.values()) if visit_counts else 1
    
    for node in G.nodes():
        if node in deleted_nodes:
            color = 'red'
            size = 300
        elif node in influenced_nodes:
            color = 'orange'
            size = 100 + 200 * (visit_counts.get(node, 0) / max_count)
        else:
            color = 'gray'
            size = 100
            
        node_colors.append(color)
        node_sizes.append(size)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes)
    
    plt.title("Influence Spread from Deleted Nodes")
    plt.axis('off')
    plt.show()

# Example usage:
def demonstrate_random_walk_influence():
    # Create sample graph
    n = 1000
    m = 5000
    
    # Generate random edges
    edge_index = torch.randint(0, n, (2, m))
    
    # Select some deleted nodes
    deleted_nodes = np.random.choice(n, size=10, replace=False)
    
    # Find influenced nodes
    influenced_nodes, visit_counts = random_walk_influence(
        edge_index, n, deleted_nodes,
        num_walks=20,
        walk_length=5,
        p=0.2
    )
    
    # Analyze results
    stats = analyze_influence_distribution(visit_counts)
    print("\nInfluence Statistics:")
    print(f"Total influenced nodes: {len(influenced_nodes)}")
    print(f"Mean visits: {stats['mean']:.2f}")
    print(f"Median visits: {stats['median']:.2f}")
    print(f"90th percentile: {stats['percentiles'][90]:.2f}")
    
    # Filter highly influenced nodes
    high_influence, threshold = filter_by_visit_threshold(
        visit_counts, threshold_percentile=90
    )
    print(f"\nHighly influenced nodes: {len(high_influence)}")
    print(f"Visit threshold: {threshold:.2f}")
    
    return influenced_nodes, visit_counts, stats

# For large graphs:
def process_large_graph(edge_index, num_nodes, deleted_nodes, num_walks = 50, walk_length = 10, p =0.2, num_works = 4):
    """
    Process large graphs with memory efficiency
    """
    # Use parallel processing
    influenced_nodes, visit_counts = parallel_random_walks(
        edge_index, num_nodes, deleted_nodes,
        num_walks = num_walks,
        walk_length = walk_length,
        p = p,
        num_workers = num_works
    )
    
    # Analyze in batches
    batch_size = 10000
    stats = {}
    
    for i in range(0, len(influenced_nodes), batch_size):
        batch_nodes = list(influenced_nodes)[i:i + batch_size]
        batch_counts = {
            node: visit_counts[node] 
            for node in batch_nodes
        }
        batch_stats = analyze_influence_distribution(batch_counts)
        
        # Update overall stats
        for key, value in batch_stats.items():
            if key not in stats:
                stats[key] = []
            stats[key].append(value)
    
    # Combine batch statistics
    final_stats = {
        'mean': np.mean([s['mean'] for s in stats['mean']]),
        'median': np.median([s['median'] for s in stats['median']]),
        'max': max([s['max'] for s in stats['max']])
    }
    
    return influenced_nodes, visit_counts, final_stats



def create_efficient_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[np.ndarray]:
    """
    Create memory-efficient adjacency list using numpy arrays
    """
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    # Initialize list of empty arrays
    adj_list = [None] * num_nodes
    
    # Count neighbors for each node
    neighbor_counts = np.zeros(num_nodes, dtype=np.int32)
    np.add.at(neighbor_counts, edge_index[0], 1)
    
    # Pre-allocate arrays
    for node in range(num_nodes):
        adj_list[node] = np.zeros(neighbor_counts[node], dtype=np.int32)
    
    # Fill arrays
    current_idx = np.zeros(num_nodes, dtype=np.int32)
    for src, dst in zip(edge_index[0], edge_index[1]):
        adj_list[src][current_idx[src]] = dst
        current_idx[src] += 1
    
    return adj_list

def process_node_batch_single_time(args: Tuple) -> Dict:
    """
    Process a batch of nodes in parallel
    """
    node_batch, adj_list, walk_length, p = args
    
    visit_counts = defaultdict(lambda: defaultdict(int))
    
    def single_random_walk(start_node: int) -> List[int]:
        path = [start_node]
        current = start_node
        prev = None
        
        for _ in range(walk_length):
            neighbors = adj_list[current]
            if len(neighbors) == 0:
                break
                
            if prev is not None and np.random.random() < p:
                next_node = prev
            else:
                next_node = np.random.choice(neighbors)
            
            path.append(next_node)
            prev = current
            current = next_node
            
        return path
    
    for node in node_batch:
        path = single_random_walk(node)
        # Use numpy for efficient counting
        unique_nodes, counts = np.unique(path[1:], return_counts=True)
        for visited_node, count in zip(unique_nodes, counts):
            visit_counts[node][visited_node] = int(count)
    
    return dict(visit_counts)

def process_node_batch(args: Tuple) -> Dict:
    """
    Process a batch of nodes in parallel, performing K random walks for each node
    """
    node_batch, adj_list, walk_length, p, num_walks = args  # Added num_walks parameter
    
    visit_counts = defaultdict(lambda: defaultdict(int))
    
    def single_random_walk(start_node: int) -> List[int]:
        path = [start_node]
        current = start_node
        prev = None
        
        for _ in range(walk_length):
            neighbors = adj_list[current]
            if len(neighbors) == 0:
                break
                
            if prev is not None and np.random.random() < p:
                next_node = prev
            else:
                next_node = np.random.choice(neighbors)
            
            path.append(next_node)
            prev = current
            current = next_node
            
        return path
    
    for node in node_batch:
        # Perform K random walks for each node
        all_paths = []
        for _ in range(num_walks):
            path = single_random_walk(node)
            all_paths.extend(path[1:])  # Exclude starting node
        
        # Count visits across all K walks
        unique_nodes, counts = np.unique(all_paths, return_counts=True)
        for visited_node, count in zip(unique_nodes, counts):
            visit_counts[node][visited_node] = int(count)
    
    return dict(visit_counts)

def generate_and_save_walk_counts_parallel(
    edge_index: torch.Tensor,
    num_nodes: int, 
    output_dir: str,
    dataset_name: str,
    walk_length: int = 3,
    p: float = 0.2,
    batch_size: int = 10000,
    num_workers: int = 4,
    num_walk_per_node: int = 30, 
    memory_limit: float = 0.75  # Maximum fraction of system memory to use
):
    """
    Generate and save random walk counts for large graphs using parallel processing
    """
    
    print(f"Dataset: {dataset_name}, Walks/Node: {num_walk_per_node}, Batch Size: {batch_size}, Walk Length: {walk_length}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # if exist os.path.join(output_dir, f'{dataset_name}_metadata.pkl'), return directly
    if(os.path.exists(os.path.join(output_dir, f'{dataset_name}_metadata.pkl'))):
        return
    
    # Create efficient adjacency list
    adj_list = create_efficient_adj_list(edge_index, num_nodes)
    
    # Clear edge_index from memory if possible
    del edge_index
    gc.collect()
    
    # Calculate batch size based on available memory
    total_memory = psutil.virtual_memory().total
    available_memory = total_memory * memory_limit
    estimated_memory_per_node = walk_length * 8  # Rough estimate in bytes
    max_batch_size = int(available_memory / (estimated_memory_per_node * num_workers))
    batch_size = min(batch_size, max_batch_size)
    
    print(f"Using batch size: {batch_size}")
    
    # Process nodes in batches
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for start_idx in tqdm(range(0, num_nodes, batch_size), desc="Processing node batches"):
            end_idx = min(start_idx + batch_size, num_nodes)
            
            # Create smaller batches for parallel processing
            sub_batch_size = batch_size // num_workers
            sub_batches = []
            
            for sub_start in range(start_idx, end_idx, sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, end_idx)
                sub_batches.append(
                    (range(sub_start, sub_end), adj_list, walk_length, p, num_walk_per_node)
                )
            
            # Process sub-batches in parallel
            results = list(executor.map(process_node_batch, sub_batches))
            
            # Combine results
            batch_counts = {}
            for result in results:
                batch_counts.update(result)
            
            # Save batch results
            batch_filename = os.path.join(output_dir, f'walkcounts_{dataset_name}_{start_idx}_{end_idx}.pkl')
            with open(batch_filename, 'wb') as f:
                pickle.dump(batch_counts, f)
            
            # Clear memory
            del batch_counts
            gc.collect()
    
    # Save metadata
    metadata = {
        'num_nodes': num_nodes,
        'walk_length': walk_length,
        'p': p,
        'batch_size': batch_size
    }
    
    with open(os.path.join(output_dir, f'{dataset_name}_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return

def load_walk_counts_efficient(
    input_dir: str,
    dataset_name: str,
    nodes: List[int] = None,
    batch_size: int = 100000
) -> Tuple[Dict, Dict]:
    """
    Memory-efficient loading of walk counts
    """
    with open(os.path.join(input_dir, f'{dataset_name}_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    walk_counts = defaultdict(lambda: defaultdict(int))
    batch_files = sorted([f for f in os.listdir(input_dir) if f.startswith(f'walkcounts_{dataset_name}_')])
    
    for filename in tqdm(batch_files, desc="Loading walk counts"):
        filepath = os.path.join(input_dir, filename)
        # remove the affix of the filepath
        
        # Extract batch range from filename
        base_name = filename.split('.')[0]
        start_idx, end_idx = map(int, base_name.split('_')[2:4])
        if nodes is not None:
            # Skip batch if no requested nodes in range
            if not any(start_idx <= node < end_idx for node in nodes):
                continue
        
        # Load and process batch
        with open(filepath, 'rb') as f:
            batch_counts = pickle.load(f)
        
        # Filter nodes if specified
        if nodes is not None:
            batch_counts = {k: v for k, v in batch_counts.items() if k in nodes}
        
        # Update walk_counts
        walk_counts.update(batch_counts)
        
        # Clear memory
        del batch_counts
        gc.collect()
    
    return dict(walk_counts), metadata

def analyze_walk_counts_efficient(
    walk_counts: Dict,
    threshold_percentile: float = 80,
    min_visits: int = 1
) -> Dict:
    """
    Memory-efficient analysis of walk counts
    """
    influenced_nodes = {}
    
    for source_node, counts in tqdm(walk_counts.items(), desc="Analyzing walk counts"):
        if not counts:
            influenced_nodes[source_node] = []
            continue
        
        # Convert to numpy array for efficient computation
        nodes = np.array(list(counts.keys()))
        values = np.array(list(counts.values()))
        
        # Filter by minimum visits
        mask = values >= min_visits
        nodes = nodes[mask]
        values = values[mask]
        
        if len(values) > 0:
            threshold = np.percentile(values, threshold_percentile)
            influenced = nodes[values >= threshold].tolist()
        else:
            influenced = []
            
        influenced_nodes[source_node] = influenced
    
    return influenced_nodes

def generate_walk_count(dataset_name, edge_index, num_nodes, output_dir = "", walk_length= 100, batch_size = 100000, num_walk_per_node = 30):
    
    if(output_dir == ""):
        output_dir = EXTRA_INFO_PATH
    
    # Generate and save walk counts
    generate_and_save_walk_counts_parallel(
        edge_index=edge_index,
        num_nodes=num_nodes,
        dataset_name = dataset_name,
        output_dir=output_dir,
        walk_length=walk_length,
        p=0.2,
        batch_size=batch_size,
        num_workers=4,
        num_walk_per_node = num_walk_per_node, 
        memory_limit=0.75
    )
    
    """
    # Load specific nodes efficiently
    specific_nodes = list(range(1000))  # Example: first 1000 nodes
    walk_counts, metadata = load_walk_counts_efficient(
        input_dir=output_dir,
        nodes=specific_nodes,
        batch_size=1000000
    )
    
    # Analyze results efficiently
    influenced_nodes = analyze_walk_counts_efficient(
        walk_counts=walk_counts,
        threshold_percentile=75,
        min_visits=1
    )
    
    # Print some statistics
    total_influenced = sum(len(nodes) for nodes in influenced_nodes.values())
    avg_influenced = total_influenced / len(influenced_nodes)
    logger.info(f"Average number of influenced nodes: {avg_influenced:.2f}")
    """

def select_approxi_method(deleted_nodes, importance_sampling_K, importance_sampling_N, random_walk_K, random_walk_l, random_walk_selection_rate):
    
    n_Du = len(deleted_nodes)
    importance_sampling_size = importance_sampling_K * importance_sampling_N
    random_walk_size = n_Du * random_walk_K * random_walk_l * (100 - random_walk_selection_rate)* 0.01
    method = "importancesampling" if importance_sampling_size < random_walk_size else "randomwalk"
    
    return method, importance_sampling_size, random_walk_size
    
def select_approxi_method_by_thres(node_classifier, deleted_nodes, metric = "pagerank", percentile_threshold = IMPORTANCE_THRESHOLD_FOR_METHOD_SELECTION):
    
    if metric.lower() == "pagerank":
        # Calculate PageRank for all nodes
        importance_scores = node_classifier.pagerank_weights
    elif metric.lower() == "degree":
        # Calculate degree centrality for all nodes
        importance_scores = node_classifier.degree_weights
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'pagerank' or 'degree'")

    # Convert deleted_nodes to numpy array if needed
    if isinstance(deleted_nodes, list):
        deleted_nodes = np.array(deleted_nodes)
    
    # Calculate threshold value
    threshold_value = np.percentile(importance_scores, percentile_threshold)
    
    # Get scores for deleted nodes
    deleted_scores = {node: importance_scores[node] for node in deleted_nodes}
    
    # Check if any deleted nodes are above threshold
    has_important_nodes = any(score >= threshold_value for score in deleted_scores.values())
    
    # Select method based on importance
    method = "importancesampling" if has_important_nodes else "randomwalk"
    
    num_important = sum(1 for score in deleted_scores.values() if score >= threshold_value)
    
    return method, num_important, threshold_value

def filter_less_important_nodes(node_classifier, unlearning_request, metric = "pagerank", percentile_threshold = IMPORTANCE_THRESHOLD_FOR_METHOD_SELECTION):
    
    if metric.lower() == "pagerank":
        # Calculate PageRank for all nodes
        importance_scores = node_classifier.pagerank_weights
    elif metric.lower() == "degree":
        # Calculate degree centrality for all nodes
        importance_scores = node_classifier.degree_weights
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'pagerank' or 'degree'")
    
    # Convert deleted_nodes to numpy array if needed
    if isinstance(unlearning_request, list):
        deleted_nodes = np.array(unlearning_request)
    else:
        deleted_nodes = unlearning_request
        
    # Calculate threshold value
    threshold_value = np.percentile(importance_scores, percentile_threshold)
    
    # Get scores for deleted nodes
    deleted_scores = {node: importance_scores[node] for node in deleted_nodes}
    
    # Filter nodes based on threshold
    filtered_nodes = [node for node, score in deleted_scores.items() if score < threshold_value]
    
    return filtered_nodes
    
def filter_less_important_edges(node_classifier, unlearning_request, metric = "pagerank", percentile_threshold = IMPORTANCE_THRESHOLD_FOR_METHOD_SELECTION):
    
    if metric.lower() == "pagerank":
        # Calculate PageRank for all nodes
        importance_scores = node_classifier.pagerank_weights
    elif metric.lower() == "degree":
        # Calculate degree centrality for all nodes
        importance_scores = node_classifier.degree_weights
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'pagerank' or 'degree'")
    
    # Convert to numpy if tensor
    if torch.is_tensor(unlearning_request):
        edges = unlearning_request.cpu().numpy()
    else:
        edges = unlearning_request
        
    # Calculate importance threshold (e.g. top 20% are considered important)
    threshold = np.percentile(importance_scores, percentile_threshold)
    
    # Get importance scores for nodes in the edges
    source_scores = importance_scores[edges[0]]
    target_scores = importance_scores[edges[1]]
    
    # Keep edges where both nodes are below threshold
    mask_unimportant = (source_scores < threshold) & (target_scores < threshold)
    filtered_edges = edges[:, mask_unimportant]
    
    # Convert back to tensor if input was tensor
    if torch.is_tensor(unlearning_request):
        filtered_edges = torch.from_numpy(filtered_edges)
        
    return filtered_edges

def analyze_deleted_nodes_influence(
    base_dir: str,
    dataset_name: str,
    deleted_nodes: List[int],
    custom_percentile: float = None
) -> Tuple[List[int], List[int], Dict]:
    """
    Analyze influence of deleted nodes using saved random walk statistics
    
    Args:
        base_dir: Directory containing walk count files
        dataset_name: Name of the dataset
        deleted_nodes: List of nodes to analyze
        custom_percentile: Optional custom percentile threshold
        
    Returns:
        high_influence_nodes: List of nodes with counts above threshold
        low_influence_nodes: List of nodes with counts below threshold
        influence_stats: Dictionary with detailed statistics
    """
    # Load dataset threshold
    stats = load_dataset_statistics(base_dir, dataset_name)
    threshold = stats['percentile_threshold']
    
    if custom_percentile is not None:
        print('calculating threshold with cunstom_percentile...')
        threshold = get_threshold_for_dataset(base_dir, dataset_name, custom_percentile)
    
    # Initialize containers
    node_total_counts = defaultdict(int)
    high_influence_nodes = []
    low_influence_nodes = []
    
    # Process walk count files
    batch_files = sorted([f for f in os.listdir(base_dir) 
                         if f.startswith(f'walkcounts_{dataset_name}_')])
    
    # Convert deleted_nodes to set for faster lookup
    deleted_set = set(deleted_nodes)
    
    for filename in tqdm(batch_files, desc=f"Processing {dataset_name} walk counts"):
        filepath = os.path.join(base_dir, filename)
        
        with open(filepath, 'rb') as f:
            batch_counts = pickle.load(f)
            
        # Only process counts for deleted nodes
        for node in deleted_set:
            if node in batch_counts:
                node_counts = batch_counts[node]
                total_count = sum(node_counts.values())
                node_total_counts[node] += total_count
    
    # Classify nodes based on threshold
    for node in deleted_nodes:
        total_count = node_total_counts[node]
        if total_count >= threshold:
            high_influence_nodes.append(node)
        else:
            low_influence_nodes.append(node)
    
    # Compile statistics
    influence_stats = {
        'dataset': dataset_name,
        'threshold': threshold,
        'num_deleted_nodes': len(deleted_nodes),
        'num_high_influence': len(high_influence_nodes),
        'num_low_influence': len(low_influence_nodes),
        'node_counts': dict(node_total_counts)
    }
    
    return high_influence_nodes, low_influence_nodes, influence_stats

def create_zero_mask_like(train_mask):
    """
    Create a zero mask with the same shape and device as the input mask
    
    Args:
        train_mask: Can be numpy array, torch.Tensor on CPU or CUDA
    
    Returns:
        sampled_train_mask: Zero mask with same type/device as input
    """
    # Case 1: If input is numpy array
    if isinstance(train_mask, np.ndarray):
        return np.zeros_like(train_mask, dtype=bool)
    
    # Case 2: If input is torch tensor
    elif torch.is_tensor(train_mask):
        # Check if tensor is on CUDA
        if train_mask.is_cuda:
            # Create zero tensor on same CUDA device
            return torch.zeros_like(train_mask, dtype=torch.bool, device=train_mask.device)
        else:
            # Create zero tensor on CPU
            return torch.zeros_like(train_mask, dtype=torch.bool)
    
    else:
        raise TypeError(f"Unsupported mask type: {type(train_mask)}")


def print_influence_analysis(influence_stats: Dict):
    """
    Print formatted analysis of node influence
    """
    print(f"\nInfluence Analysis for {influence_stats['dataset']}")
    print("-" * 50)
    print(f"Threshold value: {influence_stats['threshold']:.2f}")
    print(f"Total deleted nodes: {influence_stats['num_deleted_nodes']}")
    print(f"High influence nodes: {influence_stats['num_high_influence']}")
    print(f"Low influence nodes: {influence_stats['num_low_influence']}")
    
    # Print top 5 highest influence nodes
    sorted_nodes = sorted(
        influence_stats['node_counts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("\nTop 5 highest influence nodes:")
    for node, count in sorted_nodes[:5]:
        print(f"Node {node}: {count:.2f}")

def load_dataset_statistics(base_dir: str, dataset_name: str) -> Dict:
    """
    Load saved statistics for a specific dataset
    
    Args:
        base_dir: Directory containing the statistics files
        dataset_name: Name of the dataset to load statistics for
        
    Returns:
        Dictionary containing the dataset statistics
    """
    stats_file = os.path.join(base_dir, f'{dataset_name}_walk_statistics.json')
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
        return stats
    except FileNotFoundError:
        raise FileNotFoundError(f"Statistics file not found for dataset {dataset_name}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in statistics file for dataset {dataset_name}")

def load_multiple_dataset_statistics(base_dir: str, dataset_names: List[str] = None) -> Dict[str, Dict]:
    """
    Load statistics for multiple datasets or all available datasets
    
    Args:
        base_dir: Directory containing the statistics files
        dataset_names: List of dataset names to load (if None, loads all available)
        
    Returns:
        Dictionary of statistics for each dataset
    """
    if dataset_names is None:
        # Load the comparison file containing all datasets
        comparison_file = os.path.join(base_dir, 'dataset_comparison.json')
        try:
            with open(comparison_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # If comparison file doesn't exist, try to find individual statistics files
            pattern = os.path.join(base_dir, '*_walk_statistics.json')
            stat_files = glob.glob(pattern)
            dataset_names = [os.path.basename(f).replace('_walk_statistics.json', '') 
                           for f in stat_files]
    
    # Load statistics for specified datasets
    all_stats = {}
    for dataset in dataset_names:
        try:
            stats = load_dataset_statistics(base_dir, dataset)
            all_stats[dataset] = stats
        except Exception as e:
            print(f"Error loading statistics for {dataset}: {str(e)}")
            continue
            
    return all_stats

def get_threshold_for_dataset(base_dir: str, dataset_name: str, percentile: float = None) -> float:
    """
    Get the threshold value for a specific dataset and percentile
    
    Args:
        base_dir: Directory containing the statistics files
        dataset_name: Name of the dataset
        percentile: If provided, recalculate threshold for this percentile
                   If None, use the saved threshold
        
    Returns:
        Threshold value for the dataset
    """
    stats = load_dataset_statistics(base_dir, dataset_name)
    
    if percentile is None:
        return stats['percentile_threshold']
    else:
        # If a new percentile is requested, need to load and recalculate from raw data
        walk_counts_dir = base_dir  # Adjust if walk counts are stored elsewhere
        visit_distribution = []
        
        # Load walk counts and extract distribution
        batch_files = sorted([f for f in os.listdir(walk_counts_dir) 
                            if f.startswith(f'walkcounts_{dataset_name}_')])
        
        for filename in tqdm(batch_files, desc=f"Loading {dataset_name} walk counts"):
            filepath = os.path.join(walk_counts_dir, filename)
            with open(filepath, 'rb') as f:
                batch_counts = pickle.load(f)
                
            for counts in batch_counts.values():
                visit_distribution.extend(counts.values())
                
        return float(np.percentile(visit_distribution, percentile))
    
# Function to get statistics summary
def get_statistics_summary(stats: Dict) -> str:
    """
    Create a formatted summary of dataset statistics
    """
    summary = [
        f"Dataset: {stats['dataset']}",
        f"Total nodes: {stats['total_nodes']:,}",
        f"Total visits: {stats['total_visits']:,}",
        f"Unique visited nodes: {stats['unique_visited_nodes']:,}",
        f"Mean visits: {stats['mean_visits']:.2f}",
        f"Median visits: {stats['median_visits']:.2f}",
        f"80th percentile threshold: {stats['percentile_threshold']:.2f}",
        f"Standard deviation: {stats['std_visits']:.2f}",
        f"Range: {stats['min_visits']:.2f} - {stats['max_visits']:.2f}"
    ]
    return "\n".join(summary)


def get_influence_distribution(influence_stats: Dict) -> plt.Figure:
    """
    Create visualization of influence distribution
    """
    counts = list(influence_stats['node_counts'].values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=30, density=True)
    ax.axvline(
        influence_stats['threshold'], 
        color='r', 
        linestyle='--', 
        label='Threshold'
    )
    
    ax.set_title(f"Influence Distribution for {influence_stats['dataset']}")
    ax.set_xlabel('Total Visit Counts')
    ax.set_ylabel('Density')
    ax.legend()
    
    return fig

# Example usage:
def demonstrate_influence_analysis():
    base_dir = "path/to/your/walk_counts"
    dataset_name = "cora"
    deleted_nodes = [1, 5, 10, 15, 20]  # example nodes
    
    # Analyze influence
    high_influence, low_influence, stats = analyze_deleted_nodes_influence(
        base_dir=base_dir,
        dataset_name=dataset_name,
        deleted_nodes=deleted_nodes
    )
    
    # Print analysis
    print_influence_analysis(stats)
    
    print("\nHigh influence nodes:", high_influence)
    print("Low influence nodes:", low_influence)
    
    # Visualize distribution
    fig = get_influence_distribution(stats)
    plt.show()
    
    return stats

# For batch processing multiple sets of deleted nodes:
def batch_influence_analysis(
    base_dir: str,
    dataset_name: str,
    node_groups: List[List[int]],
    save_results: bool = True
) -> List[Dict]:
    """
    Analyze influence for multiple groups of deleted nodes
    """
    results = []
    
    for i, nodes in enumerate(node_groups):
        high_inf, low_inf, stats = analyze_deleted_nodes_influence(
            base_dir=base_dir,
            dataset_name=dataset_name,
            deleted_nodes=nodes
        )
        
        results.append(stats)
        
        if save_results:
            output_file = os.path.join(
                base_dir, 
                f'{dataset_name}_group_{i}_influence_analysis.json'
            )
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=4)
    
    return results


def select_approxi_method_with_global_randomwalk_stats(dataset_name, deleted_nodes, importance_sampling_K, importance_sampling_N):
    
    high_influence_nodes, low_influence_nodes, influence_stats = analyze_deleted_nodes_influence(
        base_dir=EXTRA_INFO_PATH,
        dataset_name=dataset_name,
        deleted_nodes=deleted_nodes,
        custom_percentile=None
    )
    
    importance_sampling_size = importance_sampling_K * importance_sampling_N
    random_walk_size = len(high_influence_nodes)
    method = "importancesampling" if importance_sampling_size < random_walk_size else "randomwalk"
    
    return method, importance_sampling_size, random_walk_size, high_influence_nodes

def analyze_and_save_walk_statistics(
    input_dir: str,
    dataset_name: str,
    percentile: float = 80,
    output_dir: str = None
) -> Dict:
    """
    Analyze random walk counts and save statistics for a dataset
    
    Args:
        input_dir: Directory containing walk count files
        dataset_name: Name of the dataset
        percentile: Percentile threshold (default 80)
        output_dir: Directory to save statistics (defaults to input_dir)
    
    Returns:
        Dictionary containing summary statistics
    """
    if output_dir is None:
        output_dir = input_dir
        
    # Load walk counts metadata
    with open(os.path.join(input_dir, f'{dataset_name}_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        
    # Initialize counters
    total_visits = defaultdict(int)
    visit_distribution = []
    
    # Process files in batches
    batch_files = sorted([f for f in os.listdir(input_dir) 
                         if f.startswith(f'walkcounts_{dataset_name}_')])
    
    for filename in tqdm(batch_files, desc=f"Processing {dataset_name} walk counts"):
        filepath = os.path.join(input_dir, filename)
        
        # Load batch
        with open(filepath, 'rb') as f:
            batch_counts = pickle.load(f)
            
        # Aggregate counts
        for source_node, counts in batch_counts.items():
            for target_node, count in counts.items():
                total_visits[target_node] += count
                visit_distribution.append(count)
                
        # Clear memory
        del batch_counts
        gc.collect()
    
    # Calculate statistics
    visit_array = np.array(visit_distribution)
    stats = {
        'dataset': dataset_name,
        'total_nodes': metadata['num_nodes'],
        'total_visits': sum(total_visits.values()),
        'unique_visited_nodes': len(total_visits),
        'mean_visits': float(np.mean(visit_array)),
        'median_visits': float(np.median(visit_array)),
        'percentile_threshold': float(np.percentile(visit_array, percentile)),
        'max_visits': float(np.max(visit_array)),
        'min_visits': float(np.min(visit_array)),
        'std_visits': float(np.std(visit_array))
    }
    
    # Save statistics
    output_file = os.path.join(output_dir, f'{dataset_name}_walk_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print('dataset: ', dataset_name, ' total_visits: ', stats['total_visits'], ' unique_visited_nodes: ', 
          stats['unique_visited_nodes'], ' mean_visits: ', stats['mean_visits'], ' median_visits: ', stats['median_visits'], 
          ' percentile_threshold: ', stats['percentile_threshold'], ' max_visits: ', stats['max_visits'], ' min_visits: ', stats['min_visits'], ' std_visits: ', stats['std_visits'],)
    """
    # Save visit distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(visit_array, bins=50, density=True)
    plt.axvline(stats['percentile_threshold'], color='r', linestyle='--', 
                label=f'{percentile}th percentile')
    plt.title(f'Visit Distribution for {dataset_name}')
    plt.xlabel('Number of Visits')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_visit_distribution.png'))
    plt.close()
    """
    return stats

def analyze_multiple_datasets(
    base_dir: str,
    dataset_names: List[str],
    percentile: float = 80
) -> Dict[str, Dict]:
    """
    Analyze walk statistics for multiple datasets
    
    Args:
        base_dir: Base directory containing dataset subdirectories
        dataset_names: List of dataset names to process
        percentile: Percentile threshold
    
    Returns:
        Dictionary of statistics for each dataset
    """
    all_stats = {}
    
    for dataset in dataset_names:
        print(f"\nProcessing dataset: {dataset}")
        try:
            stats = analyze_and_save_walk_statistics(
                input_dir=base_dir,
                dataset_name=dataset,
                percentile=percentile
            )
            all_stats[dataset] = stats
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    # Save comparative statistics
    comparison_file = os.path.join(base_dir, 'dataset_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_stats, f, indent=4)
        
    return all_stats
