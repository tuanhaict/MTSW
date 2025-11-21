# import torch

# def svd_orthogonalize(matrix):
#     U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
#     return U

# def generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
#     # random root as gaussian distribution with given mean and std
#     assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
#     root = torch.randn(ntrees, 1, d, device=device) * std + mean
#     intercept = root
    
#     if gen_mode == 'gaussian_raw':
#         theta = torch.randn(ntrees, nlines, d, device=device)
#         theta = theta / torch.norm(theta, dim=-1, keepdim=True)
#     elif gen_mode == 'gaussian_orthogonal':
#         assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
#         theta = torch.randn(ntrees, d, nlines, device=device)
#         theta = svd_orthogonalize(theta)
#         theta = theta.transpose(-2, -1)
    
#     return theta, intercept

import torch
import torch.nn.functional as F

def svd_orthogonalize(matrix):
    """Orthogonalize matrix using SVD"""
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U

def compute_tree_diversity_score(theta):
    """
    Compute diversity score between trees
    theta: (ntrees, nlines, d)
    Returns: scalar diversity score (higher is better)
    """
    ntrees, nlines, d = theta.shape
    
    # Compute pairwise cosine similarity between all direction pairs across trees
    theta_flat = theta.reshape(ntrees * nlines, d)  # (ntrees*nlines, d)
    
    # Normalize
    theta_norm = F.normalize(theta_flat, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(theta_norm, theta_norm.t())  # (ntrees*nlines, ntrees*nlines)
    
    # Remove diagonal (self-similarity)
    mask = ~torch.eye(ntrees * nlines, dtype=torch.bool, device=theta.device)
    similarities = sim_matrix[mask]
    
    # Diversity score: lower average similarity = higher diversity
    diversity_score = 1.0 - similarities.abs().mean()
    
    return diversity_score

def repulsion_sampling(ntrees, nlines, d, device='cuda', n_iterations=100, lr=0.1):
    """
    Generate diverse directions using repulsion-based optimization
    Directions are pushed away from each other to maximize diversity
    """
    # Initialize randomly
    theta = torch.randn(ntrees, nlines, d, device=device, requires_grad=True)
    theta.data = F.normalize(theta.data, dim=-1)
    
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    for _ in range(n_iterations):
        optimizer.zero_grad()
        
        # Normalize to unit sphere
        theta_norm = F.normalize(theta, dim=-1)
        
        # Flatten to compute pairwise similarities
        theta_flat = theta_norm.reshape(ntrees * nlines, d)
        
        # Compute similarity matrix (we want to minimize this)
        sim_matrix = torch.mm(theta_flat, theta_flat.t())
        
        # Remove self-similarity and compute repulsion loss
        # We want to maximize minimum distance = minimize maximum similarity
        mask = ~torch.eye(ntrees * nlines, dtype=torch.bool, device=device)
        similarities = sim_matrix[mask]
        
        # Loss: minimize squared similarities (repulsion)
        loss = (similarities ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        # Project back to unit sphere
        theta.data = F.normalize(theta.data, dim=-1)
    
    return theta.detach()

def stratified_sampling(ntrees, nlines, d, device='cuda'):
    """
    Divide the sphere into strata and sample from each
    Use spherical coordinates approach
    """
    theta_list = []
    
    # Divide trees into groups for stratification
    n_strata = min(ntrees, 8)  # Number of strata
    trees_per_stratum = ntrees // n_strata
    
    for i in range(n_strata):
        # Each stratum gets a different base rotation
        base_rotation = torch.randn(d, d, device=device)
        base_rotation = svd_orthogonalize(base_rotation)
        
        # Generate directions for this stratum
        for _ in range(trees_per_stratum):
            theta_tree = torch.randn(nlines, d, device=device)
            theta_tree = F.normalize(theta_tree, dim=-1)
            
            # Apply base rotation to create diversity
            theta_tree = torch.mm(theta_tree, base_rotation)
            theta_list.append(theta_tree)
    
    # Handle remaining trees
    remaining = ntrees - len(theta_list)
    if remaining > 0:
        theta_remaining = torch.randn(remaining, nlines, d, device=device)
        theta_remaining = F.normalize(theta_remaining, dim=-1)
        theta_list.extend([theta_remaining[i] for i in range(remaining)])
    
    theta = torch.stack(theta_list, dim=0)
    return theta

def dpp_sampling(ntrees, nlines, d, device='cuda', temperature=1.0):
    """
    Determinantal Point Process sampling for diverse directions
    Sample directions that are maximally different from each other
    """
    # Generate candidate pool (oversample)
    n_candidates = ntrees * 10
    candidates = torch.randn(n_candidates, nlines, d, device=device)
    candidates = F.normalize(candidates, dim=-1)
    
    # Compute kernel matrix (similarity matrix)
    candidates_flat = candidates.reshape(n_candidates, nlines * d)
    kernel = torch.mm(candidates_flat, candidates_flat.t())
    kernel = torch.exp(kernel / temperature)  # Temperature controls diversity
    
    # Greedy DPP sampling (approximation)
    selected_indices = []
    remaining_indices = list(range(n_candidates))
    
    # Select first point uniformly
    first_idx = torch.randint(0, n_candidates, (1,), device=device).item()
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select points that are most different from selected ones
    for _ in range(ntrees - 1):
        if not remaining_indices:
            break
            
        # Compute marginal gain for each remaining point
        selected_kernel = kernel[selected_indices, :][:, remaining_indices]
        
        # Select point with minimum similarity to selected set
        similarities = selected_kernel.max(dim=0)[0]
        best_idx_in_remaining = similarities.argmin().item()
        best_idx = remaining_indices[best_idx_in_remaining]
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Return selected directions
    theta = candidates[selected_indices]
    return theta

def orthogonal_tree_sampling(ntrees, nlines, d, device='cuda'):
    """
    Ensure that trees have at least some orthogonal directions between them
    """
    assert nlines <= d, "nlines must be <= d for orthogonal sampling"
    
    # Generate orthogonal bases for different trees
    theta_list = []
    
    # Create a pool of orthogonal directions
    n_bases = (ntrees * nlines + d - 1) // d  # Number of orthogonal bases needed
    
    for _ in range(n_bases):
        # Generate one orthogonal basis
        basis = torch.randn(d, d, device=device)
        basis = svd_orthogonalize(basis)
        theta_list.append(basis)
    
    # Concatenate and reshape
    all_directions = torch.cat(theta_list, dim=0)  # (n_bases*d, d)
    all_directions = all_directions[:ntrees * nlines]  # Take only what we need
    
    # Reshape to (ntrees, nlines, d)
    theta = all_directions.reshape(ntrees, nlines, d)
    
    # Shuffle within each tree for randomness
    for i in range(ntrees):
        perm = torch.randperm(nlines, device=device)
        theta[i] = theta[i][perm]
    
    return theta

def generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', 
                          gen_mode='gaussian_raw', diversity_mode='none'):
    """
    Generate tree frames with various diversity strategies
    
    Args:
        ntrees: number of trees
        nlines: number of lines per tree
        d: dimension
        mean: mean for root generation
        std: std for root generation
        device: computation device
        gen_mode: 'gaussian_raw', 'gaussian_orthogonal'
        diversity_mode: 'none', 'repulsion', 'stratified', 'dpp', 'orthogonal_trees'
    
    Returns:
        theta: (ntrees, nlines, d) direction vectors
        intercept: (ntrees, 1, d) root points
    """
    # Generate root as before
    root = torch.randn(ntrees, 1, d, device=device) * std + mean
    intercept = root
    
    # Generate directions based on gen_mode
    if gen_mode == 'gaussian_raw':
        if diversity_mode == 'none':
            theta = torch.randn(ntrees, nlines, d, device=device)
            theta = F.normalize(theta, dim=-1)
        elif diversity_mode == 'repulsion':
            theta = repulsion_sampling(ntrees, nlines, d, device=device)
        elif diversity_mode == 'stratified':
            theta = stratified_sampling(ntrees, nlines, d, device=device)
        elif diversity_mode == 'dpp':
            theta = dpp_sampling(ntrees, nlines, d, device=device)
        elif diversity_mode == 'orthogonal_trees':
            theta = orthogonal_tree_sampling(ntrees, nlines, d, device=device)
        else:
            raise ValueError(f"Invalid diversity_mode: {diversity_mode}")
            
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be >= number of lines for orthogonal lines"
        
        if diversity_mode == 'none':
            # Original code
            theta = torch.randn(ntrees, d, nlines, device=device)
            theta = svd_orthogonalize(theta)
            theta = theta.transpose(-2, -1)
        elif diversity_mode in ['repulsion', 'stratified', 'dpp', 'orthogonal_trees']:
            # First generate diverse directions
            if diversity_mode == 'repulsion':
                theta_init = repulsion_sampling(ntrees, nlines, d, device=device)
            elif diversity_mode == 'stratified':
                theta_init = stratified_sampling(ntrees, nlines, d, device=device)
            elif diversity_mode == 'dpp':
                theta_init = dpp_sampling(ntrees, nlines, d, device=device)
            elif diversity_mode == 'orthogonal_trees':
                theta_init = orthogonal_tree_sampling(ntrees, nlines, d, device=device)
            
            # Then orthogonalize within each tree
            theta = torch.zeros(ntrees, nlines, d, device=device)
            for i in range(ntrees):
                theta[i] = svd_orthogonalize(theta_init[i].t()).t()
        else:
            raise ValueError(f"Invalid diversity_mode: {diversity_mode}")
    else:
        raise ValueError(f"Invalid gen_mode: {gen_mode}")
    
    return theta, intercept