import torch
import time
import numpy as np

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U


def compute_adaptive_mean(X, Y, device='cuda'):
    """Compute midpoint of centroids."""
    centroid_X = X.mean(dim=0).to(device)
    centroid_Y = Y.mean(dim=0).to(device)
    return (centroid_X + centroid_Y) / 2


def generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    
    # Handle mean as scalar or tensor
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        # mean is tensor (d,)
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    if gen_mode == 'gaussian_raw':
        theta = torch.randn(ntrees, nlines, d, device=device)
        theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
        theta = torch.randn(ntrees, d, nlines, device=device)
        theta = svd_orthogonalize(theta)
        theta = theta.transpose(-2, -1)
    
    return theta, intercept

# def generate_random_projecting_tree_frames(X, Y, ntrees, nlines, d, mean=123, std = 0.1, device='cuda'):
#     X = X.to(device)
#     Y = Y.to(device)
#     if isinstance(mean, (int, float)):
#         root = torch.randn(ntrees, 1, d, device=device) * std + mean
#     else:
#         # mean is tensor (d,)
#         mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
#         root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
#     intercept = root
    
#     total_lines = ntrees * nlines
#     x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
#     y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)
    
#     theta = X[x_indices] - Y[y_indices]
#     theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
#     theta = theta.reshape(ntrees, nlines, d)
    
#     return theta, intercept
def generate_random_projecting_tree_frames(X, Y, ntrees, nlines, d, mean=123, std=0.1, device='cuda', iteration=None):
    X = X.to(device)
    Y = Y.to(device)
    
    # Compute current distance to determine convergence phase
    mean_X, mean_Y = X.mean(dim=0), Y.mean(dim=0)
    current_distance = torch.norm(mean_X - mean_Y).item()
    
    # Adaptive root placement based on convergence phase
    if current_distance < 0.1:  # Near convergence - be more conservative
        if isinstance(mean, (int, float)):
            # Place roots closer to target for final convergence
            root = mean_Y.unsqueeze(0).unsqueeze(0).repeat(ntrees, 1, 1)
            root += torch.randn(ntrees, 1, d, device=device) * (std * 0.1)  # Much smaller noise
        else:
            mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
            root = mean_tensor.view(1, 1, d).repeat(ntrees, 1, 1)
            root += torch.randn(ntrees, 1, d, device=device) * (std * 0.1)
    else:
        # Far from convergence - normal placement
        if isinstance(mean, (int, float)):
            root = torch.randn(ntrees, 1, d, device=device) * std + mean
        else:
            mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
            root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    # Enhanced direction selection
    if current_distance < 0.05:  # Very close - use best directions only
        n_candidates = ntrees * nlines * 5  # Many candidates
        x_indices = np.random.choice(X.shape[0], n_candidates, replace=True)
        y_indices = np.random.choice(Y.shape[0], n_candidates, replace=True)
        
        # Get all candidate directions
        candidate_theta = X[x_indices] - Y[y_indices]
        candidate_magnitudes = torch.norm(candidate_theta, dim=1)
        
        # Select top directions with highest magnitude (most informative)
        total_lines = ntrees * nlines
        top_indices = torch.topk(candidate_magnitudes, total_lines)[1]
        theta = candidate_theta[top_indices]
        
    elif current_distance < 0.2:  # Moderate distance - mixed strategy
        total_lines = ntrees * nlines
        n_candidates = total_lines * 2
        x_indices = np.random.choice(X.shape[0], n_candidates, replace=True)
        y_indices = np.random.choice(Y.shape[0], n_candidates, replace=True)
        
        candidate_theta = X[x_indices] - Y[y_indices]
        candidate_magnitudes = torch.norm(candidate_theta, dim=1)
        
        # Mix of top directions and random
        n_top = total_lines // 2
        n_random = total_lines - n_top
        
        top_indices = torch.topk(candidate_magnitudes, n_top)[1]
        random_indices = np.random.choice(n_candidates, n_random, replace=False)
        
        selected_indices = torch.cat([top_indices, torch.tensor(random_indices, device=device)])
        theta = candidate_theta[selected_indices]
        
    else:  # Far from convergence - original strategy
        total_lines = ntrees * nlines
        x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
        y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)
        theta = X[x_indices] - Y[y_indices]
    
    # Normalize directions
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
    theta = theta.reshape(ntrees, nlines, d)
    
    return theta, intercept

def generate_adaptive_geometric_trees(X, Y, ntrees, nlines, d, device='cuda', 
                                    concentration_factor=10, geometric_factor=0.5):
    """
    Generate trees based on the geometric structure of X and Y distributions
    using principal directions and optimal transport theory
    """
    X, Y = X.to(device), Y.to(device)
    
    # Compute principal directions from covariance difference
    mean_X, mean_Y = X.mean(dim=0), Y.mean(dim=0)
    cov_X = torch.cov(X.T)
    cov_Y = torch.cov(Y.T) 
    
    # Principal directions from covariance difference (mathematically motivated)
    cov_diff = cov_X - cov_Y
    eigenvals, eigenvecs = torch.linalg.eigh(cov_diff)
    
    # Use top-k eigenvectors as principal directions
    principal_dirs = eigenvecs[:, -nlines:].T  # (nlines, d)
    
    # Adaptive root placement using Wasserstein barycenter approximation
    alpha = torch.rand(ntrees, 1, device=device)
    roots = alpha * mean_X.unsqueeze(0) + (1 - alpha) * mean_Y.unsqueeze(0)
    
    # Add geometric perturbation
    noise_scale = geometric_factor * torch.norm(mean_X - mean_Y) / concentration_factor
    geometric_noise = torch.randn(ntrees, 1, d, device=device) * noise_scale
    intercept = roots.unsqueeze(1) + geometric_noise
    
    # Replicate principal directions for all trees
    theta = principal_dirs.unsqueeze(0).repeat(ntrees, 1, 1)  # (ntrees, nlines, d)
    
    return theta, intercept

def generate_frequency_domain_trees(X, Y, ntrees, nlines, d, device='cuda'):
    """
    Generate trees using frequency domain analysis of the distributions
    """
    X, Y = X.to(device), Y.to(device)
    # Compute empirical characteristic functions (Fourier analysis)
    def empirical_char_func(data, frequencies):
        # φ(t) = E[exp(i⟨t, X⟩)]
        inner_prod = torch.matmul(frequencies, data.T)  # (nfreq, n_samples)
        return torch.mean(torch.exp(1j * inner_prod), dim=1)
    
    # Sample frequency vectors
    freq_samples = torch.randn(nlines * 10, d, device=device)
    freq_samples = freq_samples / torch.norm(freq_samples, dim=1, keepdim=True)
    
    # Compute characteristic function difference
    char_X = empirical_char_func(X, freq_samples)
    char_Y = empirical_char_func(Y, freq_samples)
    char_diff = torch.abs(char_X - char_Y)
    
    # Select frequencies with largest differences
    top_indices = torch.topk(char_diff.real, nlines)[1]
    optimal_dirs = freq_samples[top_indices]  # (nlines, d)
    
    # Generate roots using moment matching
    moment_1_diff = X.mean(dim=0) - Y.mean(dim=0)
    roots = torch.randn(ntrees, 1, d, device=device) * 0.1
    roots = roots + moment_1_diff.unsqueeze(0).unsqueeze(0) * 0.5
    
    theta = optimal_dirs.unsqueeze(0).repeat(ntrees, 1, 1)
    
    return theta, roots

def generate_information_theoretic_trees(X, Y, ntrees, nlines, d, device='cuda'):
    """
    Generate trees to maximize mutual information with distribution differences
    """
    X, Y = X.to(device), Y.to(device)
    # Compute mutual information-based directions
    def compute_mi_direction(data1, data2, direction):
        # Project data onto direction
        proj1 = torch.matmul(data1, direction)
        proj2 = torch.matmul(data2, direction)
        
        # Estimate MI using histogram-based method (simplified)
        bins = 50
        hist1, _ = torch.histogram(proj1, bins=bins)
        hist2, _ = torch.histogram(proj2, bins=bins)
        
        # Normalize to get probabilities
        p1 = hist1.float() / torch.sum(hist1)
        p2 = hist2.float() / torch.sum(hist2)
        
        # KL divergence as proxy for MI
        kl_div = torch.sum(p1 * torch.log((p1 + 1e-8) / (p2 + 1e-8)))
        return kl_div
    
    # Optimize directions to maximize information
    candidate_dirs = torch.randn(nlines * 5, d, device=device)
    candidate_dirs = candidate_dirs / torch.norm(candidate_dirs, dim=1, keepdim=True)
    
    mi_scores = torch.zeros(nlines * 5, device=device)
    for i, direction in enumerate(candidate_dirs):
        mi_scores[i] = compute_mi_direction(X, Y, direction)
    
    # Select top directions
    top_indices = torch.topk(mi_scores, nlines)[1]
    optimal_dirs = candidate_dirs[top_indices]
    
    # Strategic root placement using Fisher information
    fisher_info_X = torch.inverse(torch.cov(X.T) + torch.eye(d, device=device) * 1e-6)
    fisher_info_Y = torch.inverse(torch.cov(Y.T) + torch.eye(d, device=device) * 1e-6)
    
    # Roots based on Fisher information geometry
    mean_X, mean_Y = X.mean(dim=0), Y.mean(dim=0)
    fisher_diff = fisher_info_X - fisher_info_Y
    
    roots = torch.randn(ntrees, 1, d, device=device) * 0.1
    roots = roots + 0.3 * torch.matmul(fisher_diff, (mean_X - mean_Y)).unsqueeze(0).unsqueeze(0)
    
    theta = optimal_dirs.unsqueeze(0).repeat(ntrees, 1, 1)
    
    return theta, roots