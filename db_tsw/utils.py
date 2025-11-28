import torch
import time
import numpy as np
from .von_mises_fisher import VonMisesFisher
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
def generate_random_projecting_tree_frames(X, Y, ntrees, nlines, d, mean=123, std=0.1, device='cuda'):
    X = X.to(device)
    Y = Y.to(device)
    
    # Root placement
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    # KEY FIX: Bias sampling toward high-difference pairs
    mean_X, mean_Y = X.mean(dim=0), Y.mean(dim=0)
    
    # Compute distances from each X point to Y mean (and vice versa)
    X_to_Y_dist = torch.norm(X - mean_Y.unsqueeze(0), dim=1)
    Y_to_X_dist = torch.norm(Y - mean_X.unsqueeze(0), dim=1)
    
    # Create probability weights (outliers more likely to be selected)
    X_weights = X_to_Y_dist / X_to_Y_dist.sum()
    Y_weights = Y_to_X_dist / Y_to_X_dist.sum()
    
    total_lines = ntrees * nlines
    
    # Weighted sampling instead of uniform
    x_indices = torch.multinomial(X_weights, total_lines, replacement=True)
    y_indices = torch.multinomial(Y_weights, total_lines, replacement=True)
    
    # Generate directions
    theta = X[x_indices] - Y[y_indices]
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
    theta = theta.reshape(ntrees, nlines, d)
    
    return theta, intercept

def generate_power_spherical_rpt_frames(X, Y, ntrees, nlines, d, mean=123, std=0.1, device='cuda', kappa=10.0):
    X, Y = X.to(device), Y.to(device)
    
    # Root placement
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    intercept = root
    
    # Generate base directions
    total_lines = ntrees * nlines
    x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
    y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)
    
    base_theta = X[x_indices] - Y[y_indices]
    base_theta = base_theta / torch.norm(base_theta, dim=1, keepdim=True)
    
    # Adaptive concentration parameter
    current_distance = torch.norm(X.mean(dim=0) - Y.mean(dim=0))
    adaptive_kappa = kappa / (current_distance.item() + 0.1)  # Convert to scalar
    
    # Apply VonMisesFisher concentration
    vmf = VonMisesFisher(loc=base_theta, scale=torch.full((total_lines,), adaptive_kappa, device=device))
    theta = vmf.rsample()
    
    return theta.reshape(ntrees, nlines, d), intercept