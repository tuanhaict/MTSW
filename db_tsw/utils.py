import torch
import time

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U


def compute_adaptive_mean(X, Y):
    """
    Compute adaptive mean from supports of both distributions.
    
    Args:
        X: Source supports (n, d)
        Y: Target supports (m, d)
    
    Returns:
        mean: Midpoint of centroids (d,)
    """
    centroid_X = X.mean(dim=0)
    centroid_Y = Y.mean(dim=0)
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