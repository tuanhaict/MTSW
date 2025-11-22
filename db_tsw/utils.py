import torch
import time

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
def generate_trees_frames_diff_aware(
    ntrees, nlines, d, 
    X=None, Y=None,
    mean=128, std=0.1, 
    device='cuda', 
    gen_mode='gaussian_raw',
    diff_ratio=0.5  # Ratio của directions align với difference
):
    """
    Tree generation với directions aware of difference between X và Y.
    """
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    
    # Root sampling (giữ nguyên hoặc adaptive)
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device)
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    # Direction sampling với difference-awareness
    if X is not None and Y is not None:
        X = X.to(device)
        Y = Y.to(device)
        
        # Compute difference direction (từ centroid X đến centroid Y)
        diff_dir = Y.mean(dim=0) - X.mean(dim=0)
        diff_norm = diff_dir.norm()
        
        if diff_norm > 1e-6:
            diff_dir = diff_dir / diff_norm
            
            # Số directions align với difference
            n_diff = max(1, int(nlines * diff_ratio))
            n_random = nlines - n_diff
            
            # Random directions
            if gen_mode == 'gaussian_raw':
                theta_random = torch.randn(ntrees, n_random, d, device=device)
                theta_random = theta_random / theta_random.norm(dim=-1, keepdim=True)
            else:  # gaussian_orthogonal
                theta_random = torch.randn(ntrees, d, n_random, device=device)
                U, _, _ = torch.linalg.svd(theta_random, full_matrices=False)
                theta_random = U.transpose(-2, -1)
            
            # Difference-aligned directions (với small perturbation)
            theta_diff = diff_dir.view(1, 1, d).expand(ntrees, n_diff, d)
            theta_diff = theta_diff + torch.randn(ntrees, n_diff, d, device=device) * 0.1
            theta_diff = theta_diff / theta_diff.norm(dim=-1, keepdim=True)
            
            theta = torch.cat([theta_diff, theta_random], dim=1)
        else:
            # Fallback to random if no difference
            theta = torch.randn(ntrees, nlines, d, device=device)
            theta = theta / theta.norm(dim=-1, keepdim=True)
    else:
        # Original random sampling
        if gen_mode == 'gaussian_raw':
            theta = torch.randn(ntrees, nlines, d, device=device)
            theta = theta / theta.norm(dim=-1, keepdim=True)
        else:
            theta = torch.randn(ntrees, d, nlines, device=device)
            theta = svd_orthogonalize(theta).transpose(-2, -1)
    
    return theta, intercept