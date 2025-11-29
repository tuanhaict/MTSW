import torch
import time
import numpy as np
from .von_mises_fisher import VonMisesFisher
from .power_spherical import PowerSpherical
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

def generate_random_projecting_tree_frames(X, Y, ntrees, nlines, d, mean=123, std = 0.1, device='cuda'):
    X = X.to(device)
    Y = Y.to(device)
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        # mean is tensor (d,)
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    total_lines = ntrees * nlines
    x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
    y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)
    
    theta = X[x_indices] - Y[y_indices]
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
    theta = theta.reshape(ntrees, nlines, d)
    
    return theta, intercept
def generate_momentum_projecting_tree_frames(X, Y, ntrees, nlines, d, 
                                           prev_theta=None, prev_distances=None,
                                           beta_max=0.9, delta=1e-3, eps_0=1e-6, alpha=0.75,
                                           mean=123, std=0.1, device='cuda'):
    """
    Generate tree projection frames with adaptive momentum regularization
    
    Args:
        X, Y: Input distributions
        ntrees, nlines, d: Tree structure parameters
        prev_theta: Previous projection directions (ntrees, nlines, d)
        prev_distances: Previous ||X-Y|| distances for adaptive regularization
        beta_max, delta, eps_0, alpha: Momentum hyperparameters
    """
    X = X.to(device)
    Y = Y.to(device)
    
    # Initialize root (unchanged)
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    # Sample indices
    total_lines = ntrees * nlines
    x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
    y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)
    
    # Compute current raw directions
    diff_vectors = X[x_indices] - Y[y_indices]  # (total_lines, d)
    
    # Compute distances for adaptive parameters
    distances = torch.sqrt(torch.sum(diff_vectors ** 2, dim=1))  # (total_lines,)
    
    # Adaptive epsilon regularization
    if prev_distances is not None:
        initial_distances = prev_distances.mean()
        eps_t = eps_0 * (distances / initial_distances) ** alpha
    else:
        eps_t = eps_0 * torch.ones_like(distances)
    
    # Regularized normalization
    eps_t = eps_t.unsqueeze(1)  # (total_lines, 1)
    current_directions = diff_vectors / (distances.unsqueeze(1) + eps_t)
    
    # Adaptive momentum coefficient
    beta_t = torch.clamp(
        beta_max - delta / (distances + eps_0),
        min=0.0, max=beta_max
    )  # (total_lines,)
    
    # Apply momentum if previous directions exist
    if prev_theta is not None:
        prev_theta_flat = prev_theta.view(total_lines, d)  # Flatten previous directions
        beta_t = beta_t.unsqueeze(1)  # (total_lines, 1)
        
        # Momentum update: v_{t+1} = β_t * v_t + (1-β_t) * current_direction
        momentum_directions = beta_t * prev_theta_flat + (1 - beta_t) * current_directions
    else:
        # First iteration: no momentum
        momentum_directions = current_directions
    
    # Final normalization
    momentum_norms = torch.sqrt(torch.sum(momentum_directions ** 2, dim=1, keepdim=True))
    theta = momentum_directions / (momentum_norms + 1e-12)  # Numerical stability
    
    # Reshape back to tree structure
    theta = theta.reshape(ntrees, nlines, d)
    
    # Return distances for next iteration
    distances_reshaped = distances.reshape(ntrees, nlines)
    
    return theta, intercept, distances_reshaped

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
    
    # Adaptive kappa strategies
    mean_distance = torch.norm(X.mean(dim=0) - Y.mean(dim=0))
    
    # Strategy 1: Exponential concentration
    # adaptive_kappa = kappa / torch.exp(-mean_distance)
    
    # Strategy 2: Inverse relationship
    # adaptive_kappa = kappa / (mean_distance + 0.1)
    
    # Strategy 3: Sigmoid concentration
    # adaptive_kappa = kappa * torch.sigmoid(5.0 / (mean_distance + 0.1))
    
    # Strategy 4: Power law concentration
    if mean_distance < 0.05:
        return generate_trees_frames(ntrees, nlines, d, mean=mean, std=std, device=device, gen_mode='gaussian_orthogonal')
    if (np.random.randint(1,10) < 5):
        print(f"Adaptive kappa: {kappa:.4f}")
        print(f"Mean distance between X and Y: {mean_distance:.4f}")    
    # Strategy 5: Logarithmic concentration
    # adaptive_kappa = kappa * torch.log(1 / (mean_distance + 0.1) + 1)
    
    ps = PowerSpherical(
        loc=base_theta,
        scale=torch.full((total_lines,), kappa, device=device),
    )
    theta = ps.rsample()
    
    return theta.reshape(ntrees, nlines, d), intercept
def generate_rational_gate_tree_frames(
    X, Y, ntrees, nlines, d,
    mean=123, std=0.1, device='cuda',
    rs=None,          # scale parameter for rational gate
    eps=1e-8
):
    """
    Generate (theta, intercept) for Tree-SW using:
        theta = normalize( (1 - w(r))*u + w(r)*mu )
    with rational gate:
        w(r) = r / (r + rs)

    - Ensures w(0) = 0  -> theta ~ uniform when x ≈ y
    - Ensures w(r)→1   -> theta ≈ mu when x far from y
    - No need for large κ, avoids numerical instability.
    """

    X = X.to(device)
    Y = Y.to(device)

    # ======= intercept (same as your original code) =======
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    intercept = root

    # ======= sample pairs =======
    total = ntrees * nlines
    x_idx = np.random.choice(X.shape[0], total, replace=True)
    y_idx = np.random.choice(Y.shape[0], total, replace=True)

    x_sel = X[x_idx]           # (total, d)
    y_sel = Y[y_idx]           # (total, d)
    diff = x_sel - y_sel       # (total, d)

    # ======= compute r and mu =======
    norms = torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True)).clamp_min(eps)  # (total,1)
    r = norms.view(-1)  # (total,)
    mu = diff / norms   # (total, d)

    # ======= rational gate w(r) = r / (r + rs) =======
    if rs is None:
        # rs = median distance (robust, auto-scaling)
        rs = float(torch.median(r).cpu().item() + eps)

    w = r / (r + rs)    # (total,)
    w = w.to(device)
    w_col = w.view(-1, 1)

    # ======= sample u ~ Uniform(S^{d-1}) via Gaussian normalize =======
    normal = torch.randn(total, d, device=device)
    normal = normal / torch.sqrt(torch.sum(normal**2, dim=1, keepdim=True)).clamp_min(eps)

    # ======= convex combination & normalization =======
    z = (1.0 - w_col) * normal + w_col * mu
    z = z / torch.sqrt(torch.sum(z**2, dim=1, keepdim=True)).clamp_min(eps)

    theta = z.reshape(ntrees, nlines, d)

    return theta, intercept