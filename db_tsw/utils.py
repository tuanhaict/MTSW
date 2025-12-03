import torch
import time
import numpy as np
from .von_mises_fisher import VonMisesFisher
from .power_spherical import PowerSpherical
from scipy.optimize import linear_sum_assignment
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
def estimate_sw_1d_from_random_pairs(X, Y, n_pairs=4, device=None):
    if device is None:
        device = X.device

    N_x, d_dim = X.shape
    N_y = Y.shape[0]

    d_list = []

    for _ in range(n_pairs):
        ix = torch.randint(0, N_x, (1,), device=device)
        iy = torch.randint(0, N_y, (1,), device=device)

        diff = X[ix] - Y[iy]       
        diff = diff.view(-1)        
        norm = torch.norm(diff)
        if norm < 1e-8:
            continue

        theta = diff / norm         # (d,)
        X_proj = X @ theta          # (N_x,)
        Y_proj = Y @ theta          # (N_y,)

        X_sorted, _ = torch.sort(X_proj)
        Y_sorted, _ = torch.sort(Y_proj)

        n = min(X_sorted.shape[0], Y_sorted.shape[0])
        d_1d = torch.mean(torch.abs(X_sorted[:n] - Y_sorted[:n]))
        d_list.append(d_1d)

    if len(d_list) == 0:
        return torch.tensor(0.0, device=device)

    d_est = torch.stack(d_list).mean()  # scalar tensor
    return d_est
def generate_random_projecting_tree_frames(
    X, Y, ntrees, nlines, d, mean=123, std=0.1,
    device='cuda',
    w_scale=0.1, n_pairs_for_d=4
):

    X = X.to(device)
    Y = Y.to(device)

    dim = d 
    with torch.no_grad():
        d_est = estimate_sw_1d_from_random_pairs(
            X, Y, n_pairs=n_pairs_for_d, device=device
        )  
    if w_scale <= 0:
        raise ValueError("w_scale must be positive.")
    w = 1.0 - torch.exp(- (d_est / w_scale) ** 100)
    w = torch.clamp(w, 0.0, 1.0)
    w_float = float(w.item()) 

    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, dim, device=device) * std + mean
    else:
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, dim, device=device) * std + mean_tensor.view(1, 1, dim)

    intercept = root

    total_lines = ntrees * nlines
    x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
    y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)

    diff_xy = X[x_indices] - Y[y_indices]
    diff_norm = torch.norm(diff_xy, dim=1, keepdim=True) + 1e-8
    diff_dir = diff_xy / diff_norm 

    u = torch.randn_like(diff_dir)
    u_norm = torch.norm(u, dim=1, keepdim=True) + 1e-8
    u_dir = u / u_norm

    theta = w_float * diff_dir + (1.0 - w_float) * u_dir

    theta_norm = torch.norm(theta, dim=1, keepdim=True) + 1e-8
    theta = theta / theta_norm
    theta = theta.view(ntrees, nlines, dim)

    return theta, intercept

def generate_hungarian_projecting_tree_frames(
    X, Y, ntrees, nlines, d,
    mean=123, std=0.1, device='cuda',
    eps=1e-8
):
    X = X.to(device)
    Y = Y.to(device)

    # ===== intercept =====
    if isinstance(mean, (int, float)):
        intercept = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device)
        intercept = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)

    total = ntrees * nlines

    # ===== Hungarian matching giữa X và Y =====
    # nếu size khác nhau thì sample lại cho bằng nhau
    n = min(X.shape[0], Y.shape[0])
    idxX = np.random.choice(X.shape[0], n, replace=False)
    idxY = np.random.choice(Y.shape[0], n, replace=False)

    Xp = X[idxX]  # (n,d)
    Yp = Y[idxY]

    # distance matrix (CPU để dùng scipy)
    dist = torch.cdist(Xp, Yp, p=2).detach().cpu().numpy()  # (n,n)

    row_ind, col_ind = linear_sum_assignment(dist)

    # cắt hoặc lặp lại nếu không đủ total
    if n < total:
        rep = total // n + 1
        row_ind = np.tile(row_ind, rep)[:total]
        col_ind = np.tile(col_ind, rep)[:total]
    else:
        row_ind = row_ind[:total]
        col_ind = col_ind[:total]

    # ===== tạo hướng =====
    diff = Xp[row_ind] - Yp[col_ind]          # (total, d)
    diff_norm = torch.norm(diff, dim=1, keepdim=True).clamp_min(eps)
    theta = (diff / diff_norm).reshape(ntrees, nlines, d)

    return theta, intercept

def generate_adaptive_projecting_tree_frames(X, Y, ntrees, nlines, d, 
                                           prev_tsw=None, schedule_type='laplace',
                                           mean=123, std=0.1, device='cuda'):
    X = X.to(device)
    Y = Y.to(device)
    
    # Compute adaptive weight directly from TSW value
    if prev_tsw is not None:
        prev_tsw = prev_tsw.detach()
        
        # Define convergence threshold (when TSW is considered "converged")
        convergence_threshold = 0.02  # Adjust based on your problem
        
        # Normalize TSW to [0, 1] range where 1 = far from convergence, 0 = converged
        normalized_tsw = torch.clamp(prev_tsw / convergence_threshold, 0.0, 1.0)
        
        # Compute adaptive weight based on schedule
        if schedule_type == 'laplace':
            # w = exp(-|log(normalized_tsw)|/b)
            # When normalized_tsw = 1 → log ≈ 0 → w ≈ 1
            # When normalized_tsw → 0 → log → -∞ → w → 0
            log_norm_tsw = torch.log(normalized_tsw + 1e-8)
            w = torch.exp(-torch.abs(log_norm_tsw) / 1.0)
            
        elif schedule_type == 'cauchy':
            # w = 1 / (1 + (shift - log(normalized_tsw))^2)
            log_norm_tsw = torch.log(normalized_tsw + 1e-8)
            w = 1.0 / (1.0 + (log_norm_tsw + 2.0)**2)  # shift = -2
            
        elif schedule_type == 'sech':
            # w = sech(log(normalized_tsw))
            log_norm_tsw = torch.log(normalized_tsw + 1e-8)
            w = 1.0 / torch.cosh(-log_norm_tsw)
            
        elif schedule_type == 'sigmoid':
            # Smooth transition: w = sigmoid(k * (TSW - threshold))
            k = 10.0  # steepness
            transition_point = 0.05  # TSW value where w = 0.5
            w = torch.sigmoid(k * (prev_tsw - transition_point))
            
        else:  # linear
            w = normalized_tsw
            
        print(f"Previous TSW: {prev_tsw.item():.6f}")
        print(f"Normalized TSW: {normalized_tsw.item():.6f}")
        print(f"Adaptive weight w: {w.item():.6f}")
        
    else:
        # First iteration: assume far from convergence
        w = torch.tensor(0.9, device=device)
        print(f"Initial weight w: {w.item():.6f}")
    
    # No clamping needed - natural range should be [0,1]
    w = torch.clamp(w, 0.01, 0.99)  # Keep minimal clamp for numerical stability
    
    # Generate tree intercepts
    if isinstance(mean, (int, float)):
        root = torch.randn(ntrees, 1, d, device=device) * std + mean
    else:
        mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
        root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
    
    intercept = root
    
    # Generate adaptive projections
    total_lines = ntrees * nlines
    x_indices = np.random.choice(X.shape[0], total_lines, replace=True)
    y_indices = np.random.choice(Y.shape[0], total_lines, replace=True)
    
    # Signal directions (X-Y normalized)
    signal_dirs = X[x_indices] - Y[y_indices]
    signal_dirs = signal_dirs / torch.sqrt(torch.sum(signal_dirs ** 2, dim=1, keepdim=True))
    
    # Random uniform directions
    random_dirs = torch.randn(total_lines, d, device=device)
    random_dirs = random_dirs / torch.sqrt(torch.sum(random_dirs ** 2, dim=1, keepdim=True))
    
    # Adaptive combination
    sqrt_w = torch.sqrt(w)
    sqrt_1_minus_w = torch.sqrt(1 - w)
    
    theta = sqrt_w * signal_dirs + sqrt_1_minus_w * random_dirs
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

class RationalGateTreeFrameGenerator:
    def __init__(self, rs_min=None, rs_max=None, eps=1e-8, num_iter=1000,
                 schedule='exp', r_floor_coef=1e-3, device='cuda'):
        """
        Rational Gate Tree Frame Generator with iteration-controlled rs.

        Args:
            rs_min: minimal rs (float). If None, set adaptively at runtime.
            rs_max: maximal rs (float). If None, set adaptively at runtime (median).
            eps: numerical stability.
            num_iter: total number of iterations used for scheduling.
            schedule: 'exp' | 'cosine' | 'linear' : schedule for rs_t between rs_min and rs_max.
            r_floor_coef: threshold coefficient relative to median distance; if r < r_floor_coef * median -> force w=0.
            device: torch device.
        """
        self.rs_min = rs_min
        self.rs_max = rs_max
        self.eps = eps
        self.num_iter = max(1, int(num_iter))
        self.schedule = schedule
        self.iter = 0
        self.device = device
        self.r_floor_coef = r_floor_coef

        # internal caches that can be set on first generate call
        self._initialized = False
        self._median_r = None

    def _init_rs_bounds(self, r_tensor):
        # r_tensor: 1D tensor of distances from current batch/pool
        median_r = float(torch.median(r_tensor).cpu().item() + 1e-12)
        self._median_r = median_r

        # default rs_min small fraction of median (so w ~ 1 for typical r)
        if self.rs_min is None:
            self.rs_min = max(1e-8, 1e-3 * median_r)  # can be very small
        if self.rs_max is None:
            # set rs_max to be median or a bit larger so that late stage leans to uniform
            self.rs_max = max(1e-6, median_r)

        # ensure ordering
        if self.rs_min <= 0:
            self.rs_min = 1e-8
        if self.rs_max < self.rs_min:
            self.rs_max = self.rs_min * 10.0

        self._initialized = True

    def _rs_at_iter(self, t):
        """Return rs_t according to chosen schedule (t in [0, num_iter])."""
        # clip t
        t = float(max(0, min(t, self.num_iter)))
        if self.schedule == 'exp':
            # exponential interpolation: rs_min * (rs_max/rs_min)^(t/T)
            ratio = (self.rs_max / (self.rs_min + 1e-12))
            # if ratio == 1 -> constant
            if ratio <= 1.0 + 1e-12:
                return float(self.rs_min)
            power = (t / float(self.num_iter))
            return float(self.rs_min * (ratio ** power))
        elif self.schedule == 'cosine':
            # cosine annealing from rs_min -> rs_max
            cos_val = 0.5 * (1.0 - np.cos(np.pi * t / float(self.num_iter)))
            return float(self.rs_min * (1 - cos_val) + self.rs_max * cos_val)
        elif self.schedule == 'linear':
            frac = t / float(self.num_iter)
            return float(self.rs_min * (1 - frac) + self.rs_max * frac)
        else:
            raise ValueError("Unknown schedule: choose 'exp'|'cosine'|'linear'")

    def step(self, n=1):
        """Advance internal iteration counter by n (call this each training iteration)."""
        self.iter = min(self.num_iter, self.iter + int(n))

    def reset(self):
        """Reset iteration counter (optional)."""
        self.iter = 0

    def generate(self, X, Y, ntrees, nlines, d,
                 mean=123, std=0.1, device=None, return_w=False):
        """
        Generate (theta, intercept) for Tree-SW using rational gate with rs_t.
        If return_w=True, also return w tensor (ntrees, nlines).
        """
        if device is None:
            device = self.device
        X = X.to(device)
        Y = Y.to(device)

        total = ntrees * nlines
        x_idx = np.random.choice(X.shape[0], total, replace=True)
        y_idx = np.random.choice(Y.shape[0], total, replace=True)

        x_sel = X[x_idx]           # (total, d)
        y_sel = Y[y_idx]           # (total, d)
        diff = x_sel - y_sel       # (total, d)

        # compute norms r and mu
        norms = torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True)).clamp_min(self.eps)  # (total,1)
        r = norms.view(-1).to(device)  # (total,)
        mu = (diff / norms).to(device)  # (total, d)

        # initialize bounds if first call
        if not self._initialized:
            self._init_rs_bounds(r)

        # compute rs_t from schedule
        rs_t = self._rs_at_iter(self.iter)
        rs_t = max(self.eps, float(rs_t))

        # compute raw w = r / (r + rs_t)
        # but enforce r_floor: if r < r_floor_coef * median_r -> set w = 0
        if self._median_r is None:
            r_floor = self.r_floor_coef
        else:
            r_floor = self.r_floor_coef * max(self._median_r, 1e-12)

        # vectorized compute
        rs_t_tensor = torch.full((total,), float(rs_t), device=device)
        w_raw = r / (r + rs_t_tensor).to(device)

        # force small r -> w = 0 (hard floor)
        w = torch.where(r < r_floor, torch.zeros_like(w_raw), w_raw)

        # optional: clamp to [0,1]
        w = w.clamp(0.0, 1.0)

        # sample u ~ Uniform(S^{d-1})
        normal = torch.randn(total, d, device=device)
        normal = normal / torch.sqrt(torch.sum(normal**2, dim=1, keepdim=True)).clamp_min(self.eps)

        # convex combination & normalization
        w_col = w.view(-1, 1)
        z = (1.0 - w_col) * normal + w_col * mu
        z = z / torch.sqrt(torch.sum(z**2, dim=1, keepdim=True)).clamp_min(self.eps)

        theta = z.reshape(ntrees, nlines, d)

        # intercept (same as before)
        if isinstance(mean, (int, float)):
            root = torch.randn(ntrees, 1, d, device=device) * std + mean
        else:
            mean_tensor = mean.to(device) if mean.device != torch.device(device) else mean
            root = torch.randn(ntrees, 1, d, device=device) * std + mean_tensor.view(1, 1, d)
        intercept = root

        if return_w:
            return theta, intercept, w.view(ntrees, nlines)
        return theta, intercept
