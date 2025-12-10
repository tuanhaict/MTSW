import torch

from db_tsw.utils import generate_trees_frames

class TWConcurrentLines():
    def __init__(self, p=2, delta=2, mass_division='distance_based', device="cuda"):
        """
        Class for computing the Tree Wasserstein distance between two distributions.
        Args:
            p: level of the norm
            delta: negative inverse of softmax temperature for distance based mass division
            mass_division: how to divide the mass, one of 'uniform', 'distance_based'
            device: device to run the code, follow torch convention
        """
        self.device = device
        self.p = p
        self.delta = delta
        self.mass_division = mass_division

        assert self.mass_division in ['uniform', 'distance_based', 'discrimination_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based', 'discrimination_based'"

    def __call__(self, X, Y, theta, intercept):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        
        combined_axis_coordinate, mass_XY = self.get_mass_and_coordinate(X, Y, theta, intercept)
        tw = self.tw_concurrent_lines(mass_XY, combined_axis_coordinate)[0]

        return tw

    def tw_concurrent_lines(self, mass_XY, combined_axis_coordinate):
        """
        Args:
            mass_XY: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, num_lines, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
        mask_right = torch.nonzero(coord_sorted > 0, as_tuple=True)
        sub_mass_target_cumsum[mask_right] = sub_mass_right_cumsum[mask_right]

        ### compute edge length
        # add root to the sorted coordinate by insert 0 to the first position <= 0
        root = torch.zeros(num_trees, num_lines, 1, device=self.device) 
        root_indices = torch.searchsorted(coord_sorted, root)
        coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device)
        # distribute other points to the correct position
        edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
        edge_mask.scatter_(2, root_indices, False)
        coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
        # compute edge length
        edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw, sub_mass_target_cumsum, edge_length

    def get_mass_and_coordinate(self, X, Y, theta, intercept):
        N, dn = X.shape
        
        # Compute discrimination scores ONCE per tree
        if self.mass_division == 'discrimination_based':
            discrimination_scores = self.compute_discrimination_mean_diff(X, Y, theta, intercept)
            mass_X, axis_coordinate_X = self.project(
                X, theta=theta, intercept=intercept, 
                discrimination_scores=discrimination_scores
            )
            mass_Y, axis_coordinate_Y = self.project(
                Y, theta=theta, intercept=intercept,
                discrimination_scores=discrimination_scores
            )
        else:
            mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
            mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)
        
        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        massXY = torch.cat((mass_X, -mass_Y), dim=2)
        
        return combined_axis_coordinate, massXY
    def compute_discrimination_mean_diff(self, X, Y, theta, intercept):
        """
        Compute discrimination score per line based on mean difference.
        
        s(θᵢ) = |⟨μ_X - μ_Y, θᵢ⟩| + ε
        
        Args:
            X: (N, d) source samples
            Y: (M, d) target samples
            theta: (num_trees, num_lines, d) projection directions (normalized)
            intercept: (num_trees, 1, d) tree roots
            
        Returns:
            discrimination_scores: (num_trees, num_lines)
        """
        # Compute means
        mu_X = X.mean(dim=0)  # (d,)
        mu_Y = Y.mean(dim=0)  # (d,)
        
        # Mean difference vector
        diff = mu_X - mu_Y  # (d,)
        
        # Project difference onto each direction
        # |⟨diff, θᵢ⟩| for each tree and line
        scores = torch.abs(torch.einsum('d,tkd->tk', diff, theta))  # (T, k)
        
        # Add small noise to avoid zero discrimination
        # This handles case when μ_X ≈ μ_Y or when θ ⊥ diff
        epsilon = 1e-3 * torch.randn_like(scores).abs()  # small positive noise
        scores = scores + epsilon
        
        return scores

    def project(self, input, theta, intercept, discrimination_scores=None):
        """
        Project points onto lines with optional discrimination-based weighting.
        
        Args:
            input: (N, d) points to project
            theta: (num_trees, num_lines, d) directions
            intercept: (num_trees, 1, d) tree roots
            discrimination_scores: (num_trees, num_lines) [optional]
                If provided, use discrimination_based mass division
        """
        N, d = input.shape
        num_trees = theta.shape[0]
        num_lines = theta.shape[1]
        
        # Translate to tree coordinate system
        input_translated = input - intercept  # (T, N, d)
        
        # Project onto each line
        axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))  # (T, k, N)
        input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
        
        # Compute mass allocation
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
            
        elif self.mass_division == 'distance_based':
            # Geometric distance only
            dist = torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim=-1)
            weight = -self.delta * dist
            mass_input = torch.softmax(weight, dim=-2) / N
            
        elif self.mass_division == 'discrimination_based':
            assert discrimination_scores is not None, \
                "discrimination_scores required for discrimination_based"
            
            # === Two-Stage Allocation ===
            
            # Stage 1: Global line weights from discrimination
            # π(θᵢ) = softmax(β · s(θᵢ))
            line_weights = torch.softmax(
                self.delta * discrimination_scores,  # using delta as β
                dim=1
            )  # (T, k), sums to 1 over lines
            
            # Stage 2: Local geometric scores
            # geometric_score = -δ · d(x, line)
            dist = torch.norm(
                input_projected_translated - input_translated.unsqueeze(1), 
                dim=-1
            )  # (T, k, N)
            geometric_scores = -self.delta * dist  # (T, k, N)
            
            # Combine: log(π) + geometric_score
            # This implements: α ∝ π(θᵢ) · exp(-δ · d(x, θᵢ))
            log_line_weights = torch.log(line_weights + 1e-8).unsqueeze(2)  # (T, k, 1)
            combined_scores = geometric_scores + log_line_weights  # (T, k, N)
            
            # Normalize: softmax over lines for each point
            mass_input = torch.softmax(combined_scores, dim=1) / N  # (T, k, N)
        
        return mass_input, axis_coordinate


class DbTSW(TWConcurrentLines):
    def __init__(self, p=2, delta=2, device="cuda"):
        super().__init__(p=p, delta=delta, device=device, mass_division='distance_based')

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    # N = 32 * 32
    # M = 32 * 32
    # dn = dm = 128
    # ntrees = 2048
    # nlines = 2
    
    # N = 5
    # M = 5
    # dn = dm = 3
    # ntrees = 7
    # nlines = 2
    
    N = 50000
    M = 50000
    dn = dm = 1000
    ntrees = 100
    nlines = 10
    
    TW_obj = torch.compile(DbTSW())
    
    
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)
    
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    torch.cuda.reset_peak_memory_stats(device=None)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        tw = TW_obj(X, Y, theta, intercept)

    prof.export_chrome_trace("trace_concurrent.json")
    with open("profile_result_concurrent.txt", "w") as f:
        table_str = prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True)
        f.write(table_str)
        print(table_str)
    print(torch.cuda.max_memory_allocated(device=None) / 1024 / 1024)
