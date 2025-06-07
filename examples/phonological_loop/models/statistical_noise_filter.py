# phonological_loop/models/statistical_noise_filter.py
import torch
import torch.nn as nn
import math

class LogDomainNoiseSuppression(nn.Module):
    """
    Histogram-based, analytic-pdf-matched noise suppressor based on Aisbett's principles.

    Compares observed feature distributions (via running histograms) to theoretical
    noise distributions to create a mask that attenuates noise-like samples.

    Inputs:  (B, C, L) feature streams, expected C=5 for [A, θ', A², AA', A²θ']
    Outputs: (B, C, L) masked feature streams
    """
    def __init__(self,
                 num_features: int = 5,
                 bins: int = 256,
                 hist_range_max: float = 8.0, # Max value for histogram normalization/binning
                 kalman_alpha: float = 0.02,  # EMA update factor for histogram
                 mask_threshold: float = -2.0, # Less strict threshold (tuneable)
                 eps: float = 1e-8):
        super().__init__()

        if num_features != 5:
            # Currently hardcoded for the 5 specific features
            raise ValueError("This implementation currently requires num_features=5")

        self.num_features = num_features
        self.bins = bins
        self.alpha = kalman_alpha
        self.mask_threshold = mask_threshold
        self.hist_range_max = hist_range_max
        self.eps = eps

        # Running histogram buffer (initialized flat)
        self.register_buffer("hist", torch.ones(num_features, bins))

        # Pre-compute the *theoretical* log-pdf for each variable on a grid
        # Grid represents the normalized feature values (0 to hist_range_max)
        grid = torch.linspace(self.eps, self.hist_range_max, bins)

        # Theoretical log-PDFs based on Aisbett (approximations)
        # Ensure shapes are (num_features, bins)
        logp_ref = torch.stack([
            torch.log(grid) - 0.5 * grid**2, # Log-Rayleigh for A (approximation, check Aisbett for exact form if needed)
            -1.5 * torch.log1p((grid / (hist_range_max/2))**2), # Heavy-tail for θ' (scaled grid)
            -grid, # Log-Exponential for A²
            -0.5 * grid**2, # Log-Gaussian approx for AA'
            -0.5 * grid**2, # Log-Gaussian approx for A²θ'
        ], dim=0) # Shape: (5, bins)

        # Add small epsilon to prevent log(0) in theoretical PDFs
        self.register_buffer("logp_ref", logp_ref + self.eps)

    def forward(self, x):
        # x: (B, C, L) feature tensor, C should be self.num_features (5)
        B, C, L = x.shape
        if C != self.num_features:
            raise ValueError(f"Input feature dimension C={C} does not match expected num_features={self.num_features}")

        # Use absolute value for histogramming as features can be negative (e.g., θ', AA')
        # Reshape for easier processing: (C, B*L)
        flat_abs = x.reshape(C, -1).abs()

        # --- 1) Update running histogram ---
        # Normalize features to range [0, hist_range_max] for binning
        # Use a robust max estimate (e.g., 99th percentile) if needed, or clamp
        max_vals = torch.quantile(flat_abs, 0.99, dim=-1, keepdim=True).clamp(min=self.eps)
        normalized_flat = (flat_abs / max_vals) * self.hist_range_max

        # Calculate bin indices
        idx = torch.clamp(normalized_flat, 0, self.hist_range_max - self.eps)
        idx = (idx / self.hist_range_max * (self.bins - 1)).long() # Shape: (C, B*L)

        # Compute histogram for the current batch
        # Use scatter_add_ for potential efficiency on GPU, fallback to bincount
        new_hist = torch.zeros_like(self.hist)
        for c in range(C):
             # Ensure indices are within [0, bins-1]
            clamped_idx_c = torch.clamp(idx[c], 0, self.bins - 1)
            new_hist[c].scatter_add_(0, clamped_idx_c, torch.ones_like(clamped_idx_c, dtype=new_hist.dtype))
            # Alternative: new_hist[c] = torch.bincount(clamped_idx_c, minlength=self.bins).float()

        # EMA update for the running histogram
        # Ensure update happens on the correct device
        self.hist = self.hist.to(x.device)
        self.hist = (1 - self.alpha) * self.hist + self.alpha * new_hist

        # Add smoothing factor and normalize to get observed log-probability
        smoothed_hist = self.hist + self.eps
        logp_obs = torch.log(smoothed_hist / smoothed_hist.sum(-1, keepdim=True)) # Shape: (C, bins)

        # --- 2) Compute Log-Likelihood Ratio (Λ) ---
        # Use the same bin indices calculated earlier
        # Gather requires indices to be LongTensor
        bucket = idx # Shape: (C, B*L)

        # Ensure logp_ref and logp_obs are on the correct device
        logp_ref_dev = self.logp_ref.to(x.device)
        logp_obs_dev = logp_obs.to(x.device)

        # Gather log probabilities for each sample based on its bin index
        # Ensure bucket indices are clamped correctly before gather
        clamped_bucket = torch.clamp(bucket, 0, self.bins - 1)
        logp_ref_samples = logp_ref_dev.gather(-1, clamped_bucket) # Shape: (C, B*L)
        logp_obs_samples = logp_obs_dev.gather(-1, clamped_bucket) # Shape: (C, B*L)

        # Calculate Lambda (log-likelihood ratio)
        Lambda = logp_ref_samples - logp_obs_samples # Shape: (C, B*L)

        # --- 3) Create and Apply Mask ---
        # Reshape Lambda back to (B, C, L)
        Lambda = Lambda.reshape(B, C, L)

        # Create soft mask: Attenuate samples where Lambda is close to 0 (more likely noise)
        # Use sigmoid centered around the threshold. Beta controls steepness.
        beta = 1.0 # Reduced steepness factor (tuneable)
        mask = torch.sigmoid(-beta * (Lambda - self.mask_threshold)) # Using self.mask_threshold (-2.0)

        # Apply soft mask
        return x * mask