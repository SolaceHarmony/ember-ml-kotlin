# phonological_loop/models/noise_filter.py
import math
import torch
import torch.nn as nn

class LogNoiseFilter(nn.Module):
    """
    Log-Domain Noise Filter that applies log transform and filters features
    based on deviation from estimated noise statistics.

    As described in the paper:
    1. Apply element-wise logarithm to features
    2. Estimate noise statistics (mean, std) for each feature
    3. Compute salience mask where features deviate significantly from noise
    4. Apply mask to filter out non-salient features
    """
    def __init__(self,
                 noise_est_samples: int = 50, # Default changed to frames
                 threshold: float = 3.0,
                 eps: float = 1e-8):
        """
        Args:
            noise_est_samples (int): Number of initial frames used to estimate noise stats.
            threshold (float): Number of standard deviations for thresholding (k).
            eps (float): Small value for log stability.
        """
        super().__init__()
        self.noise_est_samples = noise_est_samples
        self.threshold = threshold
        self.eps = eps

        # Buffers to store estimated noise statistics (not trainable parameters)
        self.register_buffer('noise_mean', torch.zeros(1)) # Shape will be updated
        self.register_buffer('noise_std', torch.ones(1))  # Shape will be updated

        # Use Python boolean instead of tensor for tracking state
        self.stats_estimated = False

    @torch.no_grad() # Statistics estimation should not involve gradients
    def estimate_noise_stats(self, features: tensor.convert_to_tensor):
        """Estimates mean and std of log features from initial noise segment."""
        # Determine actual number of samples to use, capped by available frames T
        T = features.shape[1]
        actual_noise_samples = min(T, self.noise_est_samples) # Define before use

        if actual_noise_samples < 10: # Require at least 10 frames for estimation
            print(f"Warning: Not enough frames ({T}) to estimate noise stats reliably using {self.noise_est_samples} requested samples. Using default stats (mean=0, std=1).")
            # Set default stats (mean=0, std=1)
            if self.noise_mean.shape != (features.shape[-1],):
                 self.register_buffer('noise_mean', torch.zeros(features.shape[-1], device=features.device))
                 self.register_buffer('noise_std', torch.ones(features.shape[-1], device=features.device))
            else:
                 self.noise_mean.zero_()
                 self.noise_std.fill_(1.0)
            self.stats_estimated = True
            return

        # Use the determined number of samples
        noise_segment = features[:, :actual_noise_samples, :]
        # Apply log transform (handle potential non-positive values)
        log_noise = torch.log(torch.clamp(noise_segment, min=self.eps))

        # Calculate mean and std across batch and time dimensions
        computed_mean = torch.mean(log_noise, dim=(0, 1))
        computed_std = torch.std(log_noise, dim=(0, 1))

        # Resize buffers if needed
        if self.noise_mean.shape != computed_mean.shape:
            # Re-register buffers with new shape
            self.register_buffer('noise_mean', computed_mean.clone())
            self.register_buffer('noise_std', computed_std.clone())
        else:
            # Use in-place operations to maintain device placement
            self.noise_mean.copy_(computed_mean)
            self.noise_std.copy_(computed_std)

        # Avoid very small std deviation (in-place) - Increase minimum clamp value
        self.noise_std.clamp_(min=1e-5) # Use a larger minimum value than self.eps

        self.stats_estimated = True
        print(f"Estimated noise stats: Mean={self.noise_mean.cpu()}, Std={self.noise_std.cpu()}")

    def forward(self, features: tensor.convert_to_tensor) -> tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """
        Args:
            features (tensor.convert_to_tensor): Instantaneous Aisbett features [batch, seq_len, feature_dim].

        Returns:
            tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
                - Filtered features [batch, seq_len, feature_dim]
                - Salience mask [batch, seq_len, feature_dim] (soft values between 0 and 1)
        """
        # Ensure noise stats are estimated
        if not self.stats_estimated:
            if self.training:
                # Estimate stats during training
                self.estimate_noise_stats(features)
            else:
                # In eval mode, if stats weren't estimated, raise error
                raise RuntimeError(
                    "Noise statistics have not been estimated. The model must be trained "
                    "before being used in evaluation mode."
                )

        # Apply log transform with higher epsilon for mixed precision safety
        safe_eps = 1e-5  # Safer for FP16/BF16
        log_features = torch.log(torch.clamp(features, min=safe_eps))

        # Calculate salience mask based on deviation from noise statistics
        # Expand noise_mean and noise_std to match features dimensions
        expanded_mean = self.noise_mean.view(1, 1, -1)

        # Avoid division by very small std values
        safe_std = torch.where(
            self.noise_std < 1e-4,
            torch.full_like(self.noise_std, 1e-4),
            self.noise_std
        )
        expanded_std = safe_std.view(1, 1, -1)

        # Compute deviation in terms of standard deviations
        deviation = torch.abs(log_features - expanded_mean) / expanded_std

        # Create soft mask using sigmoid with very steep slope
        # This makes it behave almost like a hard threshold but with gradients
        # log(9) makes sigmoid(τ)=0.5, sigmoid(τ+1)=0.9
        # Using 6.9 (log(1000)) makes sigmoid(τ+0.3)=0.97, effectively hard but differentiable
        steepness = 6.9  # log(1000)
        mask = torch.sigmoid(steepness * (deviation - self.threshold))

        # Apply mask to original features (not log-transformed)
        filtered_features = features * mask

        # For backward compatibility, also return the mask
        return filtered_features, mask

    def reset(self):
        """Reset the noise filter state"""
        # Reset to initial state (uninitialized)
        if self.noise_mean.shape != (1,):
            # Re-register buffers with initial shape
            self.register_buffer('noise_mean', torch.zeros(1, device=self.noise_mean.device))
            self.register_buffer('noise_std', torch.ones(1, device=self.noise_std.device))
        else:
            # Use in-place operations
            self.noise_mean.zero_()
            self.noise_std.fill_(1.0)

        self.stats_estimated = False