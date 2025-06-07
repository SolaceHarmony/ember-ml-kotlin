import torch
import torch.nn as nn
from phonological_loop.utils.signal_processing import analytic_signal

class AisbettFeatureExtractorMeanVar(nn.Module):
    """
    Extracts instantaneous Aisbett features (A^2, AA', A^2*theta')
    and calculates their mean and variance over sliding windows.
    """
    def __init__(self, hop_length=128, window_length=512, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.num_base_features = 3
        self.hop_length = hop_length
        self.window_length = window_length

        # Pooling layer for calculating the mean
        padding = max(0, (self.window_length - 1) // 2)
        self.mean_pool = nn.AvgPool1d(kernel_size=self.window_length, stride=self.hop_length, padding=padding)

    def forward(self, waveform: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Args:
            waveform (tensor.convert_to_tensor): Input waveform tensor of shape [batch, seq_len].

        Returns:
            tensor.convert_to_tensor: Mean and Variance of Aisbett features.
                          Shape [batch, time_frames, num_base_features * 2].
        """
        # 1. Calculate Instantaneous Features
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim > 2:
            waveform = waveform.squeeze(-1)

        analytic = analytic_signal(waveform)
        I = analytic.real
        Q = analytic.imag
        grad_I = torch.gradient(I, dim=-1, edge_order=1)[0]
        grad_Q = torch.gradient(Q, dim=-1, edge_order=1)[0]

        f1 = I**2 + Q**2
        f2 = I * grad_I + Q * grad_Q
        f3 = I * grad_Q - Q * grad_I

        # features_inst shape: [batch, seq_len, num_base_features=3]
        features_inst = torch.stack([f1, f2, f3], dim=-1)

        # 2. Calculate Mean and Variance over sliding windows
        # Permute for AvgPool1d: [batch, num_base_features, seq_len]
        features_permuted = features_inst.permute(0, 2, 1)

        # Calculate Mean using AvgPool1d
        # mean_features shape: [batch, num_base_features, time_frames]
        mean_features = self.mean_pool(features_permuted)

        # Calculate Variance: Var(X) = E[X^2] - (E[X])^2
        # Calculate E[X^2] using AvgPool1d on squared features
        mean_features_sq = self.mean_pool(features_permuted**2)
        # Variance shape: [batch, num_base_features, time_frames]
        variance_features = torch.clamp(mean_features_sq - mean_features**2, min=self.eps) # Use eps for clamp

        # 3. Concatenate Mean and Variance
        # combined shape: [batch, num_base_features * 2, time_frames]
        combined_features = torch.cat((mean_features, variance_features), dim=1)

        # 4. Permute back: [batch, time_frames, num_base_features * 2]
        output_features = combined_features.permute(0, 2, 1)

        return output_features

# Removed redundant MeanVariancePool class

class LogNoiseFilter(nn.Module):
    """
    Applies log transform and filters features based on deviation
    from estimated noise statistics.
    """
    def __init__(self, noise_est_samples: int = 16000, k: float = 3.0, eps: float = 1e-8):
        """
        Args:
            noise_est_samples (int): Number of initial samples used to estimate noise stats.
            k (float): Number of standard deviations for thresholding.
            eps (float): Small value for log stability.
        """
        super().__init__()
        self.noise_est_samples = noise_est_samples
        self.k = k
        self.eps = eps
        # Buffers to store estimated noise statistics (not trainable parameters)
        self.register_buffer('noise_mean', torch.zeros(1)) # Shape will be updated
        self.register_buffer('noise_std', torch.ones(1))  # Shape will be updated
        self.register_buffer('stats_estimated', tensor.convert_to_tensor(False))

    @torch.no_grad() # Statistics estimation should not involve gradients
    def estimate_noise_stats(self, features: tensor.convert_to_tensor):
        """Estimates mean and std of log features from initial noise segment."""
        if features.shape[1] < self.noise_est_samples:
            print(f"Warning: Not enough samples ({features.shape[1]}) to estimate noise stats using {self.noise_est_samples} samples. Using default stats.")
            # Keep default stats (mean=0, std=1) or handle differently
            # For safety, let's set std very high to avoid filtering if estimation fails
            self.noise_mean = torch.zeros(features.shape[-1], device=features.device)
            self.noise_std = torch.ones(features.shape[-1], device=features.device) * 1e6 # Effectively disable filtering
            self.stats_estimated = tensor.convert_to_tensor(True, device=features.device)
            return

        noise_segment = features[:, :self.noise_est_samples, :]
        # Apply log transform (handle potential non-positive values)
        log_noise = torch.log(torch.clamp(noise_segment, min=self.eps))
        # Calculate mean and std across batch and time dimensions
        self.noise_mean = torch.mean(log_noise, dim=(0, 1))
        self.noise_std = torch.std(log_noise, dim=(0, 1))
        # Avoid zero std deviation
        self.noise_std = torch.clamp(self.noise_std, min=self.eps)
        self.stats_estimated = tensor.convert_to_tensor(True, device=features.device)
        print(f"Estimated noise stats: Mean={self.noise_mean.cpu()}, Std={self.noise_std.cpu()}")

    def forward(self, features: tensor.convert_to_tensor) -> tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """
        Args:
            features (tensor.convert_to_tensor): Instantaneous Aisbett features [batch, seq_len, num_base_features].

        Returns:
            tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
                - Filtered features [batch, seq_len, num_base_features]
                - Salience mask [batch, seq_len, num_base_features] (binary 0 or 1)
        """
        if not self.stats_estimated and self.training: # Estimate only once during training
             self.estimate_noise_stats(features)
        elif not self.stats_estimated and not self.training:
             # In eval mode, if stats weren't estimated, use defaults or raise error
             print("Warning: Noise stats not estimated in eval mode. Using potentially inaccurate defaults.")
        
        # Apply log transform (handle potential non-positive values)
        log_features = torch.log(torch.clamp(features, min=self.eps))
        
        # Calculate salience mask based on deviation from noise statistics
        # Expand noise_mean and noise_std to match features dimensions
        expanded_mean = self.noise_mean.view(1, 1, -1)
        expanded_std = self.noise_std.view(1, 1, -1)
        
        # Compute deviation in terms of standard deviations
        deviation = torch.abs(log_features - expanded_mean) / expanded_std
        
        # Create binary mask where deviation exceeds threshold k
        mask = (deviation > self.k).float()
        
        # Apply mask to original features (not log-transformed)
        filtered_features = features * mask
        
        return filtered_features, mask