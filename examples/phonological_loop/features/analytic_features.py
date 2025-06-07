# phonological_loop/features/analytic_features.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Need F for unfold
from torch.fft import fft, ifft
import math # Import math for pi

class AnalyticSignalExtractor(nn.Module):
    """
    Computes the 5 analytic signal features described by Aisbett over
    overlapping frames: [A, θ', A², AA', A²θ']
    where A is envelope, θ' is instantaneous frequency.

    Input:  (B, L) raw waveform (L = samples)
    Output: (B, C, T) feature tensor (C=5 features, T = frames)
    """
    def __init__(self,
                 window_length: int = 512,
                 hop_length: int = 128,
                 eps: float = 1e-8):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.eps = eps
        # Register a Hann window buffer (optional but common for STFT-like processing)
        self.register_buffer('window', torch.hann_window(window_length))

    def _analytic_signal(self, x_frames):
        """Computes the analytic signal for frames."""
        # x_frames: (B, T, window_length)
        N = self.window_length
        # Apply window function
        windowed_frames = x_frames * self.window.to(x_frames.device)
        Xf = fft(windowed_frames, N, dim=-1) # FFT along the window dimension
        h = torch.zeros_like(Xf)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2
        # Result is analytic signal per frame
        return ifft(Xf * h, dim=-1) # (B, T, window_length) complex

    def _inst_freq(self, phase_angle_frames):
        """Calculate instantaneous frequency for frames."""
        # phase_angle_frames: (B, T, window_length)
        # Unwrap phase angle along the window dimension (last dimension) using manual method
        unwrapped_phase = self._manual_unwrap(phase_angle_frames, dim=-1) # Use manual unwrap
        # Compute the derivative using finite differences along the window dim
        # Pad beginning to maintain length
        inst_freq_frames = torch.diff(unwrapped_phase, dim=-1, prepend=unwrapped_phase[..., :1])
        # inst_freq_frames shape: (B, T, window_length)
        # We need a single value per frame, e.g., the mean or median frequency
        # Using mean for now
        return torch.mean(inst_freq_frames, dim=-1) # (B, T)

    def _manual_unwrap(self, phase_angle, dim=-1, period=2 * math.pi):
        """Manually unwraps phase angle along a given dimension."""
        # Calculate phase differences
        diff = torch.diff(phase_angle, dim=dim)

        # Identify jumps greater than pi
        jumps = (diff > period / 2).float() - (diff < -period / 2).float()

        # Calculate cumulative correction
        correction = torch.cumsum(jumps * period, dim=dim)

        # Apply correction - need to pad correction to match original phase shape
        unwrapped = phase_angle.clone()
        # Add correction to elements after the first one along the specified dimension
        slices = [slice(None)] * phase_angle.ndim
        slices[dim] = slice(1, None)
        unwrapped[tuple(slices)] -= correction

        return unwrapped

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape

        # 1. Create overlapping frames
        # unfold(dimension, size, step)
        x_frames = x.unfold(dimension=-1, size=self.window_length, step=self.hop_length)
        # x_frames shape: (B, T, window_length), where T is num_frames
        B, T, W = x_frames.shape

        # 2. Compute analytic signal per frame z(t) = x(t) + j*hilbert(x(t))
        z_frames = self._analytic_signal(x_frames) # (B, T, W) complex

        # 3. Compute Envelope A(t) = |z(t)| per frame
        A_frames = torch.abs(z_frames) + self.eps # (B, T, W)

        # 4. Compute Instantaneous Phase φ(t) = angle(z(t)) per frame
        # Use atan2 as torch.angle is not implemented for MPS
        phi_frames = torch.atan2(z_frames.imag, z_frames.real) # (B, T, W)

        # 5. Compute Instantaneous Frequency θ'(t) = dφ/dt (single value per frame)
        # _inst_freq now handles the unwrapping internally
        theta_prime = self._inst_freq(phi_frames) # (B, T)

        # 6. Compute Envelope Derivative A'(t) per frame using finite differences
        A_prime_frames = torch.diff(A_frames, dim=-1, prepend=A_frames[..., :1]) # (B, T, W)

        # 7. Compute features, taking mean over the window dim to get one value per frame
        A = torch.mean(A_frames, dim=-1) # (B, T)
        A_sq = torch.mean(A_frames**2, dim=-1) # (B, T)
        AA_prime = torch.mean(A_frames * A_prime_frames, dim=-1) # (B, T)
        # For A²θ', multiply the mean A² by the frame-level θ'
        A_sq_theta_prime = A_sq * theta_prime # (B, T)

        # Stack features: [A, θ', A², AA', A²θ']
        # Ensure all have shape (B, T) before stacking
        # Transpose at the end to get (B, C, T)
        features = torch.stack([A, theta_prime, A_sq, AA_prime, A_sq_theta_prime], dim=1) # (B, 5, T)

        return features