import torch

def analytic_signal(waveform: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
    """Computes the analytic signal using FFT."""
    # Ensure waveform is at least 2D [batch, seq_len]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    N = waveform.shape[-1]
    X = torch.fft.fft(waveform, N, dim=-1)
    h = torch.zeros_like(X)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2
    analytic = torch.fft.ifft(X * h, dim=-1)
    return analytic # Complex tensor: I + jQ

def add_noise(signal: tensor.convert_to_tensor, snr_db: float) -> tensor.convert_to_tensor:
    """Adds Gaussian noise to a signal tensor to achieve a specific SNR in dB."""
    signal_power = torch.mean(signal ** 2, dim=-1, keepdim=True)
    # Handle potential zero power signals
    signal_power = torch.clamp(signal_power, min=1e-10)
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10**(SNR/10))
    noise_power = signal_power / (10**(snr_db / 10))
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    noisy_signal = signal + noise
    return noisy_signal