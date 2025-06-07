"""
Wave analysis utilities.

This module provides utilities for analyzing wave signals.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from scipy import signal
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike # Corrected import
# Try to import librosa, but don't fail if it's not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

def compute_fft(wave: TensorLike, sample_rate: int = 44100) -> Tuple[TensorLike, TensorLike]:
    """
    Compute the FFT of a wave signal.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (frequencies, magnitudes)
    """
    n = len(wave)
    fft = linearalg.rfft(wave)
    magnitudes = ops.abs(fft) / n
    frequencies = linearalg.rfftfreq(n, 1 / sample_rate)
    return frequencies, magnitudes

def compute_stft(wave: TensorLike, sample_rate: int = 44100, 
                 window_size: int = 2048, hop_length: int = 512) -> Tuple[TensorLike, TensorLike, TensorLike]:
    """
    Compute the Short-Time Fourier Transform of a wave signal.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        window_size: Window size in samples
        hop_length: Hop length in samples
        
    Returns:
        Tuple of (times, frequencies, spectrogram)
    """
    f, t, Zxx = signal.stft(wave, fs=sample_rate, nperseg=window_size, noverlap=window_size - hop_length)
    return t, f, ops.abs(Zxx)

def compute_mfcc(wave: TensorLike, sample_rate: int = 44100, n_mfcc: int = 13) -> TensorLike:
    """
    Compute Mel-frequency cepstral coefficients.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCCs to return
        
    Returns:
        MFCCs
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for MFCC computation")
    return librosa.feature.mfcc(y=wave, sr=sample_rate, n_mfcc=n_mfcc)

def compute_spectral_centroid(wave: TensorLike, sample_rate: int = 44100) -> TensorLike:
    """
    Compute spectral centroid.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Spectral centroid
    """
    if not LIBROSA_AVAILABLE:
        # Fallback implementation using FFT
        frequencies, magnitudes = compute_fft(wave, sample_rate)
        centroid = stats.sum(frequencies * magnitudes) / stats.sum(magnitudes) if stats.sum(magnitudes) > 0 else 0
        return tensor.convert_to_tensor([centroid])
    return librosa.feature.spectral_centroid(y=wave, sr=sample_rate)[0]

def compute_spectral_bandwidth(wave: TensorLike, sample_rate: int = 44100) -> TensorLike:
    """
    Compute spectral bandwidth.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Spectral bandwidth
    """
    if not LIBROSA_AVAILABLE:
        # Fallback implementation using FFT
        frequencies, magnitudes = compute_fft(wave, sample_rate)
        centroid = stats.sum(frequencies * magnitudes) / stats.sum(magnitudes) if stats.sum(magnitudes) > 0 else 0
        bandwidth = ops.sqrt(stats.sum(((frequencies - centroid) ** 2) * magnitudes) / stats.sum(magnitudes)) if stats.sum(magnitudes) > 0 else 0
        return tensor.convert_to_tensor([bandwidth])
    return librosa.feature.spectral_bandwidth(y=wave, sr=sample_rate)[0]

def compute_spectral_contrast(wave: TensorLike, sample_rate: int = 44100) -> TensorLike:
    """
    Compute spectral contrast.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Spectral contrast
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for spectral contrast computation")
    return librosa.feature.spectral_contrast(y=wave, sr=sample_rate)

def compute_spectral_rolloff(wave: TensorLike, sample_rate: int = 44100) -> TensorLike:
    """
    Compute spectral rolloff.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Spectral rolloff
    """
    if not LIBROSA_AVAILABLE:
        # Fallback implementation using FFT
        frequencies, magnitudes = compute_fft(wave, sample_rate)
        cumsum = np.cumsum(magnitudes)
        rolloff_point = 0.85 * cumsum[-1]  # Default rolloff at 85%
        rolloff_idx = ops.where(cumsum >= rolloff_point)[0][0]
        return tensor.convert_to_tensor([frequencies[rolloff_idx]])
    return librosa.feature.spectral_rolloff(y=wave, sr=sample_rate)[0]

def compute_zero_crossing_rate(wave: TensorLike) -> float:
    """
    Compute zero crossing rate.
    
    Args:
        wave: Wave signal
        
    Returns:
        Zero crossing rate
    """
    if not LIBROSA_AVAILABLE:
        # Fallback implementation
        zero_crossings = stats.sum(ops.abs(np.diff(np.signbit(wave))))
        return zero_crossings / len(wave)
    return stats.mean(librosa.feature.zero_crossing_rate(wave))

def compute_rms(wave: TensorLike) -> float:
    """
    Compute root mean square.
    
    Args:
        wave: Wave signal
        
    Returns:
        Root mean square
    """
    return ops.sqrt(stats.mean(np.square(wave)))

def compute_peak_amplitude(wave: TensorLike) -> float:
    """
    Compute peak amplitude.
    
    Args:
        wave: Wave signal
        
    Returns:
        Peak amplitude
    """
    return stats.max(ops.abs(wave))

def compute_crest_factor(wave: TensorLike) -> float:
    """
    Compute crest factor.
    
    Args:
        wave: Wave signal
        
    Returns:
        Crest factor
    """
    rms = compute_rms(wave)
    peak = compute_peak_amplitude(wave)
    return peak / rms if rms > 0 else 0

def compute_dominant_frequency(wave: TensorLike, sample_rate: int = 44100) -> float:
    """
    Compute dominant frequency.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Dominant frequency in Hz
    """
    frequencies, magnitudes = compute_fft(wave, sample_rate)
    return frequencies[np.argmax(magnitudes)]

def compute_harmonic_ratio(wave: TensorLike, sample_rate: int = 44100) -> float:
    """
    Compute harmonic ratio.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Harmonic ratio
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for harmonic ratio computation")
    return stats.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(wave), sr=sample_rate))

def compute_wave_features(wave: TensorLike, sample_rate: int = 44100) -> Dict[str, float]:
    """
    Compute various features of a wave signal.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary of features
    """
    features = {
        'rms': compute_rms(wave),
        'peak_amplitude': compute_peak_amplitude(wave),
        'crest_factor': compute_crest_factor(wave),
        'dominant_frequency': compute_dominant_frequency(wave, sample_rate),
    }
    
    # Add librosa-dependent features if available
    if LIBROSA_AVAILABLE:
        try:
            features.update({
                'zero_crossing_rate': compute_zero_crossing_rate(wave),
                'spectral_centroid': stats.mean(compute_spectral_centroid(wave, sample_rate)),
                'spectral_bandwidth': stats.mean(compute_spectral_bandwidth(wave, sample_rate)),
                'spectral_rolloff': stats.mean(compute_spectral_rolloff(wave, sample_rate))
            })
        except Exception as e:
            print(f"Warning: Could not compute some librosa-dependent features: {e}")
    
    return features