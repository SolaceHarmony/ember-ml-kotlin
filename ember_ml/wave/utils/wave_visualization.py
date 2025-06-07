"""
Wave visualization utilities.

This module provides utilities for visualizing wave signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Dict
from matplotlib.figure import Figure
from ember_ml.nn.tensor.types import TensorLike # Added import
from .wave_analysis import compute_fft, compute_stft
from ember_ml import ops
# Try to import librosa, but don't fail if it's not available
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

def plot_waveform(wave: TensorLike, sample_rate: int = 44100, title: str = "Waveform") -> Figure:
    """
    Plot a waveform.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    time = tensor.arange(len(wave)) / sample_rate
    ax.plot(time, wave)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_spectrum(wave: TensorLike, sample_rate: int = 44100, title: str = "Spectrum") -> Figure:
    """
    Plot a spectrum.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    frequencies, magnitudes = compute_fft(wave, sample_rate)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frequencies, magnitudes)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True)
    
    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    ax.set_xlim([20, sample_rate / 2])  # Limit to audible range
    
    fig.tight_layout()
    return fig

def plot_spectrogram(wave: TensorLike, sample_rate: int = 44100, 
                     window_size: int = 2048, hop_length: int = 512,
                     title: str = "Spectrogram") -> Figure:
    """
    Plot a spectrogram.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        window_size: Window size in samples
        hop_length: Hop length in samples
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    times, frequencies, spectrogram = compute_stft(wave, sample_rate, window_size, hop_length)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(times, frequencies, spectrogram, shading='gouraud', cmap='viridis')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Magnitude')
    
    # Set y-axis to log scale for better visualization
    ax.set_yscale('log')
    ax.set_ylim([20, sample_rate / 2])  # Limit to audible range
    
    fig.tight_layout()
    return fig

def plot_mel_spectrogram(wave: TensorLike, sample_rate: int = 44100,
                         n_fft: int = 2048, hop_length: int = 512,
                         n_mels: int = 128, title: str = "Mel Spectrogram") -> Figure:
    """
    Plot a mel spectrogram.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples
        n_mels: Number of mel bands
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for mel spectrogram visualization")
        
    mel_spec = librosa.feature.melspectrogram(y=wave, sr=sample_rate, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=stats.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                  sr=sample_rate, hop_length=hop_length, ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='dB')
    
    fig.tight_layout()
    return fig

def plot_chromagram(wave: TensorLike, sample_rate: int = 44100,
                   n_fft: int = 2048, hop_length: int = 512,
                   title: str = "Chromagram") -> Figure:
    """
    Plot a chromagram.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for chromagram visualization")
        
    chromagram = librosa.feature.chroma_stft(y=wave, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma',
                                  sr=sample_rate, hop_length=hop_length, ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label='Magnitude')
    
    fig.tight_layout()
    return fig

def plot_mfcc(wave: TensorLike, sample_rate: int = 44100,
             n_mfcc: int = 13, title: str = "MFCC") -> Figure:
    """
    Plot MFCCs.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCCs
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for MFCC visualization")
        
    mfccs = librosa.feature.mfcc(y=wave, sr=sample_rate, n_mfcc=n_mfcc)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label='Magnitude')
    
    fig.tight_layout()
    return fig

def plot_wave_features(wave: TensorLike, sample_rate: int = 44100) -> Dict[str, Figure]:
    """
    Plot various features of a wave signal.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary of figures
    """
    features = {
        'waveform': plot_waveform(wave, sample_rate),
        'spectrum': plot_spectrum(wave, sample_rate),
        'spectrogram': plot_spectrogram(wave, sample_rate),
    }
    
    # Add librosa-dependent features if available
    if LIBROSA_AVAILABLE:
        try:
            features.update({
                'mel_spectrogram': plot_mel_spectrogram(wave, sample_rate),
                'chromagram': plot_chromagram(wave, sample_rate),
                'mfcc': plot_mfcc(wave, sample_rate)
            })
        except Exception as e:
            print(f"Warning: Could not generate some librosa-dependent plots: {e}")
    
    return features

def plot_wave_comparison(waves: List[TensorLike], labels: List[str], 
                        sample_rate: int = 44100, title: str = "Wave Comparison") -> Figure:
    """
    Plot a comparison of multiple wave signals.
    
    Args:
        waves: List of wave signals
        labels: List of labels for each wave
        sample_rate: Sample rate in Hz
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_waves = len(waves)
    fig, axes = plt.subplots(n_waves, 1, figsize=(10, 3 * n_waves), sharex=True)
    
    if n_waves == 1:
        axes = [axes]
    
    for i, (wave, label) in enumerate(zip(waves, labels)):
        time = tensor.arange(len(wave)) / sample_rate
        axes[i].plot(time, wave)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(label)
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig