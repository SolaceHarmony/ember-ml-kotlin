"""
Harmonic wave processing components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

class FrequencyAnalyzer:
    """Analyzer for frequency components of signals."""
    
    def __init__(self, sampling_rate: float, window_size: int = 1024, overlap: float = 0.5):
        """
        Initialize frequency analyzer.

        Args:
            sampling_rate: Sampling rate in Hz
            window_size: FFT window size
            overlap: Window overlap ratio
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        
    def compute_spectrum(self, signal: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Compute frequency spectrum.

        Args:
            signal: Input signal tensor

        Returns:
            Frequency spectrum tensor
        """
        # Apply Hann window
        window = torch.hann_window(self.window_size)
        padded_signal = F.pad(signal, (0, self.window_size - signal.size(-1) % self.window_size))
        
        # Compute FFT
        spectrum = torch.fft.rfft(padded_signal * window)
        return torch.abs(spectrum)
        
    def find_peaks(self, signal: tensor.convert_to_tensor, threshold: float = 0.1, tolerance: float = 0.01) -> List[Dict[str, float]]:
        """
        Find peak frequencies.

        Args:
            signal: Input signal tensor
            threshold: Peak detection threshold
            tolerance: Frequency matching tolerance

        Returns:
            List of peak information dictionaries
        """
        spectrum = self.compute_spectrum(signal)
        freqs = torch.fft.rfftfreq(self.window_size, 1/self.sampling_rate)
        
        # Find peaks
        peaks = []
        for i in range(1, len(spectrum)-1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum[i] > threshold * torch.max(spectrum):
                    # Find closest frequency bin
                    freq = freqs[i]
                    # Round to nearest integer frequency
                    rounded_freq = round(freq.item())
                    peaks.append({
                        'frequency': rounded_freq,
                        'amplitude': spectrum[i].item() / torch.max(spectrum).item()
                    })
        
        # Merge peaks within tolerance
        merged_peaks = {}
        for peak in peaks:
            freq = peak['frequency']
            amp = peak['amplitude']
            found = False
            for existing_freq in list(merged_peaks.keys()):
                if abs(freq - existing_freq) <= tolerance:
                    if amp > merged_peaks[existing_freq]['amplitude']:
                        merged_peaks[existing_freq] = peak
                    found = True
                    break
            if not found:
                merged_peaks[freq] = peak
                
        return sorted(merged_peaks.values(), key=lambda x: x['amplitude'], reverse=True)
        
    def harmonic_ratio(self, signal: tensor.convert_to_tensor) -> float:
        """
        Compute harmonic to noise ratio.

        Args:
            signal: Input signal tensor

        Returns:
            Harmonic ratio value
        """
        spectrum = self.compute_spectrum(signal)
        peaks = self.find_peaks(signal)
        
        if not peaks:
            return 0.0
            
        # Sum peak amplitudes
        peak_sum = sum(p['amplitude'] for p in peaks)
        total_sum = stats.sum(spectrum).item()
        
        return peak_sum / total_sum if total_sum > 0 else 0.0

class WaveSynthesizer:
    """Synthesizer for harmonic wave generation."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize wave synthesizer.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def sine_wave(self,
                 frequency: float,
                 duration: float,
                 amplitude: float = 1.0,
                 phase: float = 0.0) -> tensor.convert_to_tensor:
        """
        Generate sine wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude
            phase: Initial phase in radians

        Returns:
            Sine wave tensor
        """
        t = torch.linspace(0, duration, int(duration * self.sampling_rate))
        return amplitude * torch.sin(2 * math.pi * frequency * t + phase)
        
    def harmonic_wave(self,
                     frequencies: List[float],
                     amplitudes: List[float],
                     duration: float) -> tensor.convert_to_tensor:
        """
        Generate wave with harmonics.

        Args:
            frequencies: List of frequencies
            amplitudes: List of amplitudes
            duration: Signal duration in seconds

        Returns:
            Harmonic wave tensor
        """
        # Normalize amplitudes to ensure sum is at most 1
        total_amp = sum(abs(a) for a in amplitudes)
        if total_amp > 0:
            norm_amplitudes = [a / total_amp for a in amplitudes]
        else:
            norm_amplitudes = amplitudes
            
        wave = torch.zeros(int(duration * self.sampling_rate))
        for freq, amp in zip(frequencies, norm_amplitudes):
            wave += self.sine_wave(freq, duration, amp)
            
        return wave
        
    def apply_envelope(self,
                      wave: tensor.convert_to_tensor,
                      envelope: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Apply amplitude envelope.

        Args:
            wave: Input wave tensor
            envelope: Amplitude envelope tensor

        Returns:
            Modulated wave tensor
        """
        if len(envelope) != len(wave):
            envelope = F.interpolate(
                envelope.view(1, 1, -1),
                size=len(wave),
                mode='linear'
            ).squeeze()
            
        # Ensure envelope is positive and normalized
        envelope = torch.abs(envelope)
        envelope = envelope / torch.max(envelope)
        
        # Apply envelope while preserving wave sign
        modulated = wave * envelope
        
        # Ensure modulated signal doesn't exceed original in absolute terms
        modulated = torch.sign(wave) * torch.minimum(torch.abs(modulated), torch.abs(wave))
            
        return modulated

class HarmonicProcessor:
    """Processor for harmonic signal analysis and manipulation."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize harmonic processor.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.analyzer = FrequencyAnalyzer(sampling_rate)
        self.synthesizer = WaveSynthesizer(sampling_rate)
        
    def decompose(self, signal: tensor.convert_to_tensor) -> Dict[str, List[float]]:
        """
        Decompose signal into harmonic components.

        Args:
            signal: Input signal tensor

        Returns:
            Dictionary of frequency components
        """
        peaks = self.analyzer.find_peaks(signal)
        return {
            'frequencies': [p['frequency'] for p in peaks],
            'amplitudes': [p['amplitude'] for p in peaks]
        }
        
    def reconstruct(self,
                   frequencies: List[float],
                   amplitudes: List[float],
                   duration: float) -> tensor.convert_to_tensor:
        """
        Reconstruct signal from components.

        Args:
            frequencies: List of frequencies
            amplitudes: List of amplitudes
            duration: Signal duration in seconds

        Returns:
            Reconstructed signal tensor
        """
        # Sort by frequency to maintain phase relationships
        freq_amp = sorted(zip(frequencies, amplitudes), key=lambda x: x[0])
        frequencies = [f for f, _ in freq_amp]
        amplitudes = [a for _, a in freq_amp]
        
        # Generate signal
        signal = self.synthesizer.harmonic_wave(frequencies, amplitudes, duration)
        
        # Scale to match original signal amplitude
        max_amp = max(abs(a) for a in amplitudes) if amplitudes else 1.0
        if max_amp > 0:
            signal = signal * max_amp
            
        return signal
        
    def filter_harmonics(self,
                        signal: tensor.convert_to_tensor,
                        keep_frequencies: List[float],
                        tolerance: float = 0.1) -> tensor.convert_to_tensor:
        """
        Filter specific harmonics.

        Args:
            signal: Input signal tensor
            keep_frequencies: Frequencies to keep
            tolerance: Frequency matching tolerance

        Returns:
            Filtered signal tensor
        """
        # Get original signal amplitude
        original_max = torch.max(torch.abs(signal))
        
        # Generate filtered signal
        duration = len(signal) / self.sampling_rate
        filtered = torch.zeros_like(signal)
        
        # Keep only matching frequencies
        for freq in keep_frequencies:
            # Scale amplitude to match original signal
            filtered += self.synthesizer.sine_wave(freq, duration, original_max)
            
        # Scale to match original signal
        filtered = filtered / len(keep_frequencies)
            
        return filtered
