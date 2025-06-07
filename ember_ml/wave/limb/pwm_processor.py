"""
PWM (Pulse Width Modulation) signal processing for wave segments.
Handles conversion between PCM and PWM representations.
"""

import numpy as np
from typing import Tuple, Optional

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from .hpc_limb import HPCLimb

class PWMProcessor:
    """
    Handles conversion between PCM samples and PWM signals using HPC limb arithmetic.
    Supports various bit depths and carrier frequencies.
    """
    
    def __init__(self, 
                 bits_per_block: int = 4,
                 carrier_freq: int = 24000,
                 sample_rate: int = 24000):
        """
        Initialize PWM processor.
        
        Args:
            bits_per_block: Number of bits for PWM quantization
            carrier_freq: PWM carrier frequency in Hz
            sample_rate: Audio sample rate in Hz
        """
        self.bits = bits_per_block
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        
        # Calculate samples per PWM period
        self.samples_per_period = max(1, int(sample_rate / carrier_freq))
        
        # Number of possible PWM levels
        self.levels = 2 ** bits_per_block
        
    def pcm_to_pwm(self, pcm_data: TensorLike) -> TensorLike:
        """
        Convert PCM samples to PWM representation.
        
        Args:
            pcm_data: Input PCM samples (16-bit)
            
        Returns:
            PWM signal as binary values
        """
        # Create output array
        pwm_signal = tensor.zeros_like(pcm_data)
        
        # Process each block
        for i in range(0, len(pcm_data), self.samples_per_period):
            # Get block of samples
            block = pcm_data[i:i + self.samples_per_period]
            if len(block) < self.samples_per_period:
                break
                
            # Calculate average amplitude for block
            avg_amplitude = stats.mean(block)
            
            # Convert to duty cycle [0,1]
            duty_cycle = (avg_amplitude + 32768) / 65536
            
            # Quantize duty cycle to available levels
            quantized_duty = np.round(duty_cycle * (self.levels - 1)) / (self.levels - 1)
            
            # Generate PWM pattern
            high_samples = int(quantized_duty * self.samples_per_period)
            pwm_signal[i:i + high_samples] = 32767
            pwm_signal[i + high_samples:i + self.samples_per_period] = -32768
            
        return pwm_signal
        
    def pwm_to_pcm(self, pwm_signal: TensorLike) -> TensorLike:
        """
        Convert PWM signal back to PCM samples.
        
        Args:
            pwm_signal: Input PWM signal
            
        Returns:
            Reconstructed PCM samples
        """
        # Create output array
        pcm_out = tensor.zeros_like(pwm_signal)
        
        # Process each block
        for i in range(0, len(pwm_signal), self.samples_per_period):
            block = pwm_signal[i:i + self.samples_per_period]
            if len(block) < self.samples_per_period:
                break
                
            # Count high samples to determine duty cycle
            high_count = stats.sum(block > 0)
            duty_cycle = high_count / self.samples_per_period
            
            # Convert duty cycle back to PCM value
            pcm_value = int((duty_cycle * 65536) - 32768)
            pcm_out[i:i + self.samples_per_period] = pcm_value
            
        return pcm_out
        
    def analyze_pwm_signal(self, pwm_signal: TensorLike) -> dict:
        """
        Analyze PWM signal characteristics.
        
        Args:
            pwm_signal: PWM signal to analyze
            
        Returns:
            Dictionary containing analysis metrics
        """
        # Calculate actual duty cycles used
        duty_cycles = []
        for i in range(0, len(pwm_signal), self.samples_per_period):
            block = pwm_signal[i:i + self.samples_per_period]
            if len(block) < self.samples_per_period:
                break
            duty_cycles.append(stats.sum(block > 0) / self.samples_per_period)
            
        duty_cycles = tensor.convert_to_tensor(duty_cycles)
        
        return {
            'mean_duty_cycle': stats.mean(duty_cycles),
            'min_duty_cycle': stats.min(duty_cycles),
            'max_duty_cycle': stats.max(duty_cycles),
            'unique_levels': len(np.unique(duty_cycles)),
            'theoretical_levels': self.levels
        }
        
    def get_pwm_parameters(self) -> dict:
        """Get current PWM processing parameters."""
        return {
            'bits_per_block': self.bits,
            'carrier_frequency': self.carrier_freq,
            'sample_rate': self.sample_rate,
            'samples_per_period': self.samples_per_period,
            'quantization_levels': self.levels
        }