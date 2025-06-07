"""
Wave pattern and signal generation components.
"""

import math
from typing import List, Optional, Dict, Tuple, Union
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules
from ember_ml.nn import container
from ember_ml.nn.modules import activations
from ember_ml.backend import get_backend
from .binary_wave import WaveConfig

class SignalSynthesizer:
    """Synthesizer for generating various waveforms."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize signal synthesizer.

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
            Tensor containing sine wave
        """
        t = tensor.linspace(0, duration, int(duration * self.sampling_rate))
        return amplitude * ops.sin(2 * math.pi * frequency * t + phase)
        
    def square_wave(self,
                   frequency: float,
                   duration: float,
                   amplitude: float = 1.0,
                   duty_cycle: float = 0.5) -> tensor.convert_to_tensor:
        """
        Generate square wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude
            duty_cycle: Duty cycle (0 to 1)

        Returns:
            Tensor containing square wave
        """
        t = tensor.linspace(0, duration, int(duration * self.sampling_rate))
        wave = ops.sin(2 * math.pi * frequency * t)
        return amplitude * ops.sign(wave - math.cos(math.pi * duty_cycle))
        
    def sawtooth_wave(self,
                     frequency: float,
                     duration: float,
                     amplitude: float = 1.0) -> tensor.convert_to_tensor:
        """
        Generate sawtooth wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude

        Returns:
            Tensor containing sawtooth wave
        """
        t = tensor.linspace(0, duration, int(duration * self.sampling_rate))
        wave = t * frequency - ops.floor(t * frequency)
        return 2 * amplitude * (wave - 0.5)
        
    def triangle_wave(self,
                     frequency: float,
                     duration: float,
                     amplitude: float = 1.0) -> tensor.convert_to_tensor:
        """
        Generate triangle wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Signal duration in seconds
            amplitude: Wave amplitude

        Returns:
            Tensor containing triangle wave
        """
        t = tensor.linspace(0, duration, int(duration * self.sampling_rate))
        wave = t * frequency - ops.floor(t * frequency)
        return 2 * amplitude * ops.abs(2 * wave - 1) - amplitude
        
    def noise(self,
             duration: float,
             amplitude: float = 1.0,
             distribution: str = 'uniform') -> tensor.convert_to_tensor:
        """
        Generate noise signal.

        Args:
            duration: Signal duration in seconds
            amplitude: Noise amplitude
            distribution: Noise distribution ('uniform' or 'gaussian')

        Returns:
            Tensor containing noise signal
        """
        n_samples = int(duration * self.sampling_rate)
        if distribution == 'uniform':
            return 2 * amplitude * (tensor.random_uniform((n_samples,)) - 0.5)
        elif distribution == 'gaussian':
            return amplitude * tensor.random_normal((n_samples,))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

class PatternGenerator:
    """Generator for 2D wave patterns."""
    
    def __init__(self, config: WaveConfig):
        """
        Initialize pattern generator.

        Args:
            config: Wave configuration
        """
        self.config = config
        
    def binary_pattern(self, density: float) -> tensor.convert_to_tensor:
        """
        Generate binary pattern.

        Args:
            density: Target pattern density (0 to 1)

        Returns:
            Binary pattern tensor
        """
        if isinstance(self.config.grid_size, tuple):
            grid_shape = self.config.grid_size
        else:
            grid_shape = (self.config.grid_size, self.config.grid_size)
            
        pattern = tensor.random_uniform(shape=grid_shape, minval=0.0, maxval=1.0)
        return tensor.cast(pattern < density, tensor.float32)
        
    def wave_pattern(self,
                    frequency: float,
                    duration: float) -> tensor.convert_to_tensor:
        """
        Generate wave-based pattern.

        Args:
            frequency: Pattern frequency
            duration: Time duration for pattern

        Returns:
            Wave pattern tensor
        """
        x = tensor.linspace(0, duration, self.config.grid_size)
        y = tensor.linspace(0, duration, self.config.grid_size)
        
        # Use meshgrid without the indexing parameter for MLX compatibility
        # and handle the transpose if needed based on backend behavior
        X, Y = tensor.meshgrid(x, y)
        
        # Check if we need to transpose based on backend behavior
        backend_name = get_backend().__class__.__name__
        if backend_name != "MLXBackend":
            # For PyTorch and NumPy, we might need to transpose to match 'ij' indexing
            # This is backend-specific behavior that should be handled in the backend implementation
            pass
        
        pattern = ops.sin(2 * math.pi * frequency * X) * \
                 ops.sin(2 * math.pi * frequency * Y)
        return 0.5 * (pattern + 1)  # Normalize to [0, 1]
        
    def interference_pattern(self,
                           frequencies: List[float],
                           amplitudes: List[float],
                           duration: float) -> tensor.convert_to_tensor:
        """
        Generate interference pattern from multiple waves.

        Args:
            frequencies: List of wave frequencies
            amplitudes: List of wave amplitudes
            duration: Time duration for pattern

        Returns:
            Interference pattern tensor
        """
        pattern = tensor.zeros((self.config.grid_size, self.config.grid_size))
        for freq, amp in zip(frequencies, amplitudes):
            pattern += amp * self.wave_pattern(freq, duration)
        return ops.clip(pattern, 0, 1)

class WaveGenerator(modules.Module):
    """Neural network-based wave pattern generator."""
    
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 config: WaveConfig):
        """
        Initialize wave generator.

        Args:
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            config: Wave configuration
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Generator network
        self.net = container.Sequential([
            container.Linear(latent_dim, hidden_dim),
            activations.ReLU(),
            container.Linear(hidden_dim, hidden_dim),
            activations.ReLU(),
            container.Linear(hidden_dim, config.grid_size[0] * config.grid_size[1] if isinstance(config.grid_size, tuple) else config.grid_size * config.grid_size),
            activations.Sigmoid()
        ])
        
        # Phase network
        self.phase_net = container.Sequential([
            container.Linear(latent_dim, hidden_dim),
            activations.ReLU(),
            container.Linear(hidden_dim, config.num_phases if hasattr(config, 'num_phases') else 8),
            activations.Sigmoid()
        ])
        
    def forward(self,
                z: tensor.convert_to_tensor,
                return_phases: bool = False) -> Union[tensor.convert_to_tensor, Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]]:
        """
        Generate wave pattern.

        Args:
            z: Latent vector [batch_size, latent_dim]
            return_phases: Whether to return phase information

        Returns:
            Generated pattern tensor [batch_size, grid_size, grid_size]
            and optionally phases [batch_size, num_phases]
        """
        # Generate pattern
        pattern = self.net(z)
        grid_size = self.config.grid_size
        if isinstance(grid_size, tuple):
            pattern = tensor.reshape(pattern, (tensor.shape(z)[0], grid_size[0], grid_size[1]))
        else:
            pattern = tensor.reshape(pattern, (tensor.shape(z)[0], grid_size, grid_size))
        
        if return_phases:
            phases = self.phase_net(z)
            return pattern, phases
        return pattern
        
    def interpolate(self,
                   z1: tensor.convert_to_tensor,
                   z2: tensor.convert_to_tensor,
                   steps: int) -> tensor.convert_to_tensor:
        """
        Interpolate between two latent vectors.

        Args:
            z1: First latent vector [latent_dim]
            z2: Second latent vector [latent_dim]
            steps: Number of interpolation steps

        Returns:
            Tensor of interpolated patterns [steps, grid_size, grid_size]
        """
        alphas = tensor.linspace(0, 1, steps)
        patterns = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            z = tensor.expand_dims(z, 0)  # Add batch dimension
            pattern = self(z)
            patterns.append(tensor.squeeze(pattern, 0))  # Remove batch dimension
            
        return tensor.stack(patterns)
        
    def random_sample(self,
                     num_samples: int,
                     seed: Optional[int] = None) -> tensor.convert_to_tensor:
        """
        Generate random patterns.

        Args:
            num_samples: Number of patterns to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of generated patterns [num_samples, grid_size, grid_size]
        """
        if seed is not None:
            tensor.set_seed(seed)
            
        z = tensor.random_normal((num_samples, self.latent_dim))
        return self(z)