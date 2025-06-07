# Wave Generation

This section describes components responsible for generating various waveforms and 2D wave patterns.

## Core Concepts

These components provide tools to synthesize signals (like sine, square, sawtooth waves) and generate 2D patterns based on wave interference or learned latent representations.

## Components

### `ember_ml.wave.generator`

*   **`SignalSynthesizer`**: Synthesizes various 1D waveforms.
    *   `__init__(sampling_rate)`: Initializes with a sampling rate.
    *   `sine_wave(frequency, duration, amplitude, phase)`: Generates a sine wave.
    *   `square_wave(frequency, duration, amplitude, duty_cycle)`: Generates a square wave.
    *   `sawtooth_wave(frequency, duration, amplitude)`: Generates a sawtooth wave.
    *   `triangle_wave(frequency, duration, amplitude)`: Generates a triangle wave.
    *   `noise(duration, amplitude, distribution)`: Generates uniform or Gaussian noise.
*   **`PatternGenerator`**: Generates 2D wave patterns.
    *   `__init__(config)`: Initializes with a `WaveConfig`.
    *   `binary_pattern(density)`: Generates a random binary pattern with a target density.
    *   `wave_pattern(frequency, duration)`: Generates a 2D pattern based on interfering sine waves.
    *   `interference_pattern(frequencies, amplitudes, duration)`: Generates a pattern from the interference of multiple waves.
*   **`WaveGenerator(nn.Module)`**: A neural network (using linear layers and ReLU/Sigmoid) that generates 2D wave patterns from a latent vector `z`. It can also generate associated phase information.
    *   `__init__(latent_dim, hidden_dim, config)`: Initializes the generator network and phase network.
    *   `forward(z, return_phases)`: Generates a pattern (and optionally phases) from a latent vector `z`.
    *   `interpolate(z1, z2, steps)`: Generates patterns by interpolating between two latent vectors.
    *   `random_sample(num_samples, seed)`: Generates random patterns by sampling from the latent space.