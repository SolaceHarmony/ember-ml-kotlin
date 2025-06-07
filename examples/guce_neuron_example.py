"""
GUCE Neuron Example

This example demonstrates how to use the Grand Unified Cognitive Equation (GUCE) neuron,
which integrates several advanced concepts:
- PCM waveform encoding
- b-Symplectic gradient flow
- Holographic error correction
- Theta-gamma oscillatory gating
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.nn.modules import GUCE
from ember_ml.nn import tensor
from ember_ml import ops
ops.set_backend("torch")  # Set the backend to PyTorch
def main():
    """Run the GUCE neuron example."""
    print("GUCE Neuron Example")
    print("==================")
    
    # Generate test signal: 440 Hz sine wave (1 second)
    sampling_rate = 44100
    duration = 1.0  # seconds
    t = tensor.linspace(0, duration, int(duration * sampling_rate))
    frequency = 440.0  # Hz (A4 note)
    signal = ops.sin(ops.multiply(ops.multiply(ops.multiply(2.0, ops.pi), frequency), t))
    
    # Create GUCE neuron
    state_dim = 64
    guce = GUCE(
        state_dim=state_dim,
        step_size=0.01,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=1.0/sampling_rate
    )
    
    # Process signal through GUCE neuron
    print("Processing signal through GUCE neuron...")
    states = []
    energy_history = []
    
    # Take a subset of samples for demonstration
    num_samples = 1000
    samples = tensor.slice(signal, 0, num_samples)
    
    for i in range(num_samples):
        # Get current sample
        sample = samples[i]
        
        # Create input vector (replicate sample across state dimensions)
        inputs = tensor.full((state_dim,), sample)
        
        # Process through GUCE neuron
        state, energy = guce(inputs)
        
        # Store results
        states.append(state)
        energy_history.append(energy)
    
    # Convert lists to tensors
    states_tensor = tensor.stack(states)
    energy_tensor = tensor.stack(energy_history)
    
    # Reconstruct signal from neuron states
    # Ensure reconstructed is at least 1-dimensional
    reconstructed = ops.stats.mean(states_tensor, axis=1)
    if len(tensor.shape(reconstructed)) == 0:
        reconstructed = tensor.reshape(reconstructed, (1,))
    
    # Compute reconstruction error
    mse = ops.stats.mean(ops.square(ops.subtract(samples, reconstructed)))
    print(f"Reconstruction MSE: {tensor.item(mse):.4f}")
    
    # Visualize results
    visualize_results(samples, reconstructed, energy_tensor, states_tensor)

def visualize_results(original, reconstructed, energy, states):
    """Visualize the GUCE neuron results."""
    # Convert tensors to numpy for plotting
    original_np = tensor.to_numpy(original)
    reconstructed_np = tensor.to_numpy(reconstructed)
    # Ensure reconstructed_np is at least 1-dimensional
    if reconstructed_np.ndim == 0:
        reconstructed_np = reconstructed_tensor.reshape(1)
    energy_np = tensor.to_numpy(energy)
    
    # Plot original vs reconstructed signal
    plt.figure(figsize=(12, 8))
    
    # Original vs reconstructed
    plt.subplot(2, 1, 1)
    plt.title("Original vs Reconstructed Signal")
    plt.plot(original_np[:200], label="Original")
    plt.plot(reconstructed_np[:200], label="Reconstructed", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Energy history
    plt.subplot(2, 1, 2)
    plt.title("GUCE Neuron Energy")
    plt.plot(energy_np[:200])
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("guce_results.png")
    print("Saved visualization to 'guce_results.png'")
    
    # Plot state evolution
    plt.figure(figsize=(12, 6))
    plt.title("GUCE Neuron State Evolution")
    
    # Get a subset of states for visualization
    state_subset = tensor.to_numpy(states[100])
    plt.plot(state_subset)
    plt.xlabel("State Dimension")
    plt.ylabel("Activation")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("guce_state.png")
    print("Saved state visualization to 'guce_state.png'")

def demonstrate_advanced_features():
    """Demonstrate advanced features of the GUCE neuron."""
    print("\nDemonstrating Advanced Features")
    print("==============================")
    
    # Create GUCE neuron with different configurations
    state_dim = 128
    guce = GUCE(
        state_dim=state_dim,
        step_size=0.005,
        nu_0=2.0,
        beta=0.2,
        theta_freq=6.0,
        gamma_freq=60.0,
        dt=1.0/44100
    )
    
    # Generate complex input: combination of two frequencies
    sampling_rate = 44100
    duration = 2.0  # seconds
    t = tensor.linspace(0, duration, int(duration * sampling_rate))
    
    # Create a signal that changes over time
    f1 = 440.0  # Hz (A4 note)
    f2 = 880.0  # Hz (A5 note)
    
    # First half: f1 only
    half_samples = int(duration * sampling_rate) // 2
    signal1 = ops.sin(ops.multiply(ops.multiply(2.0 * ops.pi, f1), t[:half_samples]))
    
    # Second half: f1 + f2
    signal2 = ops.add(
        ops.sin(ops.multiply(ops.multiply(2.0 * ops.pi, f1), t[half_samples:])),
        ops.multiply(0.5, ops.sin(ops.multiply(ops.multiply(2.0 * ops.pi, f2), t[half_samples:])))
    )
    
    # Combine signals
    signal = tensor.concatenate([signal1, signal2])
    
    # Process signal through GUCE neuron
    print("Processing complex signal...")
    states = []
    energy_history = []
    
    # Take a subset of samples for demonstration
    num_samples = 2000
    samples = tensor.slice(signal, 0, num_samples)
    
    for i in range(num_samples):
        # Get current sample
        sample = samples[i]
        
        # Create input vector (replicate sample across state dimensions)
        inputs = tensor.full((state_dim,), sample)
        
        # Process through GUCE neuron
        state, energy = guce(inputs)
        
        # Store results
        states.append(state)
        energy_history.append(energy)
    
    # Convert lists to tensors
    states_tensor = tensor.stack(states)
    energy_tensor = tensor.stack(energy_history)
    
    # Visualize frequency detection
    visualize_frequency_detection(samples, energy_tensor, half_samples)

def visualize_frequency_detection(signal, energy, transition_point):
    """Visualize how GUCE neuron responds to frequency changes."""
    # Convert tensors to numpy for plotting
    signal_np = tensor.to_numpy(signal)
    # Ensure signal_np is at least 1-dimensional
    if signal_np.ndim == 0:
        signal_np = signal_tensor.reshape(1)
    energy_np = tensor.to_numpy(energy)
    # Ensure energy_np is at least 1-dimensional
    if energy_np.ndim == 0:
        energy_np = energy_tensor.reshape(1)
    
    plt.figure(figsize=(12, 8))
    
    # Original signal
    plt.subplot(2, 1, 1)
    plt.title("Input Signal with Frequency Change")
    plt.plot(signal_np)
    plt.axvline(x=transition_point, color='r', linestyle='--', label="Frequency Change")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Energy response
    plt.subplot(2, 1, 2)
    plt.title("GUCE Neuron Energy Response")
    plt.plot(energy_np)
    plt.axvline(x=transition_point, color='r', linestyle='--', label="Frequency Change")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("guce_frequency_detection.png")
    print("Saved frequency detection visualization to 'guce_frequency_detection.png'")

if __name__ == "__main__":
    main()
    demonstrate_advanced_features()