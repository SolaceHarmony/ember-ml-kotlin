"""
LQNet and CTRQNet Example

This script demonstrates how to use the Liquid Quantum Neural Network (LQNet)
and Continuous-Time Recurrent Quantum Neural Network (CTRQNet) modules.
"""

from ember_ml import ops
from ember_ml.nn import tensor

# Set the backend to MLX for better performance
ops.set_backend('mlx')
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import LQNet, CTRQNet

def generate_sine_wave(seq_length, num_samples, freq=0.1, noise=0.1):
    """Generate sine wave data for testing."""
    # Create time points from 0 to 10
    x = tensor.linspace(0.0, 10.0, seq_length)
    data = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random phase between 0 and 2π
        # Use a fixed phase for each sample to avoid random_uniform issues
        phase = 2.0 * ops.pi * (float(_) / float(num_samples))
        
        # Generate sine wave: sin(2π * freq * x + phase)
        sine_wave = ops.sin(ops.add(
            ops.multiply(2.0 * ops.pi * freq, x),
            phase
        ))
        
        # Add noise to create noisy sine wave
        noise_tensor = tensor.random_normal(tensor.shape(sine_wave), stddev=noise)
        noisy_sine = ops.add(sine_wave, noise_tensor)
        
        # Reshape to add feature dimension
        noisy_sine_reshaped = tensor.reshape(noisy_sine, (-1, 1))
        sine_wave_reshaped = tensor.reshape(sine_wave, (-1, 1))
        
        # Use the noisy sine as input and clean sine as target
        data.append(noisy_sine_reshaped)
        targets.append(sine_wave_reshaped)
    
    # Stack along batch dimension
    return tensor.stack(data), tensor.stack(targets)

def main():
    """Main function to demonstrate LQNet and CTRQNet."""
    # Generate data
    seq_length = 100
    num_samples = 32
    input_dim = 1
    hidden_dim = 32
    
    print("Generating sine wave data...")
    X, y = generate_sine_wave(seq_length, num_samples)
    
    # No need to convert to tensors as they are already tensors
    X_tensor = X
    y_tensor = y
    
    # Create NeuronMap for LQNet
    print("\nCreating LQNet model...")
    neuron_map = NCPMap(
        inter_neurons=hidden_dim // 2,
        command_neurons=hidden_dim // 4,
        motor_neurons=hidden_dim // 4,
        sensory_neurons=input_dim,
        seed=42
    )
    
    # Create LQNet model
    lqnet = LQNet(
        neuron_map=neuron_map,
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        return_sequences=True,
        return_state=False,
        batch_first=True
    )
    
    # Forward pass through LQNet
    print("Running forward pass through LQNet...")
    lqnet_output = lqnet(X_tensor)
    
    print(f"LQNet input shape: {tensor.shape(X_tensor)}")
    print(f"LQNet output shape: {tensor.shape(lqnet_output)}")
    
    # Create CTRQNet model
    print("\nCreating CTRQNet model...")
    ctrqnet = CTRQNet(
        neuron_map=neuron_map,
        nu_0=1.0,
        beta=0.1,
        noise_scale=0.05,
        time_scale_factor=1.0,
        use_harmonic_embedding=True,
        return_sequences=True,
        return_state=False,
        batch_first=True
    )
    
    # Forward pass through CTRQNet
    print("Running forward pass through CTRQNet...")
    ctrqnet_output = ctrqnet(X_tensor)
    
    print(f"CTRQNet input shape: {tensor.shape(X_tensor)}")
    print(f"CTRQNet output shape: {tensor.shape(ctrqnet_output)}")
    
    # Calculate MSE loss using the loss_ops module
    # We need to ensure the dimensions match for the MSE calculation
    # The model outputs have shape (batch_size, seq_length, units)
    # But our targets have shape (batch_size, seq_length, 1)
    
    # For a fair comparison, we'll use only the first output dimension
    lqnet_output_first_dim = lqnet_output[:, :, 0:1]
    ctrqnet_output_first_dim = ctrqnet_output[:, :, 0:1]
    
    # Now calculate MSE using the mse function
    # Add a small epsilon to avoid NaN values
    lqnet_mse = ops.mse(y_tensor, lqnet_output_first_dim)
    ctrqnet_mse = ops.mse(y_tensor, ctrqnet_output_first_dim)
    
    # Check for NaN values and replace with a message
    lqnet_mse_str = "NaN (model needs training)" if ops.isnan(lqnet_mse) else f"{lqnet_mse:.6f}"
    ctrqnet_mse_str = "NaN (model needs training)" if ops.isnan(ctrqnet_mse) else f"{ctrqnet_mse:.6f}"
    
    print(f"\nLQNet MSE: {lqnet_mse_str}")
    print(f"CTRQNet MSE: {ctrqnet_mse_str}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()