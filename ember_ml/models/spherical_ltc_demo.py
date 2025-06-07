"""Demonstration of spherical (non-Euclidean) LTC neurons."""

import torch
import matplotlib.pyplot as plt
import numpy as np
# Updated import path for Spherical LTC components
from ember_ml.nn.modules.rnn.spherical_ltc import (
    SphericalLTCConfig,
    SphericalLTCChain
)

from ember_ml.nn import tensor
from ember_ml import ops

def generate_input_signal(
    total_time: float = 10.0,
    pattern_time: float = 5.0,
    dt: float = 0.01,
    freq: float = 1.0
) -> tensor.convert_to_tensor:
    """Generate sinusoidal input signal.
    
    Args:
        total_time: Total simulation time
        pattern_time: Duration of input pattern
        dt: Time step
        freq: Signal frequency
        
    Returns:
        Input signal tensor (num_steps, 3)
    """
    num_steps = int(total_time / dt)
    pattern_steps = int(pattern_time / dt)
    
    # Create time array
    t = torch.linspace(0, total_time, num_steps)
    
    # Generate signal
    signal = tensor.zeros(num_steps)
    signal[:pattern_steps] = torch.sin(2 * ops.pi * freq * t[:pattern_steps])
    
    # Convert to 3D
    input_3d = tensor.zeros(num_steps, 3)
    input_3d[:, 0] = signal  # Project onto x-axis
    
    return input_3d

def run_simulation(
    chain: SphericalLTCChain,
    input_signal: tensor.convert_to_tensor,
    batch_size: int = 1
) -> tensor.convert_to_tensor:
    """Run simulation of spherical LTC chain.
    
    Args:
        chain: Spherical LTC chain
        input_signal: Input signal (num_steps, 3)
        batch_size: Batch size for parallel simulation
        
    Returns:
        State history tensor (batch_size, num_steps, num_neurons, 3)
    """
    num_steps = input_signal.size(0)
    states_history = []
    
    # Reset chain
    chain.reset_states(batch_size)
    
    # Run simulation
    for step in range(num_steps):
        input_batch = input_signal[step:step+1].expand(batch_size, -1)
        states, _ = chain(input_batch)
        states_history.append(states)
        
    return torch.stack(states_history, dim=1)

def plot_results(
    states: tensor.convert_to_tensor,
    input_signal: tensor.convert_to_tensor,
    pattern_time: float,
    dt: float
):
    """Plot simulation results.
    
    Args:
        states: State history (batch_size, time, num_neurons, 3)
        input_signal: Input signal (time, 3)
        pattern_time: Duration of input pattern
        dt: Time step
    """
    num_steps = states.size(1)
    num_neurons = states.size(2)
    time = tensor.arange(num_steps) * dt
    
    # Plot norms (should stay close to 1)
    plt.figure(figsize=(12, 6))
    norms = torch.norm(states[0], dim=-1)
    
    for i in range(num_neurons):
        plt.plot(time, norms[:, i], label=f'LTC {i+1} Norm')
        
    plt.axvline(pattern_time, color='r', linestyle='--', label='Pattern End')
    plt.title('Spherical LTC Chain - State Vector Norms')
    plt.xlabel('Time (s)')
    plt.ylabel('||x||')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot x-axis projections
    plt.figure(figsize=(12, 6))
    
    # Input signal
    plt.plot(time, input_signal[:, 0], label='Input', linewidth=2)
    
    # Neuron states
    for i in range(num_neurons):
        plt.plot(
            time,
            states[0, :, i, 0],
            label=f'LTC {i+1}',
            alpha=0.8
        )
        
    plt.axvline(pattern_time, color='r', linestyle='--', label='Pattern End')
    plt.title('Spherical LTC Chain - X-axis Projections')
    plt.xlabel('Time (s)')
    plt.ylabel('x(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    # Parameters
    total_time = 10.0
    pattern_time = 5.0
    dt = 0.01
    num_neurons = 3
    
    # Create config
    config = SphericalLTCConfig(
        tau=1.0,
        gleak=0.5,
        dt=dt
    )
    
    # Create chain
    chain = SphericalLTCChain(num_neurons, config)
    
    # Generate input
    input_signal = generate_input_signal(
        total_time=total_time,
        pattern_time=pattern_time,
        dt=dt
    )
    
    # Run simulation
    states = run_simulation(chain, input_signal)
    
    # Plot results
    plot_results(states, input_signal, pattern_time, dt)
    
    # Compute forgetting times
    forgetting_times = chain.get_forgetting_times(
        states,
        threshold=0.05
    )
    
    # Print results
    print("\nForgetting Analysis:")
    for i, time in forgetting_times.items():
        if time is not None:
            print(f"LTC {i+1} forgot pattern after {time:.2f}s")
        else:
            print(f"LTC {i+1} maintained pattern")
            
if __name__ == "__main__":
    main()