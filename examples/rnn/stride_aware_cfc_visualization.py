# examples/rnn/stride_aware_cfc_visualization.py

"""
Visualization Example for Stride-Aware CfC Temporal Dynamics.

This script demonstrates how different stride lengths affect the temporal
dynamics in StrideAwareWiredCfCCell neurons using the visualization function
originally part of the stride_aware_cfc module.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import importlib # Added importlib for potential dynamic imports if needed

# Corrected imports based on our refactoring
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.rnn.stride_aware_cfc import StrideAwareWiredCfCCell
from ember_ml.nn.modules.auto_ncp import AutoNCP # Assuming AutoNCP is the wiring used

# NOTE: This function is for visualization purposes only and is not used in the core functionality.
# It's acceptable to use NumPy here because:
# 1. The visualization function is not part of the core functionality
# 2. It's only used for debugging and demonstration purposes
# 3. The visualization libraries (matplotlib) require NumPy arrays
# The core functionality of the module uses the ops abstraction layer as required.
def visualize_stride_temporal_dynamics(time_steps=100, stride_lengths=[1, 3, 5],
                                        units=16, input_dim=3, seed=42):
    """
    Visualizes how different stride lengths affect the temporal dynamics in CfC neurons.

    This visualization creates a rigorous analysis of:
    1. State evolution trajectories across different temporal scales
    2. Information retention characteristics as function of stride length
    3. Comparative phase space analysis of multi-timescale representations

    Args:
        time_steps: Total number of time steps to simulate
        stride_lengths: List of stride lengths to compare
        units: Number of hidden units in each CfC cell
        input_dim: Input dimensionality
        seed: Random seed for reproducibility
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
    except ImportError:
        Axes3D = None  # If not available, we'll skip 3D plotting

    # Set seeds for reproducibility
    np.random.seed(seed)
    tensor.set_seed(seed)

    # Import the correct class
    from ember_ml.nn.modules.wiring import NCPMap
    
    # Create a NeuronMap for each stride cell
    # We need to create a neuron_map that works with the refactored architecture
    neuron_map = NCPMap(
        inter_neurons=units - units//4,  # Reserve some neurons for output
        motor_neurons=units//4,          # Output size
        sensory_neurons=0,               # No sensory neurons in our case
        input_dim=input_dim,             # Explicitly set input dimension
        sparsity_level=0.5,
        seed=seed
    )
    
    # Ensure the neuron map is built properly
    if not neuron_map.is_built():
        neuron_map.build(input_dim)
    
    # Verify the map was built correctly
    if not neuron_map.is_built():
        raise ValueError("NeuronMap failed to build properly")

    # Generate synthetic input sequence with temporal structure
    # Using sinusoidal patterns with varying frequencies to test multi-scale dynamics
    t = tensor.linspace(0, 4*ops.pi, time_steps)
    frequencies = [1.0, 2.0, 0.5]
    input_signals = []
    for freq in frequencies[:input_dim]:
        signal = ops.sin(freq * t) + 0.1 * np.random.randn(time_steps)
        input_signals.append(signal)
    input_sequence = tensor.stack(input_signals, axis=1).astype(tensor.float32)

    # Create cells for each stride length
    stride_cells = {}
    for stride in stride_lengths:
        # Use the properly implemented StrideAwareWiredCfCCell
        # Pass the neuron_map instance
        cell = StrideAwareWiredCfCCell(
            neuron_map=neuron_map,  # Pass the neuron_map we created
            stride_length=stride,
            time_scale_factor=1.0,
            mode="default"
            # ModuleWiredCell will handle input_size using neuron_map.input_dim
        )
        stride_cells[stride] = cell

    # Initialize states for each cell
    # Use the cell's get_initial_state method
    # This returns a list of tensors [h] for each cell
    states = {}
    for stride, cell in stride_cells.items():
        initial_state = cell.get_initial_state(batch_size=1)
        # Ensure we have a valid state - should be a list with at least one tensor
        if not isinstance(initial_state, list) or len(initial_state) == 0:
            raise ValueError(f"Cell for stride {stride} returned invalid initial state: {initial_state}")
        states[stride] = initial_state


    # Track state evolution for each stride
    state_evolution = {stride: tensor.zeros((time_steps, units)) for stride in stride_lengths}

    # Process sequence through each stride-specific cell
    for t_idx in range(time_steps):
        x_t = input_sequence[t_idx:t_idx+1] # Get current time step input (shape [1, input_dim])
        x_t = tensor.convert_to_tensor(x_t) # Convert slice to tensor

        for stride, cell in stride_cells.items():
            # Only process input at stride-specific intervals
            if t_idx % stride == 0:
                current_state = states[stride] # Should be a list [h]
                output, new_state = cell.forward(x_t, current_state, time=1.0)
                states[stride] = new_state

            # Record state at every time step for all cells
            # Ensure we're getting a numpy array with the right shape
            state_array = states[stride][0]  # Get the hidden state tensor h from the list
            if len(tensor.shape(state_array)) > 1: # Remove batch dim if present
                state_array = state_array[0]
            # Update the state_evolution tensor for the current time step
            # Create a new copy of the state evolution tensor
            updated_state = tensor.copy(state_evolution[stride])
            # Directly assign the state array to the specific time step
            # This avoids potential issues with tensor_scatter_nd_update implementation
            updated_state = tensor.slice_update(
                updated_state,
                [t_idx],  # Simple slice with time index
                tensor.expand_dims(state_array, axis=0)  # Ensure shape matches expected dimensions
            )
            state_evolution[stride] = updated_state


    # Convert tensors to NumPy for visualization
    state_evolution_np = {}
    for stride, states_tensor in state_evolution.items():
         state_evolution_np[stride] = tensor.to_numpy(states_tensor)


    # === CREATE MULTI-PANEL ANALYTICAL VISUALIZATION ===
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)

    # 1. Time series evolution of key state components
    ax1 = fig.add_subplot(gs[0, :])
    neurons_to_plot = min(3, units)  # Plot first few neurons

    for stride, states_np in state_evolution_np.items():
        for n in range(neurons_to_plot):
            ax1.plot(states_np[:, n], label=f"Stride {stride}, Neuron {n}")

    ax1.set_title("Temporal Evolution of Neuronal States Across Strides", fontsize=14)
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Activation", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. State space trajectories (3D phase plot)
    if units >= 3 and Axes3D is not None:
        ax2 = fig.add_subplot(gs[1, :2], projection='3d')

        for stride, states_np in state_evolution_np.items():
            ax2.plot(
                states_np[:, 0],
                states_np[:, 1],
                states_np[:, 2],
                label=f"Stride {stride}"
            )
            # Mark start and end points
            ax2.scatter([states_np[0, 0]], [states_np[0, 1]], [states_np[0, 2]],
                        color='green', marker='o', label="_start" if stride > stride_lengths[0] else "Start")
            ax2.scatter([states_np[-1, 0]], [states_np[-1, 1]], [states_np[-1, 2]],
                        color='red', marker='o', label="_end" if stride > stride_lengths[0] else "End")

        ax2.set_title("Phase Space Trajectory", fontsize=14)
        ax2.set_xlabel("State Dimension 1", fontsize=10)
        ax2.set_ylabel("State Dimension 2", fontsize=10)
        # set_zlabel is a valid method for 3D axes, but Pylance doesn't recognize it
        # This is fine because we're only using it when Axes3D is available
        if hasattr(ax2, 'set_zlabel'):
            ax2.set_zlabel("State Dimension 3", fontsize=10)  # type: ignore
        ax2.legend(loc="upper right", fontsize=10)

    # 3. Information retention analysis
    ax3 = fig.add_subplot(gs[1, 2:])

    # Calculate state change rates for each stride
    change_rates = {}
    for stride, states_np in state_evolution_np.items():
        # Compute L2 norm of state differences
        diffs = ops.linearalg.norm(states_np[1:] - states_np[:-1], axis=1)
        change_rates[stride] = diffs

    for stride, rates in change_rates.items():
        rate_smoothed = np.convolve(rates, tensor.ones(5)/5, mode='valid') # Smooth rates
        ax3.plot(rate_smoothed, label=f"Stride {stride}")

    ax3.set_title("State Change Magnitude Over Time (Smoothed)", fontsize=14)
    ax3.set_xlabel("Time Step", fontsize=12)
    ax3.set_ylabel("L2 Norm of State Δ", fontsize=12)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Input sensitivity analysis - how different strides respond to input features
    # input_idx = tensor.arange(0, time_steps, max(stride_lengths)) # Unused variable
    ax4 = fig.add_subplot(gs[2, :2])

    # Plot input signals
    for i in range(input_dim):
        ax4.plot(input_sequence[:, i], '--', alpha=0.5, label=f"Input {i}")

    # Overlay vertical lines at each stride's sampling points
    colors = plt.cm.viridis(tensor.linspace(0, 1, len(stride_lengths))) # Use a colormap
    for i, stride in enumerate(stride_lengths):
        for idx in range(0, time_steps, stride):
            ax4.axvline(x=idx, color=colors[i], linestyle=':', alpha=0.4)

    ax4.set_title("Input Signals with Stride Sampling Points", fontsize=14)
    ax4.set_xlabel("Time Step", fontsize=12)
    ax4.set_ylabel("Input Value", fontsize=12)
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Spectral analysis - frequency domain comparison
    ax5 = fig.add_subplot(gs[2, 2:])

    for stride, states_np in state_evolution_np.items():
        # Take FFT of first few neurons and average
        fft_magnitudes = []
        for n in range(min(5, units)):
            fft = ops.abs(linearalg.rfft(states_np[:, n]))
            fft_magnitudes.append(fft)

        avg_fft = stats.mean(tensor.convert_to_tensor(fft_magnitudes), axis=0)
        freqs = linearalg.rfftfreq(time_steps)

        ax5.plot(freqs, avg_fft, label=f"Stride {stride}")

    ax5.set_title("Frequency Domain Analysis (Avg Magnitude)", fontsize=14)
    ax5.set_xlabel("Frequency", fontsize=12)
    ax5.set_ylabel("Magnitude", fontsize=12)
    ax5.legend(loc="upper right", fontsize=10)
    ax5.set_xlim(0, 0.5)  # Only show meaningful frequency range
    ax5.grid(True, alpha=0.3)

    # Add title and adjust layout
    fig.suptitle(
        f"Multi-scale CfC Temporal Dynamics Analysis\n"
        f"Comparing Stride Lengths: {stride_lengths}",
        fontsize=16
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    return fig

if __name__ == "__main__":
    fig = visualize_stride_temporal_dynamics()
    plt.show()