"""
GUCE with AutoNCP for Control Theory

This example demonstrates how to integrate the Grand Unified Cognitive Equation (GUCE) neuron
with AutoNCP wiring for control theory applications.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.nn.modules.rnn import GUCE
from ember_ml.nn.modules import AutoNCP
from ember_ml.nn import tensor
from ember_ml import ops

class GUCEControlSystem:
    """
    Control system using GUCE neurons with AutoNCP wiring.
    
    This system integrates the b-symplectic dynamics of GUCE neurons
    with the structured connectivity of AutoNCP for control applications.
    """
    
    def __init__(
        self,
        state_dim=64,
        input_dim=3,
        output_dim=2,
        hidden_units=32,
        sparsity_level=0.5,
        step_size=0.01,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    ):
        """
        Initialize the GUCE control system.
        
        Args:
            state_dim: Dimension of the GUCE neuron state
            input_dim: Number of input dimensions (sensors)
            output_dim: Number of output dimensions (actuators)
            hidden_units: Number of hidden units in the AutoNCP wiring
            sparsity_level: Sparsity level of the AutoNCP wiring
            step_size: Learning rate for GUCE neurons
            nu_0: Base viscosity for GUCE neurons
            beta: Energy scaling for GUCE neurons
            theta_freq: Theta oscillation frequency (Hz)
            gamma_freq: Gamma oscillation frequency (Hz)
            dt: Time step size
        """
        # Create AutoNCP wiring
        self.wiring = AutoNCP(
            units=hidden_units,
            output_size=output_dim,
            sparsity_level=sparsity_level
        )
        
        # Create GUCE neurons (one per hidden unit)
        self.neurons = [
            GUCE(
                state_dim=state_dim,
                step_size=step_size,
                nu_0=nu_0,
                beta=beta,
                theta_freq=theta_freq,
                gamma_freq=gamma_freq,
                dt=dt
            )
            for _ in range(hidden_units)
        ]
        
        # Store dimensions
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        
        # Initialize weights
        self.input_weights = tensor.random_normal((input_dim, hidden_units))
        
        # Get output weights from wiring
        self.output_weights = self.wiring.adjacency_matrix
        
        # Initialize hidden states
        self.hidden_states = tensor.zeros((hidden_units,))
        self.neuron_states = [tensor.zeros((state_dim,)) for _ in range(hidden_units)]
        
        # For tracking
        self.energy_history = []
    
    def forward(self, inputs):
        """
        Forward pass through the control system.
        
        Args:
            inputs: Input tensor of shape (input_dim,)
            
        Returns:
            Output tensor of shape (output_dim,)
        """
        # Project inputs to hidden units
        hidden_inputs = ops.matmul(inputs, self.input_weights)
        
        # Update GUCE neurons
        new_hidden_states = tensor.zeros((self.hidden_units,))
        total_energy = 0.0
        
        for i in range(self.hidden_units):
            # Scale input to neuron
            neuron_input = tensor.full((self.state_dim,), hidden_inputs[i])
            
            # Update neuron
            state, energy = self.neurons[i](neuron_input)
            
            # Store state
            self.neuron_states[i] = state
            
            # Compute hidden state (mean of neuron state)
            new_hidden_states = tensor.with_value(
                new_hidden_states, i, tensor.mean(state)
            )
            
            # Accumulate energy
            total_energy += energy
        
        # Update hidden states
        self.hidden_states = new_hidden_states
        
        # Track energy
        self.energy_history.append(total_energy / self.hidden_units)
        
        # Compute outputs using wiring
        outputs = ops.matmul(self.hidden_states, self.output_weights)
        
        return outputs
    
    def reset(self):
        """Reset the control system state."""
        # Reset hidden states
        self.hidden_states = tensor.zeros((self.hidden_units,))
        
        # Reset neuron states
        self.neuron_states = [tensor.zeros((self.state_dim,)) for _ in range(self.hidden_units)]
        
        # Reset energy history
        self.energy_history = []
        
        # Reset GUCE neurons
        for neuron in self.neurons:
            neuron.reset()


def pendulum_dynamics(state, action, dt=0.01, g=9.8, m=1.0, l=1.0, b=0.1):
    """
    Simulate pendulum dynamics.
    
    Args:
        state: Current state [theta, theta_dot]
        action: Control action (torque)
        dt: Time step
        g: Gravity constant
        m: Mass
        l: Length
        b: Damping coefficient
        
    Returns:
        New state [theta, theta_dot]
    """
    # Extract state
    theta, theta_dot = state
    
    # Compute acceleration
    # theta_ddot = (action - b * theta_dot - m * g * l * ops.sin(theta)) / (m * l**2)
    term1 = ops.subtract(action, ops.multiply(b, theta_dot))
    term2 = ops.multiply(ops.multiply(ops.multiply(m, g), l), ops.sin(theta))
    numerator = ops.subtract(term1, term2)
    denominator = ops.multiply(m, ops.power(l, 2))
    theta_ddot = ops.divide(numerator, denominator)
    
    # Update state
    new_theta = ops.add(theta, ops.multiply(theta_dot, dt)) # Use ops.add, ops.multiply
    new_theta_dot = ops.add(theta_dot, ops.multiply(theta_ddot, dt)) # Use ops.add, ops.multiply
    
    # Normalize angle to [-pi, pi] using ops functions
    two_pi = ops.multiply(2.0, ops.pi)
    normalized_angle = ops.add(ops.divide(new_theta, two_pi), 0.5)
    floor_val = ops.floor(normalized_angle)
    new_theta = ops.subtract(new_theta, ops.multiply(two_pi, floor_val))
    
    return tensor.stack([new_theta, new_theta_dot])


def main():
    """Run the GUCE with AutoNCP control example."""
    print("GUCE with AutoNCP for Control Theory")
    print("===================================")
    
    # Create control system
    control_system = GUCEControlSystem(
        state_dim=32,
        input_dim=3,  # [theta, theta_dot, target]
        output_dim=1,  # [torque]
        hidden_units=16,
        sparsity_level=0.7,
        step_size=0.005,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    )
    
    # Simulation parameters
    dt = 0.01
    sim_time = 10.0  # seconds
    steps = int(sim_time / dt)
    
    # Initial state: pendulum at bottom position
    state = tensor.convert_to_tensor([ops.pi, 0.0])
    
    # Target state: pendulum at top position
    target = tensor.convert_to_tensor([0.0, 0.0])
    
    # Storage for results
    states = [state]
    actions = []
    
    # Run simulation
    print("Running pendulum control simulation...")
    for i in range(steps):
        # Prepare input for control system
        inputs = tensor.concat([state, target[0:1]])
        
        # Get control action
        action = control_system.forward(inputs)
        
        # Apply control action to pendulum
        state = pendulum_dynamics(state, action[0], dt)
        
        # Store results
        states.append(state)
        actions.append(action[0])
        
        # Print progress
        if i % 100 == 0:
            print(f"Step {i}/{steps}, Angle: {tensor.item(state[0]):.2f}, Action: {tensor.item(action[0]):.2f}")
    
    # Convert results to tensors
    states_tensor = tensor.stack(states)
    actions_tensor = tensor.stack(actions)
    
    # Visualize results
    visualize_results(states_tensor, actions_tensor, control_system.energy_history, dt)


def visualize_results(states, actions, energy_history, dt):
    """Visualize the control results."""
    # Convert tensors to numpy for plotting
    states_np = tensor.to_numpy(states)
    actions_np = tensor.to_numpy(actions)
    energy_np = tensor.to_numpy(tensor.stack(energy_history))
    
    # Create time vector
    time = tensor.to_numpy(tensor.linspace(0, dt * len(states_np), len(states_np)))
    action_time = tensor.to_numpy(tensor.linspace(0, dt * len(actions_np), len(actions_np)))
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot pendulum angle
    plt.subplot(3, 1, 1)
    plt.title("Pendulum Angle")
    plt.plot(time, states_np[:, 0])
    plt.axhline(y=0.0, color='r', linestyle='--', label="Target")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    
    # Plot control actions
    plt.subplot(3, 1, 2)
    plt.title("Control Actions")
    plt.plot(action_time, actions_np)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque")
    plt.grid(True)
    
    # Plot energy history
    plt.subplot(3, 1, 3)
    plt.title("GUCE Neurons Energy")
    plt.plot(action_time, energy_np)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("guce_control_results.png")
    print("Saved control results to 'guce_control_results.png'")
    
    # Visualize pendulum animation
    animate_pendulum(states_np, action_time)


def animate_pendulum(states, time):
    """Create an animation of the pendulum."""
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.title("Pendulum Animation")
    
    # Plot pendulum at key frames
    num_frames = 10
    frame_indices = [int(i * len(states) / num_frames) for i in range(num_frames)]
    
    for i, idx in enumerate(frame_indices):
        # Get pendulum angle
        theta = states[idx, 0]
        
        # Compute pendulum position
        x = tensor.to_numpy(ops.sin(theta))
        y = -tensor.to_numpy(ops.cos(theta))
        
        # Plot pendulum
        plt.plot([0, x], [0, y], 'k-', linewidth=2)
        plt.plot(x, y, 'bo', markersize=10, alpha=(i+1)/num_frames)
        
        # Add time label
        plt.text(x, y, f"t={time[idx]:.1f}s", fontsize=8)
    
    # Set axis limits
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.axis('equal')
    
    plt.savefig("pendulum_animation.png")
    print("Saved pendulum animation to 'pendulum_animation.png'")


def learning_example():
    """Demonstrate learning with GUCE and AutoNCP."""
    print("\nLearning with GUCE and AutoNCP")
    print("=============================")
    
    # Create control system
    control_system = GUCEControlSystem(
        state_dim=32,
        input_dim=2,  # [x, y]
        output_dim=2,  # [dx, dy]
        hidden_units=16,
        sparsity_level=0.5,
        step_size=0.01,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    )
    
    # Generate training data: circular trajectory
    num_samples = 1000
    t = tensor.linspace(0, 2 * ops.pi, num_samples)
    
    # Input: position on circle
    X = tensor.stack([ops.cos(t), ops.sin(t)], axis=1)
    
    # Output: velocity tangent to circle
    Y = tensor.stack([-ops.sin(t), ops.cos(t)], axis=1)
    
    # Training parameters
    epochs = 50
    batch_size = 32
    
    # Storage for results
    losses = []
    
    # Train the system
    print("Training the system...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        indices = tensor.random_permutation(num_samples)
        X_shuffled = tensor.take(X, indices, axis=0)
        Y_shuffled = tensor.take(Y, indices, axis=0)
        
        # Train in batches
        for i in range(0, num_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, num_samples)
            X_batch = tensor.slice(X_shuffled, i, end_idx - i, axis=0)
            Y_batch = tensor.slice(Y_shuffled, i, end_idx - i, axis=0)
            
            batch_loss = 0.0
            
            # Process each sample in batch
            for j in range(end_idx - i):
                # Forward pass
                output = control_system.forward(X_batch[j])
                
                # Compute loss
                loss = ops.stats.mean(ops.square(ops.subtract(output, Y_batch[j])))
                batch_loss += tensor.item(loss)
            
            # Average batch loss
            batch_loss /= (end_idx - i)
            epoch_loss += batch_loss
        
        # Average epoch loss
        epoch_loss /= (num_samples // batch_size)
        losses.append(epoch_loss)
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Test the trained system
    print("Testing the trained system...")
    
    # Generate test data: spiral trajectory
    test_samples = 200
    test_t = tensor.linspace(0, 4 * ops.pi, test_samples)
    
    # Input: position on spiral
    radius = ops.add(1.0, ops.divide(test_t, 4 * ops.pi))
    test_X = tensor.stack([
        ops.multiply(radius, ops.cos(test_t)),
        ops.multiply(radius, ops.sin(test_t))
    ], axis=1)
    
    # Storage for predictions
    predictions = []
    
    # Reset control system
    control_system.reset()
    
    # Generate predictions
    for i in range(test_samples):
        output = control_system.forward(test_X[i])
        predictions.append(output)
    
    # Convert predictions to tensor
    predictions_tensor = tensor.stack(predictions)
    
    # Visualize results
    visualize_learning_results(test_X, predictions_tensor, losses)


def visualize_learning_results(inputs, predictions, losses):
    """Visualize the learning results."""
    # Convert tensors to numpy for plotting
    inputs_np = tensor.to_numpy(inputs)
    predictions_np = tensor.to_numpy(predictions)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot trajectory and predictions
    plt.subplot(1, 2, 1)
    plt.title("Spiral Trajectory with Predicted Velocities")
    plt.scatter(inputs_np[:, 0], inputs_np[:, 1], c='b', s=10, label="Position")
    
    # Plot velocity vectors (subsample for clarity)
    step = 10
    for i in range(0, len(inputs_np), step):
        plt.arrow(
            inputs_np[i, 0], inputs_np[i, 1],
            predictions_np[i, 0] * 0.1, predictions_np[i, 1] * 0.1,
            head_width=0.05, head_length=0.1, fc='r', ec='r', alpha=0.5
        )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.title("Training Loss")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("guce_learning_results.png")
    print("Saved learning results to 'guce_learning_results.png'")


if __name__ == "__main__":
    main()
    learning_example()