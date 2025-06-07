"""
GUCE Neural Circuit Policy Example

This example demonstrates how to use the GUCENCP and AutoGUCENCP classes
for control and learning tasks.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.nn.modules import GUCENCP, AutoGUCENCP
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor
from ember_ml import ops

def main():
    """Run the GUCE Neural Circuit Policy example."""
    print("GUCE Neural Circuit Policy Example")
    print("=================================")
    
    # Example 1: Using GUCENCP with a custom NCPMap
    print("\nExample 1: GUCENCP with custom NCPMap")
    print("-------------------------------------")
    
    # Create a custom NCPMap
    ncp_map = NCPMap(
        inter_neurons=20,
        command_neurons=10,
        motor_neurons=2,
        sensory_neurons=3,
        sparsity_level=0.5
    )
    
    # Create a GUCENCP with the custom map
    guce_ncp = GUCENCP(
        neuron_map=ncp_map,
        state_dim=32,
        step_size=0.01,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    )
    
    # Generate some input data
    inputs = tensor.random_normal((1, 3))
    
    # Process the input
    outputs = guce_ncp(inputs)
    
    print(f"Input shape: {tensor.shape(inputs)}")
    print(f"Output shape: {tensor.shape(outputs)}")
    print(f"Output values: {outputs}")
    
    # Example 2: Using AutoGUCENCP
    print("\nExample 2: AutoGUCENCP")
    print("---------------------")
    
    # Create an AutoGUCENCP
    auto_guce_ncp = AutoGUCENCP(
        units=32,
        output_size=2,
        sparsity_level=0.5,
        state_dim=32,
        step_size=0.01,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    )
    
    # Generate some input data
    inputs = tensor.random_normal((1, 3))
    
    # Process the input
    outputs = auto_guce_ncp(inputs)
    
    print(f"Input shape: {tensor.shape(inputs)}")
    print(f"Output shape: {tensor.shape(outputs)}")
    print(f"Output values: {outputs}")
    
    # Example 3: Control task with AutoGUCENCP
    print("\nExample 3: Control task with AutoGUCENCP")
    print("---------------------------------------")
    
    # Create an AutoGUCENCP for control
    control_ncp = AutoGUCENCP(
        units=64,
        output_size=1,
        sparsity_level=0.7,
        state_dim=32,
        step_size=0.005,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    )
    
    # Simulation parameters
    dt = 0.01
    sim_time = 5.0  # seconds
    steps = int(sim_time / dt)
    
    # Initial state: pendulum at bottom position
    # Create a 2-element vector directly
    state = tensor.zeros((2,))
    # Set the first element to pi (which is already a tensor)
    state = tensor.with_value(state, 0, ops.pi)  # Extract the scalar value from ops.pi
    
    # Target state: pendulum at top position
    target = tensor.zeros((2,))
    
    # Storage for results
    states = [state]
    actions = []
    
    # Run simulation
    print("Running pendulum control simulation...")
    for i in range(steps):
        # Prepare input for control system
        inputs = tensor.concatenate([state, target[0:1]])
        inputs = tensor.reshape(inputs, (1, 3))
        
        # Get control action
        action = control_ncp(inputs)
        
        # Apply control action to pendulum
        state = pendulum_dynamics(state, action[0, 0], dt)
        
        # Store results
        states.append(state)
        actions.append(action[0, 0])
        
        # Print progress
        if i % 100 == 0:
            print(f"Step {i}/{steps}, Angle: {tensor.item(state[0]):.2f}, Action: {tensor.item(action[0, 0]):.2f}")
    
    # Convert results to tensors
    states_tensor = tensor.stack(states)
    actions_tensor = tensor.stack(actions)
    
    # Visualize results
    visualize_control_results(states_tensor, actions_tensor, dt)


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
    
    # Compute acceleration using ops functions
    term1 = ops.subtract(action, ops.multiply(b, theta_dot))
    term2 = ops.multiply(ops.multiply(ops.multiply(m, g), l), ops.sin(theta))
    numerator = ops.subtract(term1, term2)
    denominator = ops.multiply(m, ops.power(l, 2))
    theta_ddot = ops.divide(numerator, denominator)
    
    # Update state using ops functions
    new_theta = ops.add(theta, ops.multiply(theta_dot, dt))
    new_theta_dot = ops.add(theta_dot, ops.multiply(theta_ddot, dt))
    
    # Normalize angle to [-pi, pi] using ops functions
    two_pi = ops.multiply(2.0, ops.pi) # Use ops.pi directly
    normalized_angle = ops.add(ops.divide(new_theta, two_pi), 0.5)
    floor_val = ops.floor(normalized_angle)
    new_theta = ops.subtract(new_theta, ops.multiply(two_pi, floor_val))
    
    # Create a new state tensor
    new_state = tensor.zeros((2,))
    new_state = tensor.with_value(new_state, 0, new_theta)
    new_state = tensor.with_value(new_state, 1, new_theta_dot)
    
    return new_state


def visualize_control_results(states, actions, dt):
    """Visualize the control results."""
    # Convert tensors to numpy for plotting
    states_np = tensor.to_numpy(states)
    actions_np = tensor.to_numpy(actions)
    
    # Create time vector
    time = tensor.to_numpy(tensor.linspace(0, dt * len(states_np), len(states_np)))
    action_time = tensor.to_numpy(tensor.linspace(0, dt * len(actions_np), len(actions_np)))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot pendulum angle
    plt.subplot(2, 1, 1)
    plt.title("Pendulum Angle")
    plt.plot(time, states_np[:, 0])
    plt.axhline(y=0.0, color='r', linestyle='--', label="Target")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    
    # Plot control actions
    plt.subplot(2, 1, 2)
    plt.title("Control Actions")
    plt.plot(action_time, actions_np)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("guce_ncp_control_results.png")
    print("Saved control results to 'guce_ncp_control_results.png'")


def learning_example():
    """Demonstrate learning with AutoGUCENCP."""
    print("\nExample 4: Learning with AutoGUCENCP")
    print("----------------------------------")
    
    # Create an AutoGUCENCP for learning
    learning_ncp = AutoGUCENCP(
        units=64,
        output_size=2,
        sparsity_level=0.5,
        state_dim=32,
        step_size=0.01,
        nu_0=1.0,
        beta=0.1,
        theta_freq=4.0,
        gamma_freq=40.0,
        dt=0.01
    )
    
    # Generate training data: circular trajectory
    num_samples = 1000
    t = tensor.linspace(0, ops.multiply(2.0, ops.pi), num_samples) # Use ops.pi directly
    
    # Input: position on circle
    X = tensor.zeros((num_samples, 2))
    for i in range(num_samples):
        X = tensor.with_value(X, i, 0, ops.cos(t[i]))
        X = tensor.with_value(X, i, 1, ops.sin(t[i]))
    
    # Output: velocity tangent to circle
    Y = tensor.zeros((num_samples, 2))
    for i in range(num_samples):
        Y = tensor.with_value(Y, i, 0, -ops.sin(t[i]))
        Y = tensor.with_value(Y, i, 1, ops.cos(t[i]))
    
    # Reshape X for the network
    X_reshaped = tensor.reshape(X, (num_samples, 1, 2))
    
    # Training parameters
    epochs = 20
    batch_size = 32
    
    # Storage for results
    losses = []
    
    # Define loss function
    def mse_loss(y_true, y_pred):
        return ops.stats.mean(ops.square(ops.subtract(y_true, y_pred)))
    
    # Simple SGD optimizer
    class SGDOptimizer:
        def __init__(self, learning_rate=0.01):
            self.learning_rate = learning_rate
        
        def update(self, model, grads):
            for param, grad in zip(model.parameters(), grads):
                param.data = ops.subtract(param.data, ops.multiply(grad, self.learning_rate))
    
    # Create optimizer
    optimizer = SGDOptimizer(learning_rate=0.01)
    
    # Train the network
    print("Training the network...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        indices = tensor.random_permutation(num_samples)
        X_shuffled = tensor.take(X_reshaped, indices, axis=0)
        Y_shuffled = tensor.take(Y, indices, axis=0)
        
        # Train in batches
        for i in range(0, num_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, num_samples)
            # Use slice_tensor with start indices and sizes. Assumes 3D/2D input.
            X_batch = tensor.slice_tensor(X_shuffled, [i, 0, 0], [end_idx - i, tensor.shape(X_shuffled)[1], tensor.shape(X_shuffled)[2]])
            Y_batch = tensor.slice_tensor(Y_shuffled, [i, 0], [end_idx - i, tensor.shape(Y_shuffled)[1]])
            
            batch_loss = 0.0
            
            # Process each sample in batch
            for j in range(end_idx - i):
                # Forward pass
                output = learning_ncp(X_batch[j])
                
                # Compute loss
                loss = mse_loss(Y_batch[j], output[0])
                batch_loss += tensor.item(loss)
            
            # Average batch loss using ops.divide
            batch_loss = ops.divide(batch_loss, float(end_idx - i))
            epoch_loss += batch_loss
        
        # Average epoch loss using ops.divide and ops.floor_divide
        batches_per_epoch = ops.floor_divide(num_samples, batch_size)
        # Avoid division by zero if batches_per_epoch is 0
        if tensor.item(batches_per_epoch) > 0:
             epoch_loss = ops.divide(epoch_loss, float(tensor.item(batches_per_epoch)))
        else:
             epoch_loss = 0.0 # Or handle as appropriate
        losses.append(epoch_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Test the trained network
    print("Testing the trained network...")
    
    # Generate test data: spiral trajectory
    test_samples = 200
    test_t = tensor.linspace(0, ops.multiply(4.0, ops.pi), test_samples) # Use ops.pi directly
    
    # Input: position on spiral
    radius = ops.add(1.0, ops.divide(test_t, ops.multiply(4.0, ops.pi))) # Use ops.pi directly
    test_X = tensor.zeros((test_samples, 2))
    for i in range(test_samples):
        test_X = tensor.with_value(test_X, i, 0, ops.multiply(radius[i], ops.cos(test_t[i])))
        test_X = tensor.with_value(test_X, i, 1, ops.multiply(radius[i], ops.sin(test_t[i])))
    
    # Reshape for the network
    test_X_reshaped = tensor.reshape(test_X, (test_samples, 1, 2))
    
    # Generate predictions
    predictions = []
    for i in range(test_samples):
        output = learning_ncp(test_X_reshaped[i])
        predictions.append(output[0])
    
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
    plt.savefig("guce_ncp_learning_results.png")
    print("Saved learning results to 'guce_ncp_learning_results.png'")


if __name__ == "__main__":
    main()
    learning_example()