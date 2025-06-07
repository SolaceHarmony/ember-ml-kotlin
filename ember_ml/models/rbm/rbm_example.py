"""
RBM Example Script

This script demonstrates how to use the RestrictedBoltzmannMachine and RBMVisualizer
classes to train an RBM on a simple dataset and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Import our modules
from ember_ml.nn import tensor
from ember_ml.models.rbm import RBMModule, train_rbm, save_rbm, load_rbm
from ember_ml.visualization.rbm_visualizer import RBMVisualizer


def generate_toy_data(n_samples=500, n_features=100, pattern_size=10, n_patterns=3):
    """
    Generate toy data with embedded patterns for RBM training.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        pattern_size: Size of each pattern
        n_patterns: Number of different patterns to embed
        
    Returns:
        Generated data with embedded patterns
    """
    # Initialize data with low random values
    data = tensor.random_uniform(0, 0.1, (n_samples, n_features))
    
    # Create patterns
    patterns = []
    for i in range(n_patterns):
        # Create a random binary pattern
        pattern = ops.random_choice([0, 1], size=pattern_size, p=[0.7, 0.3])
        # Scale to [0.7, 1.0] for more pronounced patterns
        pattern = pattern * 0.3 + 0.7
        patterns.append(pattern)
    
    # Embed patterns in random positions
    for i in range(n_samples):
        # Choose a random pattern
        pattern_idx = np.random.randint(0, n_patterns)
        pattern = patterns[pattern_idx]
        
        # Choose a random position to embed the pattern
        start_pos = np.random.randint(0, n_features - pattern_size + 1)
        
        # Embed the pattern
        data[i, start_pos:start_pos + pattern_size] = pattern
    
    return data


def generate_image_data(n_samples=500, width=10, height=10, n_patterns=3):
    """
    Generate image-like data with embedded patterns for RBM training.
    
    Args:
        n_samples: Number of samples to generate
        width: Width of the image
        height: Height of the image
        n_patterns: Number of different patterns to embed
        
    Returns:
        Generated image data with embedded patterns
    """
    n_pixels = width * height
    
    # Initialize data with low random values
    data = tensor.random_uniform(0, 0.1, (n_samples, n_pixels))
    
    # Create patterns (simple geometric shapes)
    patterns = []
    
    # Pattern 1: Horizontal line
    pattern1 = tensor.zeros((height, width))
    mid_row = height // 2
    pattern1[mid_row, :] = 1
    patterns.append(pattern1.flatten())
    
    # Pattern 2: Vertical line
    pattern2 = tensor.zeros((height, width))
    mid_col = width // 2
    pattern2[:, mid_col] = 1
    patterns.append(pattern2.flatten())
    
    # Pattern 3: Cross
    pattern3 = tensor.zeros((height, width))
    pattern3[mid_row, :] = 1
    pattern3[:, mid_col] = 1
    patterns.append(pattern3.flatten())
    
    # Additional patterns if needed
    if n_patterns > 3:
        # Pattern 4: Diagonal line
        pattern4 = tensor.zeros((height, width))
        for i in range(min(height, width)):
            pattern4[i, i] = 1
        patterns.append(pattern4.flatten())
    
    if n_patterns > 4:
        # Pattern 5: Circle (approximated)
        pattern5 = tensor.zeros((height, width))
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        for i in range(height):
            for j in range(width):
                if ((i - center_y) ** 2 + (j - center_x) ** 2) <= radius ** 2:
                    pattern5[i, j] = 1
        patterns.append(pattern5.flatten())
    
    # Ensure we only use the requested number of patterns
    patterns = patterns[:n_patterns]
    
    # Scale patterns to [0.7, 1.0] for more pronounced features
    patterns = [p * 0.3 + 0.7 for p in patterns]
    
    # Embed patterns in the data
    for i in range(n_samples):
        # Choose a random pattern
        pattern_idx = np.random.randint(0, len(patterns))
        pattern = patterns[pattern_idx]
        
        # Add some noise to the pattern
        noisy_pattern = pattern + tensor.random_normal(0, 0.05, n_pixels)
        noisy_pattern = ops.clip(noisy_pattern, 0, 1)
        
        # Embed the pattern with some random background
        data[i] = data[i] * 0.2 + noisy_pattern * 0.8
    
    return data, (height, width)


def main():
    """Main function to demonstrate RBM training and visualization."""
    print("RBM Example Script")
    print("=================")
    
    # Create output directories if they don't exist
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/animations', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    print("\nGenerating toy data...")
    data = generate_toy_data(n_samples=200, n_features=100, pattern_size=10, n_patterns=3)
    
    # Split into training and validation sets
    train_data = data[:180]
    val_data = data[180:]
    
    # Initialize RBM
    print("\nInitializing RBM...")
    rbm = RBMModule(
        n_visible=100,
        n_hidden=20,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        use_binary_states=False
    )
    
    # Train RBM
    print("\nTraining RBM...")
    start_time = time.time()
    
    # Convert to generator for train_rbm
    def data_generator(data, batch_size=10):
        """Convert data to a generator yielding batches."""
        import numpy as np
        from ember_ml.nn import tensor
        n_samples = len(data)
        indices = tensor.random_permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) == 0:
                continue
            batch = data[batch_indices]
            # Return numpy array directly, the training function will convert it
            yield batch
    
    # Train the RBM using the train_rbm function
    training_errors = train_rbm(
        rbm=rbm,
        data_generator=data_generator(train_data),
        epochs=50,
        k=1,
        validation_data=tensor.convert_to_tensor(val_data, dtype=tensor.float32) if val_data is not None else None,
        early_stopping_patience=5
    )
    
    # Print training errors to show progress
    print(f"Training errors: {[float(err) for err in training_errors]}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Print RBM info
    print("\nRBM Info:")
    print(f"Visible units: {rbm.n_visible}")
    print(f"Hidden units: {rbm.n_hidden}")
    print(f"Learning rate: {rbm.learning_rate}")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"outputs/models/rbm_toy_data_{timestamp}.npy"
    save_rbm(rbm, model_path)
    
    # Initialize visualizer
    visualizer = RBMVisualizer()
    
    # Plot training curve
    print("\nPlotting training curve...")
    visualizer.plot_training_curve(rbm, show=True)
    
    # Plot weight matrix
    print("\nPlotting weight matrix...")
    visualizer.plot_weight_matrix(rbm, show=True)
    
    # Plot reconstructions
    print("\nPlotting reconstructions...")
    visualizer.plot_reconstructions(rbm, val_data, n_samples=5, show=True)
    
    # Plot hidden activations
    print("\nPlotting hidden activations...")
    visualizer.plot_hidden_activations(rbm, val_data, n_samples=5, show=True)
    
    # Animate weight evolution
    print("\nAnimating weight evolution...")
    visualizer.animate_weight_evolution(rbm, show=True)
    
    # Animate dreaming
    print("\nAnimating dreaming process...")
    visualizer.animate_dreaming(rbm, n_steps=50, show=True)
    
    # Animate reconstruction
    print("\nAnimating reconstruction process...")
    visualizer.animate_reconstruction(rbm, val_data, n_samples=3, n_steps=10, show=True)
    
    # Generate image data for a more visual example
    print("\n\nGenerating image data...")
    image_data, image_shape = generate_image_data(n_samples=200, width=10, height=10, n_patterns=4)
    
    # Split into training and validation sets
    train_image_data = image_data[:180]
    val_image_data = image_data[180:]
    
    # Initialize RBM for image data
    print("\nInitializing RBM for image data...")
    image_rbm = RBMModule(
        n_visible=100,  # 10x10 pixels
        n_hidden=25,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        use_binary_states=False
    )
    
    # Print training errors to show progress
    print(f"Training errors: {[float(err) for err in training_errors]}")
    
    # Train RBM on image data
    print("\nTraining RBM on image data...")
    start_time = time.time()
    
    # Train the RBM using the train_rbm function
    training_errors = train_rbm(
        rbm=image_rbm,
        data_generator=data_generator(train_image_data),
        epochs=50,
        k=1,
        validation_data=tensor.convert_to_tensor(val_image_data, dtype=tensor.float32) if val_image_data is not None else None,
        early_stopping_patience=5
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    model_path = f"outputs/models/rbm_image_data_{timestamp}.npy"
    save_rbm(image_rbm, model_path)
    
    # Plot reconstructions with image reshaping
    print("\nPlotting image reconstructions...")
    visualizer.plot_reconstructions(
        image_rbm,
        val_image_data,
        n_samples=5,
        reshape=image_shape,
        show=True
    )
    
    # Plot weight matrix with image reshaping
    print("\nPlotting image weight matrix...")
    visualizer.plot_weight_matrix(
        image_rbm,
        reshape_visible=image_shape,
        reshape_hidden=(5, 5),
        show=True
    )
    
    # Animate dreaming with image reshaping
    print("\nAnimating image dreaming process...")
    visualizer.animate_dreaming(
        image_rbm,
        n_steps=50,
        reshape=image_shape,
        show=True
    )
    
    # Animate reconstruction with image reshaping
    print("\nAnimating image reconstruction process...")
    visualizer.animate_reconstruction(
        image_rbm,
        val_image_data,
        n_samples=3,
        n_steps=10,
        reshape=image_shape,
        show=True
    )
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()