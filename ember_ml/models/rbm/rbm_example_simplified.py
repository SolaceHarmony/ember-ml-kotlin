"""
RBM Example Script (Simplified)

This script demonstrates how to use the RBMModule and RBMVisualizer
classes to visualize an RBM on a simple dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Import our modules
from ember_ml.nn import tensor
from ember_ml.models.rbm import RBMModule
from ember_ml.visualization.rbm_visualizer import RBMVisualizer


def generate_toy_data(n_samples=500, n_features=100, pattern_size=10, n_patterns=3):
    """
    Generate toy data with embedded patterns for RBM visualization.
    
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
    Generate image-like data with embedded patterns for RBM visualization.
    
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
    """Main function to demonstrate RBM visualization."""
    print("RBM Example Script (Simplified)")
    print("==============================")
    
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
    
    # Initialize RBM with random weights
    print("\nInitializing RBM...")
    rbm = RBMModule(
        n_visible=100,
        n_hidden=20,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        use_binary_states=False
    )
    
    # Initialize visualizer
    visualizer = RBMVisualizer()
    
    # Plot weight matrix
    print("\nPlotting weight matrix...")
    visualizer.plot_weight_matrix(rbm, show=True)
    
    # Plot reconstructions
    print("\nPlotting reconstructions...")
    # Convert val_data to tensor
    val_tensor = tensor.convert_to_tensor(val_data, dtype=tensor.float32)
    visualizer.plot_reconstructions(rbm, val_data, n_samples=5, show=True)
    
    # Plot hidden activations
    print("\nPlotting hidden activations...")
    visualizer.plot_hidden_activations(rbm, val_data, n_samples=5, show=True)
    
    # Animate dreaming
    print("\nAnimating dreaming process...")
    visualizer.animate_dreaming(rbm, n_steps=50, show=True)
    # Skip the reconstruction animation for now as it has issues
    print("\nSkipping reconstruction animation due to compatibility issues...")
    
    # Generate image data for a more visual example
    print("\n\nGenerating image data...")
    image_data, image_shape = generate_image_data(n_samples=200, width=10, height=10, n_patterns=3)
    
    # Split into training and validation sets
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
    
    # Plot weight matrix with image reshaping
    print("\nPlotting image weight matrix...")
    visualizer.plot_weight_matrix(
        image_rbm,
        reshape_visible=image_shape,
        reshape_hidden=(5, 5),
        show=True
    )
    
    # Skip the dreaming animation for now as it might have issues
    print("\nSkipping image dreaming animation due to potential compatibility issues...")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()