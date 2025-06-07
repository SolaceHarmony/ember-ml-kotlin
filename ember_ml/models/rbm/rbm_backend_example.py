"""
Example script demonstrating the backend-agnostic RBM implementation.

This script shows how to use the RBM implementation with different backends
(NumPy, PyTorch, MLX) and compares their performance.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import importlib.util

# Import our neural library
import ember_ml as nl
from ember_ml.models.rbm_backend import RBM

def is_backend_available(backend_name):
    """Check if a backend is available."""
    if backend_name == 'numpy':
        return True
    elif backend_name == 'torch':
        return importlib.util.find_spec('torch') is not None
    elif backend_name == 'mlx':
        return importlib.util.find_spec('mlx') is not None
    return False

def load_digits_dataset():
    """Load and preprocess the digits dataset."""
    # Load digits dataset
    digits = datasets.load_digits()
    X = digits.data.astype(tensor.float32)
    y = digits.target
    
    # Scale data to [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_rbm_with_backend(X_train, X_test, backend, device=None):
    """Train an RBM with the specified backend."""
    print(f"\n--- Training RBM with {backend} backend ---")
    
    # Create RBM
    rbm = RBM(
        n_visible=X_train.shape[1],
        n_hidden=100,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        batch_size=10,
        backend=backend,
        device=device
    )
    
    # Print summary
    print(rbm.summary())
    
    # Train RBM
    start_time = time.time()
    training_errors = rbm.train(
        X_train,
        epochs=20,
        k=1,
        validation_data=X_test,
        verbose=True
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Compute reconstruction error on test set
    test_error = float(nl.to_numpy(rbm.reconstruction_error(X_test)))
    print(f"Test reconstruction error: {test_error:.4f}")
    
    return rbm, training_errors, training_time, test_error

def visualize_results(results):
    """Visualize training errors and performance comparison."""
    backends = list(results.keys())
    
    # Plot training errors
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for backend in backends:
        plt.plot(results[backend]['training_errors'], label=backend)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('Training Errors by Backend')
    plt.legend()
    plt.grid(True)
    
    # Plot performance comparison
    plt.subplot(1, 2, 2)
    training_times = [results[backend]['training_time'] for backend in backends]
    test_errors = [results[backend]['test_error'] for backend in backends]
    
    x = tensor.arange(len(backends))
    width = 0.35
    
    ax1 = plt.gca()
    ax1.bar(x - width/2, training_times, width, label='Training Time (s)')
    ax1.set_ylabel('Training Time (s)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(backends)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, test_errors, width, color='orange', label='Test Error')
    ax2.set_ylabel('Test Error')
    
    plt.title('Performance Comparison')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('rbm_backend_comparison.png')
    plt.show()

def visualize_reconstructions(rbms, X_test):
    """Visualize original and reconstructed digits."""
    backends = list(rbms.keys())
    n_samples = 5
    
    plt.figure(figsize=(12, 2 * (len(backends) + 1)))
    
    # Plot original digits
    for i in range(n_samples):
        plt.subplot(len(backends) + 1, n_samples, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
        plt.title(f"Original {i+1}")
        plt.axis('off')
    
    # Plot reconstructions for each backend
    for b_idx, backend in enumerate(backends):
        rbm = rbms[backend]
        
        # Get reconstructions
        X_reconstructed = nl.to_numpy(rbm.reconstruct(X_test[:n_samples]))
        
        for i in range(n_samples):
            plt.subplot(len(backends) + 1, n_samples, (b_idx + 1) * n_samples + i + 1)
            plt.imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
            plt.title(f"{backend} {i+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('rbm_reconstructions.png')
    plt.show()

def main():
    """Main function."""
    print("RBM Backend Example")
    print("==================")
    
    # Check available backends
    print("\nChecking available backends:")
    backends = ['numpy', 'torch', 'mlx']
    available_backends = []
    
    for backend in backends:
        if is_backend_available(backend):
            print(f"  - {backend}: Available")
            available_backends.append(backend)
        else:
            print(f"  - {backend}: Not available")
    
    # Load dataset
    print("\nLoading digits dataset...")
    X_train, X_test, y_train, y_test = load_digits_dataset()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train RBMs with different backends
    results = {}
    rbms = {}
    
    for backend in available_backends:
        # Determine device
        device = None
        if backend == 'torch' and importlib.util.find_spec('torch.cuda') is not None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else None
        elif backend == 'torch' and importlib.util.find_spec('torch.backends.mps') is not None:
            import torch
            device = 'mps' if torch.backends.mps.is_available() else None
        
        # Train RBM
        rbm, training_errors, training_time, test_error = train_rbm_with_backend(
            X_train, X_test, backend, device
        )
        
        # Store results
        results[backend] = {
            'training_errors': training_errors,
            'training_time': training_time,
            'test_error': test_error
        }
        
        rbms[backend] = rbm
    
    # Visualize results
    visualize_results(results)
    
    # Visualize reconstructions
    visualize_reconstructions(rbms, X_test)
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()