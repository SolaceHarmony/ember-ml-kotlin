"""
Memory-Optimized Training Example

This example demonstrates how to use the MemoryOptimizedTrainer to train
neural networks efficiently on Apple Silicon hardware.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.nn.modules import CfC, LTC, ELTC
from ember_ml.nn.modules.wiring import AutoNCP
from ember_ml.nn.modules.trainers import MemoryOptimizedTrainer
from ember_ml.nn import tensor
from ember_ml import ops

def main():
    """Run the memory-optimized training example."""
    print("Memory-Optimized Training Example")
    print("================================")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_time_series_data()
    
    # Create models with different configurations
    configs = [
        ('Small', 32, 32),
        ('Medium', 64, 64),
        ('Large', 128, 128)
    ]
    
    results = {}
    for name, hidden_size, batch_size in configs:
        print(f"\nTraining {name} model...")
        
        # Create model
        model = create_model(hidden_size)
        
        # Create optimizer
        optimizer = create_optimizer(learning_rate=0.001)
        
        # Create trainer
        trainer = MemoryOptimizedTrainer(model, optimizer, compile_training=True)
        
        # Train model
        start_time = time.time()
        history = trainer.train(
            X, y,
            batch_size=batch_size,
            epochs=5,
            verbose=True
        )
        total_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'history': history,
            'total_time': total_time,
            'batch_size': batch_size,
            'hidden_size': hidden_size
        }
        
        # Print summary
        print(f"  Final loss: {history['loss'][-1]:.4f}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Memory usage: {history['memory'][-1]:.2f} MB")
    
    # Compare results
    compare_results(results)

def generate_time_series_data(n_samples=1000, seq_length=16, n_features=8):
    """Generate synthetic time series data."""
    # Time points
    t_values = tensor.linspace(0, 8 * ops.pi, seq_length)
    
    # Generate data
    X = tensor.zeros((n_samples, seq_length, n_features))
    y = tensor.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        # Generate components with different frequencies using ops
        trend = ops.multiply(0.1, t_values)
        
        # Random phase shift using ops
        phase_shift = ops.multiply(ops.pi, tensor.random_uniform(()))
        seasonal1 = ops.multiply(2.0, ops.sin(ops.add(t_values, phase_shift)))
        
        phase_shift2 = ops.multiply(ops.pi, tensor.random_uniform(()))
        seasonal2 = ops.sin(ops.add(ops.multiply(2.0, t_values), phase_shift2))
        
        noise = ops.multiply(0.2, tensor.random_normal((seq_length,)))
        
        # Combine components using ops
        base_signal = ops.add(
            ops.add(trend, seasonal1),
            ops.add(seasonal2, noise)
        )
        
        # Create multiple features
        for j in range(n_features):
            phase_shift_j = ops.multiply(ops.divide(ops.multiply(float(j), ops.pi), 4.0), tensor.ones(())) # Use ops
            amplitude = ops.add(1.0, ops.multiply(0.2, float(j))) # Use ops
            
            feature_signal = ops.add(
                ops.multiply(amplitude, base_signal),
                ops.sin(ops.add(t_values, phase_shift_j))
            )
            
            # Update X tensor
            for k in range(seq_length):
                X = tensor.with_value(X, i, k, j, feature_signal[k])
        
        # Generate target (next value) using ops
        t_step = ops.subtract(t_values[1], t_values[0])
        t_next = ops.add(t_values[-1], t_step)
        
        for j in range(n_features):
            phase_shift_j = ops.multiply(ops.divide(ops.multiply(float(j), ops.pi), 4.0), tensor.ones(())) # Use ops
            amplitude = ops.add(1.0, ops.multiply(0.2, float(j))) # Use ops
            
            trend_next = ops.multiply(0.1, t_next)
            
            phase_next = ops.multiply(ops.pi, tensor.random_uniform(()))
            seasonal1_next = ops.multiply(2.0, ops.sin(ops.add(t_next, phase_next)))
            
            phase_next2 = ops.multiply(ops.pi, tensor.random_uniform(()))
            seasonal2_next = ops.sin(ops.add(ops.multiply(2.0, t_next), phase_next2))
            
            target_value = ops.multiply(
                amplitude,
                ops.add(
                    ops.add(trend_next, seasonal1_next),
                    seasonal2_next
                )
            )
            
            # Update y tensor
            y = tensor.with_value(y, i, j, target_value)
    
    return X, y

def create_model(hidden_size):
    """Create a model with the specified hidden size."""
    # Create wiring
    wiring = AutoNCP(units=hidden_size, output_size=8, sparsity_level=0.5)
    
    # Create model
    return CfC(neuron_map=wiring) # Use neuron_map argument

def create_optimizer(learning_rate=0.001):
    """Create an optimizer."""
    # Simple SGD optimizer
    class SGDOptimizer:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate
        
        def update(self, model, grads):
            for param, grad in zip(model.parameters(), grads):
                # Use ops functions for update
                update = ops.multiply(grad, self.learning_rate)
                param.data = ops.subtract(param.data, update)
    
    return SGDOptimizer(learning_rate)

def compare_results(results):
    """Compare training results."""
    # Extract metrics
    names = list(results.keys())
    final_losses = [results[name]['history']['loss'][-1] for name in names]
    training_times = [results[name]['total_time'] for name in names]
    memory_usages = [results[name]['history']['memory'][-1] for name in names]
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(221)
    for name in names:
        plt.plot(results[name]['history']['loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot memory usage
    plt.subplot(222)
    for name in names:
        plt.plot(results[name]['history']['memory'], marker='o', label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title('Peak Memory Usage')
    plt.legend()
    plt.grid(True)
    
    # Plot training time
    plt.subplot(223)
    for name in names:
        plt.plot(results[name]['history']['time'], marker='o', label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot batch size vs. memory
    plt.subplot(224)
    batch_sizes = [results[name]['batch_size'] for name in names]
    hidden_sizes = [results[name]['hidden_size'] for name in names]
    plt.scatter(batch_sizes, memory_usages, s=100, c=range(len(names)), cmap='viridis')
    for i, name in enumerate(names):
        plt.annotate(name, (batch_sizes[i], memory_usages[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Batch Size vs. Memory Usage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png')
    print("\nSaved training comparison to 'training_comparison.png'")
    
    # Print optimization recommendations
    print("\nOptimization Recommendations:")
    print("  1. Use power-of-2 sizes for tensors and batch sizes")
    print("  2. Enable compilation for static shapes")
    print("  3. Balance batch size and model size based on your hardware")
    print("  4. Monitor memory usage during training")
    print("  5. Consider using sparse connectivity for large models")
    
    # Device-specific recommendations
    print("\nDevice-Specific Settings:")
    print("  - M1: 32-64 batch size")
    print("  - M1 Pro/Max: 64-128 batch size")
    print("  - M1 Ultra: 128-256 batch size")
    print("  - Adjust based on your model size")

if __name__ == "__main__":
    main()