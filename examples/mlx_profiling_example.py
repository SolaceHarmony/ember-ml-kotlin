"""
MLX Profiling Example

This example demonstrates how to use the MLXProfiler to analyze the performance
of neural networks on Apple Silicon hardware.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.nn.modules import CfC, LTC, ELTC, CTGRU, CTRNN
from ember_ml.nn.modules.wiring import RandomMap, NCPMap, AutoNCP
from ember_ml.nn import tensor
from ember_ml import ops
from examples.mlx_profiler import MLXProfiler, quick_profile

def main():
    """Run the MLX profiling example."""
    print("MLX Profiling Example")
    print("=====================")
    
    # Create different models to compare
    models = {
        'CfC': create_model('cfc'),
        'LTC': create_model('ltc'),
        'ELTC': create_model('eltc'),
        'CTGRU': create_model('ctgru'),
        'CTRNN': create_model('ctrnn')
    }
    
    # Quick profile all models
    results = {}
    for name, model in models.items():
        print(f"\nProfiling {name}...")
        results[name] = quick_profile(model, batch_size=32, seq_length=10, num_runs=20)
        
        # Print summary
        print(f"  Compute time: {results[name]['compute']['time_mean']*1000:.2f} ms")
        print(f"  TFLOPS: {results[name]['compute']['tflops']:.2f}")
        print(f"  Memory usage: {results[name]['memory']['peak_usage']:.2f} MB")
    
    # Compare performance
    compare_performance(results)
    
    # Detailed analysis of one model
    print("\nDetailed Analysis of CfC")
    print("=======================")
    detailed_analysis(models['CfC'])

def create_model(cell_type, hidden_size=100):
    """Create a model with the specified cell type."""
    # Create wiring
    wiring = RandomMap(units=hidden_size, sparsity_level=0.5)
    
    # Create model based on cell type
    if cell_type == 'cfc':
        return CfC(neuron_map=wiring) # Use neuron_map argument
    elif cell_type == 'ltc':
        return LTC(neuron_map=wiring) # Use neuron_map argument
    elif cell_type == 'eltc':
        return ELTC(neuron_map=wiring) # Use neuron_map argument
    elif cell_type == 'ctgru':
        return CTGRU(neuron_map=wiring) # Use neuron_map argument
    elif cell_type == 'ctrnn':
        return CTRNN(neuron_map=wiring) # Use neuron_map argument
    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")

def compare_performance(results):
    """Compare performance of different models."""
    # Extract metrics
    names = list(results.keys())
    compute_times = [results[name]['compute']['time_mean']*1000 for name in names]
    tflops = [results[name]['compute']['tflops'] for name in names]
    memory_usage = [results[name]['memory']['peak_usage'] for name in names]
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Plot compute time
    plt.subplot(131)
    plt.bar(names, compute_times)
    plt.ylabel('Compute Time (ms)')
    plt.title('Computation Time')
    plt.xticks(rotation=45)
    
    # Plot TFLOPS
    plt.subplot(132)
    plt.bar(names, tflops)
    plt.ylabel('TFLOPS')
    plt.title('Compute Efficiency')
    plt.xticks(rotation=45)
    
    # Plot memory usage
    plt.subplot(133)
    plt.bar(names, memory_usage)
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nSaved performance comparison to 'model_comparison.png'")

def detailed_analysis(model):
    """Perform detailed analysis of a model."""
    profiler = MLXProfiler(model)
    
    # Analyze batch size impact
    batch_sizes = [1, 8, 16, 32, 64, 128]
    batch_results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        stats = profiler.profile_compute(batch_size=batch_size, num_runs=10)
        batch_results.append({
            'batch_size': batch_size,
            'time': stats['time_mean'],
            'tflops': stats['tflops']
        })
    
    # Plot batch size impact
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot([r['batch_size'] for r in batch_results],
             [r['time']*1000 for r in batch_results],
             marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title('Compute Time vs Batch Size')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot([r['batch_size'] for r in batch_results],
             [r['tflops'] for r in batch_results],
             marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('TFLOPS')
    plt.title('Compute Efficiency vs Batch Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('batch_size_analysis.png')
    print("Saved batch size analysis to 'batch_size_analysis.png'")
    
    # Get optimization recommendations
    recommendations = profiler.get_optimization_recommendations()
    print("\nOptimization Recommendations:")
    print(f"  Optimal batch size: {recommendations['optimal_batch_size']}")
    print(f"  Peak TFLOPS: {recommendations['peak_tflops']:.2f}")
    print(f"  Memory usage: {recommendations['memory_usage']:.2f} MB")
    print("\nTips:")
    for i, tip in enumerate(recommendations['tips'], 1):
        print(f"  {i}. {tip}")

if __name__ == "__main__":
    main()