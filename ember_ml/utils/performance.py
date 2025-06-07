"""
Performance utilities for the ember_ml library.

This module provides performance utilities for the ember_ml library.
"""

import time
import functools
from typing import Callable, Any, Dict, List, Optional
import matplotlib.pyplot as plt

def timeit(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds to execute")
        return result
    return wrapper

def benchmark(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with benchmark results
    """
    # Warm-up
    for _ in range(3):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(10):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean': stats.mean(times),
        'std': stats.std(times),
        'min': stats.min(times),
        'max': stats.max(times),
        'times': times,
        'result': result
    }

def compare_functions(funcs: List[Callable], args_list: List[tuple], kwargs_list: Optional[List[dict]] = None, 
                      labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare the performance of multiple functions.
    
    Args:
        funcs: List of functions to compare
        args_list: List of arguments to pass to each function
        kwargs_list: List of keyword arguments to pass to each function
        labels: List of labels for each function
        
    Returns:
        Dictionary with comparison results
    """
    if kwargs_list is None:
        kwargs_list = [{} for _ in funcs]
    
    if labels is None:
        labels = [func.__name__ for func in funcs]
    
    results = {}
    for i, (func, args, kwargs, label) in enumerate(zip(funcs, args_list, kwargs_list, labels)):
        print(f"Benchmarking {label}...")
        results[label] = benchmark(func, *args, **kwargs)
    
    return results

def plot_benchmark_results(results: Dict[str, Dict[str, Any]], title: str = "Benchmark Results") -> plt.Figure:
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary with benchmark results
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    labels = list(results.keys())
    means = [results[label]['mean'] for label in labels]
    stds = [results[label]['std'] for label in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = tensor.arange(len(labels))
    ax.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax.set_ylabel('Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)
    
    # Add exact values on top of bars
    for i, v in enumerate(means):
        ax.text(i, v + stds[i], f"{v:.6f}s", ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def memory_usage(func: Callable) -> Callable:
    """
    Decorator to measure the memory usage of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # in MB
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss / 1024 / 1024  # in MB
            print(f"{func.__name__} used {mem_after - mem_before:.2f} MB of memory")
            return result
        except ImportError:
            print("psutil not installed, cannot measure memory usage")
            return func(*args, **kwargs)
    return wrapper

def profile_function(func: Callable, *args, **kwargs) -> None:
    """
    Profile a function using cProfile.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    import cProfile
    import pstats
    from pstats import SortKey
    
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Print top 20 functions