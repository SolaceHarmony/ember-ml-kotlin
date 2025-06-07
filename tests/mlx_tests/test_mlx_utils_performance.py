import pytest
import numpy as np # For comparison with known correct results
import time # For timing
import matplotlib.pyplot as plt # For plotting tests (if any)

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.utils import performance # Import performance utilities
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# --- Helper function to simulate a task ---
def _simulate_task(duration=0.01):
    """Simulate a task that takes some time."""
    time.sleep(duration)
    return "task_completed"

# Test cases for utils.performance functions

def test_timeit_decorator():
    # Test timeit decorator
    @performance.timeit
    def timed_task():
        return _simulate_task(0.02) # Simulate a 0.02 second task

    result = timed_task()

    # timeit decorator prints output, doesn't return time directly.
    # We can check the return value of the decorated function.
    assert result == "task_completed"

    # Checking the printed output would require capturing stdout, which can be complex.
    # For now, we rely on the decorator not raising errors and the function returning correctly.


def test_benchmark():
    # Test benchmark function
    def task_to_benchmark(size):
        # Simulate a task whose time depends on size
        data = tensor.random_normal((size, size))
        result = ops.matmul(data, tensor.transpose(data))
        ops.eval(result) # Ensure computation is complete
        return tensor.item(ops.stats.mean(result)) # Return a scalar

    size = 100
    num_runs = 5
    avg_time, last_result = performance.benchmark(task_to_benchmark, num_runs, size)

    assert isinstance(avg_time, float)
    assert avg_time > 0 # Should take some time
    assert isinstance(last_result, (float, tensor.floating)) # Should return a scalar

    # Check if the benchmark function returns statistics (mean, std, min, max, times)
    # The benchmark function itself returns only avg_time and last_result.
    # The internal timing is handled by time_function.
    # If we wanted to test the statistics calculation, we would need to test time_function directly
    # or modify benchmark to return more details.
    # Based on the documentation, benchmark returns avg_time and last_result.


def test_compare_functions():
    # Test compare_functions
    def task1(size):
        data = tensor.random_normal((size, size))
        result = ops.add(data, data)
        ops.eval(result)
        return tensor.item(ops.stats.mean(result))

    def task2(size):
        data = tensor.random_normal((size, size))
        result = ops.multiply(data, 2.0)
        ops.eval(result)
        return tensor.item(ops.stats.mean(result))

    funcs = [task1, task2]
    args_list = [(100,), (100,)]
    kwargs_list = [{}, {}]
    labels = ["Task 1 (Add)", "Task 2 (Multiply)"]
    num_runs = 5

    # compare_functions calls benchmark internally and prints/plots results.
    # It doesn't return the benchmark results directly.
    # We can test that it runs without errors.
    try:
        performance.compare_functions(funcs, args_list, kwargs_list, labels, num_runs=num_runs)
    except Exception as e:
        pytest.fail(f"compare_functions raised an exception: {e}")

    # Checking the plots would require inspecting generated files, which is outside the scope.


def test_memory_usage_decorator():
    # Test memory_usage decorator
    # Note: Memory profiling can be platform dependent and might require specific setup.
    # This test primarily checks that the decorator runs without errors.
    @performance.memory_usage
    def memory_intensive_task():
        # Simulate creating a large tensor
        large_tensor = tensor.random_normal((1000, 1000))
        # Keep it in scope for a moment
        time.sleep(0.01)
        return "memory_task_completed"

    # Skip if psutil is not installed (required by memory_usage)
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed, skipping memory_usage test")

    result = memory_intensive_task()
    assert result == "memory_task_completed"

    # Checking the printed output would require capturing stdout.


def test_profile_function():
    # Test profile_function
    # Note: Profiling can be complex to test programmatically.
    # This test primarily checks that the function runs without errors.
    def function_to_profile(size):
        data = tensor.random_normal((size, size))
        result = ops.add(data, 1.0)
        ops.eval(result)
        return tensor.item(ops.stats.mean(result))

    size = 50
    try:
        performance.profile_function(function_to_profile, size)
    except Exception as e:
        pytest.fail(f"profile_function raised an exception: {e}")

    # Checking the printed output would require capturing stdout.


# Add more test functions for other performance utilities if any exist and are testable