"""
Test the sort and argsort functions with descending parameter for EmberTensor.

This module tests the sort and argsort functions with the descending parameter
to ensure they work correctly across different backends.
"""

import pytest

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import convert_to_tensor


@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = ops.get_backend()
    yield original
    # Ensure original is not None before setting it
    if original is not None:
        ops.set_backend(original)
    else:
        # Default to 'numpy' if original is None
        ops.set_backend('numpy')


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_sort_descending(backend_name, original_backend):
    """Test sort with descending parameter using tensor interface."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor using the frontend interface
        data = [[5, 2, 8], [1, 9, 3], [7, 4, 6]]
        t = tensor.convert_to_tensor(data)
        
        # Sort in ascending order (default)
        sorted_asc = tensor.sort(t, axis=0)
        
        # Sort in descending order
        sorted_desc = tensor.sort(t, axis=0, descending=True)
        
        # Verify that the descending sort is the reverse of the ascending sort
        for col in range(3):
            # Create slices to extract the column
            asc_col = tensor.slice(sorted_asc, [0, col], [3, 1])
            desc_col = tensor.slice(sorted_desc, [0, col], [3, 1])
            
            # Check if the descending sort is the reverse of the ascending sort
            # For each column, the first element of desc should equal the last element of asc
            assert ops.equal(
                tensor.slice(desc_col, [0, 0], [1, 1]), 
                tensor.slice(sorted_asc, [2, col], [1, 1])
            )
            
            # The middle element should be the same if all values are unique
            # or could be the same if there are duplicates
            
            # The last element of desc should equal the first element of asc
            assert ops.equal(
                tensor.slice(desc_col, [2, 0], [1, 1]), 
                tensor.slice(sorted_asc, [0, col], [1, 1])
            )
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_argsort_descending(backend_name, original_backend):
    """Test argsort with descending parameter using tensor interface."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor using the frontend interface
        data = [[5, 2, 8], [1, 9, 3], [7, 4, 6]]
        t = tensor.convert_to_tensor(data)
        
        # Get the argsort indices in ascending order (default)
        indices_asc = tensor.argsort(t, axis=0)
        
        # Get the argsort indices in descending order
        indices_desc = tensor.argsort(t, axis=0, descending=True)
        
        # Verify that the indices produce values in the correct order
        for col in range(3):
            # Get the original column values
            col_values = tensor.slice(t, [0, col], [3, 1])
            
            # Get the indices for this column
            asc_indices_col = tensor.slice(indices_asc, [0, col], [3, 1])
            desc_indices_col = tensor.slice(indices_desc, [0, col], [3, 1])
            
            # Instead of using gather, we'll directly check the values in the original tensor
            # For the first element in descending order (should be the maximum value)
            first_desc_idx = tensor.slice(desc_indices_col, [0, 0], [1, 1])
            first_desc_idx_val = first_desc_idx.tolist()[0][0]  # Extract the index value
            first_desc_val = tensor.slice(col_values, [first_desc_idx_val, 0], [1, 1])
            
            # For the first element in ascending order (should be the minimum value)
            first_asc_idx = tensor.slice(asc_indices_col, [0, 0], [1, 1])
            first_asc_idx_val = first_asc_idx.tolist()[0][0]  # Extract the index value
            first_asc_val = tensor.slice(col_values, [first_asc_idx_val, 0], [1, 1])
            
            # For the last element in descending order (should be the minimum value)
            last_desc_idx = tensor.slice(desc_indices_col, [2, 0], [1, 1])
            last_desc_idx_val = last_desc_idx.tolist()[0][0]  # Extract the index value
            last_desc_val = tensor.slice(col_values, [last_desc_idx_val, 0], [1, 1])
            
            # For the last element in ascending order (should be the maximum value)
            last_asc_idx = tensor.slice(asc_indices_col, [2, 0], [1, 1])
            last_asc_idx_val = last_asc_idx.tolist()[0][0]  # Extract the index value
            last_asc_val = tensor.slice(col_values, [last_asc_idx_val, 0], [1, 1])
            
            # Verify that first_desc_val is the maximum value
            assert ops.all(ops.greater_equal(first_desc_val, col_values))
            
            # Verify that first_asc_val is the minimum value
            assert ops.all(ops.less_equal(first_asc_val, col_values))
            
            # Verify that last_desc_val is the minimum value
            assert ops.all(ops.less_equal(last_desc_val, col_values))
            
            # Verify that last_asc_val is the maximum value
            assert ops.all(ops.greater_equal(last_asc_val, col_values))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")