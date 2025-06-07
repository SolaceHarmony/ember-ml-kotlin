"""
Test for using int32 dtype with tensors.

This module tests the usage of the int32 dtype with tensors.
"""

import pytest
from ember_ml.nn.tensor import EmberTensor, int32


def test_int32_dtype_usage():
    """Test using int32 dtype with tensors."""
    # Create a tensor with int32 dtype
    tensor = EmberTensor([1, 2, 3], dtype=int32)
    
    # Check tensor properties
    assert tensor.shape == (3,)
    assert tensor.dtype is not None
    assert str(tensor.dtype).endswith('int32')
    
    # Create another tensor with the same dtype
    tensor2 = EmberTensor([4, 5, 6], dtype=int32)
    assert str(tensor2.dtype).endswith('int32')
    
    # Check that int32 has the expected attributes
    assert hasattr(int32, 'name')
    assert 'int32' in str(int32)


if __name__ == "__main__":
    test_int32_dtype_usage()
    print("Test passed!")