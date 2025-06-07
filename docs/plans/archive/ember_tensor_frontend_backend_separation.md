# Architectural Plan: EmberTensor Frontend/Backend Separation

## Current Architecture Analysis

Based on examination of the codebase, I've identified a key architectural issue:

1. The current `tensor.convert_to_tensor` function returns native backend tensor types (numpy.ndarray, torch.Tensor, mlx.core.array) rather than EmberTensor objects.
2. This creates inconsistency in the frontend API, where some operations return EmberTensor objects while others return backend-specific tensor types.
3. Tests are currently written with mixed approaches - some using EmberTensor directly, others using the tensor module functions that return backend tensors.

## Architectural Principle

The frontend should never expose backend tensors to users. All frontend interfaces should consistently return EmberTensor objects, maintaining a clean separation between the frontend API and backend implementations.

## Memory Efficiency Requirements

To maintain memory efficiency:

1. EmberTensor should only have two internal variables: `_tensor` and `_dtype` to avoid doubling memory requirements.
2. When passing EmberTensor objects between functions, we should leverage Python's reference passing to avoid creating unnecessary copies.
3. Operations should modify EmberTensor objects in-place when appropriate, or return new EmberTensor objects that reference the same underlying data when possible.

## Proposed Architectural Changes

### 1. Create Private/Public API Separation

- Rename the current `convert_to_tensor` to `_convert_to_backend_tensor` (private)
- Create a new public `convert_to_tensor` that wraps the result in an EmberTensor

```python
# In ember_ml/nn/tensor/common/__init__.py
_convert_to_backend_tensor = lambda *args, **kwargs: _get_tensor_ops().convert_to_tensor(*args, **kwargs)

# In ember_ml/nn/tensor/__init__.py
def convert_to_tensor(data, dtype=None, device=None, requires_grad=False):
    """
    Convert any tensor or array-like object to an EmberTensor.
    
    Args:
        data: Input data (array, list, scalar, or tensor)
        dtype: Optional data type
        device: Optional device to place the tensor on
        requires_grad: Whether the tensor requires gradients
        
    Returns:
        EmberTensor
    """
    # If already an EmberTensor, return it directly (reference passing)
    if isinstance(data, EmberTensor):
        return data
    
    # Convert to backend tensor first
    backend_tensor = _convert_to_backend_tensor(data, dtype=dtype)
    
    # Wrap in EmberTensor
    return EmberTensor(backend_tensor, dtype=dtype)
```

### 2. Streamline EmberTensor Class

Simplify the EmberTensor class to only maintain the essential internal state:

```python
class EmberTensor:
    def __init__(self, data, dtype=None):
        """Initialize an EmberTensor with minimal state."""
        # If data is already an EmberTensor, extract its backend tensor (reference)
        if isinstance(data, EmberTensor):
            self._tensor = data._tensor  # Direct reference to the same backend tensor
            self._dtype = data._dtype if dtype is None else dtype
        else:
            # Convert to backend tensor if needed
            self._tensor = data if is_backend_tensor(data) else _convert_to_backend_tensor(data, dtype=dtype)
            self._dtype = dtype
    
    @property
    def dtype(self):
        """Get the dtype of the tensor."""
        return self._dtype
    
    @property
    def shape(self):
        """Get the shape of the tensor using the backend function."""
        return shape(self._tensor)
    
    # Other properties and methods that don't store additional state
```

### 3. Implement Python Protocol Methods

Implement Python protocol methods to make EmberTensor behave like native tensors:

```python
class EmberTensor:
    # ... existing code ...
    
    # String representation protocols
    def __repr__(self):
        """Return a string representation of the tensor for developers."""
        return f"EmberTensor({to_numpy(self._tensor)}, dtype={self.dtype})"
    
    def __str__(self):
        """Return a string representation of the tensor for users."""
        return f"EmberTensor({to_numpy(self._tensor)}, dtype={self.dtype})"
    
    # Iteration protocols
    def __iter__(self):
        """Return an iterator over the tensor."""
        # Get the first dimension size
        size = self.shape[0] if len(self.shape) > 0 else 0
        
        # Iterate over the first dimension
        for i in range(size):
            # Extract the i-th element and wrap it in an EmberTensor
            yield EmberTensor(slice(self._tensor, [i], [1]), dtype=self._dtype)
    
    def __len__(self):
        """Return the length of the first dimension."""
        return self.shape[0] if len(self.shape) > 0 else 0
    
    # Container protocols
    def __getitem__(self, key):
        """Get values at specified indices."""
        # Use slice_update to implement indexing
        result = slice_update(self._tensor, key, None)
        return EmberTensor(result, dtype=self._dtype)
    
    def __setitem__(self, key, value):
        """Set values at specified indices."""
        if isinstance(value, EmberTensor):
            value = value._tensor
        self._tensor = slice_update(self._tensor, key, value)
    
    def __contains__(self, item):
        """Check if item is in the tensor."""
        # Convert item to a tensor if needed
        if not isinstance(item, EmberTensor):
            item = _convert_to_backend_tensor(item)
        # Check if any element equals item
        return ops.any(ops.equal(self._tensor, item))
    
    # Numeric protocols - arithmetic operations
    def __add__(self, other):
        """Add two tensors."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.add(self._tensor, other_tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __radd__(self, other):
        """Reverse add operation."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtract two tensors."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.subtract(self._tensor, other_tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __rsub__(self, other):
        """Reverse subtract operation."""
        other_tensor = other if not isinstance(other, EmberTensor) else other._tensor
        result = ops.subtract(other_tensor, self._tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __mul__(self, other):
        """Multiply two tensors."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.multiply(self._tensor, other_tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __rmul__(self, other):
        """Reverse multiply operation."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divide two tensors."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.divide(self._tensor, other_tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __rtruediv__(self, other):
        """Reverse divide operation."""
        other_tensor = other if not isinstance(other, EmberTensor) else other._tensor
        result = ops.divide(other_tensor, self._tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __neg__(self):
        """Negate the tensor."""
        result = ops.negative(self._tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __abs__(self):
        """Absolute value of the tensor."""
        result = ops.abs(self._tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __pow__(self, other):
        """Raise tensor to a power."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.power(self._tensor, other_tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    def __rpow__(self, other):
        """Reverse power operation."""
        other_tensor = other if not isinstance(other, EmberTensor) else other._tensor
        result = ops.power(other_tensor, self._tensor)
        return EmberTensor(result, dtype=self._dtype)
    
    # Comparison protocols
    def __eq__(self, other):
        """Check if two tensors are equal."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.equal(self._tensor, other_tensor)
        return EmberTensor(result, dtype=bool_)
    
    def __ne__(self, other):
        """Check if two tensors are not equal."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.not_equal(self._tensor, other_tensor)
        return EmberTensor(result, dtype=bool_)
    
    def __lt__(self, other):
        """Check if tensor is less than other."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.less(self._tensor, other_tensor)
        return EmberTensor(result, dtype=bool_)
    
    def __le__(self, other):
        """Check if tensor is less than or equal to other."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.less_equal(self._tensor, other_tensor)
        return EmberTensor(result, dtype=bool_)
    
    def __gt__(self, other):
        """Check if tensor is greater than other."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.greater(self._tensor, other_tensor)
        return EmberTensor(result, dtype=bool_)
    
    def __ge__(self, other):
        """Check if tensor is greater than or equal to other."""
        other_tensor = other._tensor if isinstance(other, EmberTensor) else other
        result = ops.greater_equal(self._tensor, other_tensor)
        return EmberTensor(result, dtype=bool_)
    
    # Type conversion protocols
    def __int__(self):
        """Convert tensor to int."""
        # Only works for scalar tensors
        if len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1):
            return int(ops.item(self._tensor))
        raise ValueError("Only scalar tensors can be converted to int")
    
    def __float__(self):
        """Convert tensor to float."""
        # Only works for scalar tensors
        if len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1):
            return float(ops.item(self._tensor))
        raise ValueError("Only scalar tensors can be converted to float")
    
    def __bool__(self):
        """Convert tensor to bool."""
        # Only works for scalar tensors
        if len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1):
            return bool(ops.item(self._tensor))
        raise ValueError("Only scalar tensors can be converted to bool")
    
    def __array__(self):
        """NumPy array interface."""
        return to_numpy(self._tensor)
```

### 4. Update All Tensor Module Functions

Modify all functions in the tensor module to wrap their results in EmberTensor objects. This ensures consistent return types across the entire frontend API.

For each function in the tensor module:

```python
# Current implementation
function_name = lambda *args, **kwargs: _get_tensor_ops().function_name(*args, **kwargs)

# New implementation
def function_name(*args, **kwargs):
    # Process arguments, unwrapping EmberTensor objects to get backend tensors
    processed_args = [arg._tensor if isinstance(arg, EmberTensor) else arg for arg in args]
    processed_kwargs = {k: v._tensor if isinstance(v, EmberTensor) else v for k, v in kwargs.items()}
    
    # Call backend function
    backend_result = _get_tensor_ops().function_name(*processed_args, **processed_kwargs)
    
    # Wrap result in EmberTensor if it's a tensor
    if is_backend_tensor(backend_result):
        return EmberTensor(backend_result)
    # If result is a list of tensors, wrap each one
    elif isinstance(backend_result, list) and all(is_backend_tensor(item) for item in backend_result):
        return [EmberTensor(item) for item in backend_result]
    # Otherwise, return as is (for scalar results, etc.)
    return backend_result
```

### 5. Implementation Strategy

1. Create a helper function to determine if an object is a backend tensor:

```python
def is_backend_tensor(obj):
    """Check if an object is a backend tensor."""
    backend = get_backend()
    if backend == 'numpy':
        import numpy as np
        return isinstance(obj, np.ndarray)
    elif backend == 'torch':
        import torch
        return isinstance(obj, torch.Tensor)
    elif backend == 'mlx':
        import mlx.core
        return isinstance(obj, mlx.core.array)
    return False
```

2. Implement the streamlined EmberTensor class with minimal state and Python protocol methods.

3. Update all tensor operations to ensure they return EmberTensor objects.

### 6. Testing Strategy

1. Create tests to verify that all tensor operations return EmberTensor objects:

```python
def test_operations_return_ember_tensor():
    """Test that all tensor operations return EmberTensor objects."""
    # Test creation functions
    assert isinstance(tensor.zeros((2, 3)), EmberTensor)
    assert isinstance(tensor.ones((2, 3)), EmberTensor)
    
    # Test conversion function
    data = [1, 2, 3]
    assert isinstance(tensor.convert_to_tensor(data), EmberTensor)
    
    # Test operations
    t = tensor.convert_to_tensor(data)
    assert isinstance(tensor.reshape(t, (1, 3)), EmberTensor)
    # etc.
```

2. Add tests to verify memory efficiency:

```python
def test_reference_passing():
    """Test that EmberTensor objects are passed by reference."""
    t1 = tensor.convert_to_tensor([1, 2, 3])
    t2 = t1  # Should be a reference to the same object
    assert t1 is t2  # Identity check
    
    # Modifying t2 should affect t1
    t2[0] = 10
    assert tensor.item(t1[0]) == 10
```

3. Add comprehensive tests for Python protocol methods:

```python
def test_string_representation():
    """Test string representation of EmberTensor."""
    t = tensor.convert_to_tensor([[1, 2], [3, 4]])
    
    # Test __str__ and __repr__
    assert "EmberTensor" in str(t)
    assert "dtype" in str(t)
    assert "EmberTensor" in repr(t)
    assert "dtype" in repr(t)

def test_iteration_and_length():
    """Test iteration and length protocols."""
    t = tensor.convert_to_tensor([[1, 2], [3, 4]])
    
    # Test __iter__
    items = [item for item in t]
    assert len(items) == 2
    assert isinstance(items[0], EmberTensor)
    assert tensor.item(items[0][0]) == 1
    assert tensor.item(items[0][1]) == 2
    assert tensor.item(items[1][0]) == 3
    assert tensor.item(items[1][1]) == 4
    
    # Test __len__
    assert len(t) == 2

def test_container_protocols():
    """Test container protocols (__getitem__, __setitem__, __contains__)."""
    t = tensor.convert_to_tensor([1, 2, 3, 4])
    
    # Test __getitem__
    assert tensor.item(t[0]) == 1
    assert tensor.item(t[1]) == 2
    
    # Test __setitem__
    t[0] = 10
    assert tensor.item(t[0]) == 10
    
    # Test __contains__
    assert 10 in t
    assert 5 not in t

def test_arithmetic_operations():
    """Test arithmetic operations."""
    t1 = tensor.convert_to_tensor([1, 2, 3])
    t2 = tensor.convert_to_tensor([4, 5, 6])
    scalar = 2
    
    # Test __add__ and __radd__
    result = t1 + t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 5
    
    result = t1 + scalar
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 3
    
    result = scalar + t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 3
    
    # Test __sub__ and __rsub__
    result = t2 - t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 3
    
    result = t1 - scalar
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == -1
    
    result = scalar - t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 1
    
    # Test __mul__ and __rmul__
    result = t1 * t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 4
    
    result = t1 * scalar
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 2
    
    result = scalar * t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 2
    
    # Test __truediv__ and __rtruediv__
    result = t2 / t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 4
    
    result = t1 / scalar
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 0.5
    
    result = scalar / t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 2
    
    # Test __neg__
    result = -t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == -1
    
    # Test __abs__
    result = abs(-t1)
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 1
    
    # Test __pow__ and __rpow__
    result = t1 ** 2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 1
    
    result = 2 ** t1
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == 2

def test_comparison_operations():
    """Test comparison operations."""
    t1 = tensor.convert_to_tensor([1, 2, 3])
    t2 = tensor.convert_to_tensor([3, 2, 1])
    
    # Test __eq__
    result = t1 == t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == False
    assert tensor.item(result[1]) == True
    
    # Test __ne__
    result = t1 != t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == True
    assert tensor.item(result[1]) == False
    
    # Test __lt__
    result = t1 < t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == True
    assert tensor.item(result[2]) == False
    
    # Test __le__
    result = t1 <= t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == True
    assert tensor.item(result[1]) == True
    
    # Test __gt__
    result = t1 > t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == False
    assert tensor.item(result[2]) == True
    
    # Test __ge__
    result = t1 >= t2
    assert isinstance(result, EmberTensor)
    assert tensor.item(result[0]) == False
    assert tensor.item(result[1]) == True

def test_type_conversion():
    """Test type conversion protocols."""
    # Scalar tensor
    t_scalar = tensor.convert_to_tensor(5)
    
    # Test __int__
    assert int(t_scalar) == 5
    
    # Test __float__
    assert float(t_scalar) == 5.0
    
    # Test __bool__
    assert bool(t_scalar) == True
    
    # Test __array__
    import numpy as np
    arr = np.array(t_scalar)
    assert arr.item() == 5
```

4. Update existing tests to use the new API consistently.

## Migration Path

This change will require significant refactoring, but can be implemented incrementally:

1. First, create the new API structure with both private and public functions
2. Update the most critical/commonly used functions first:
   - convert_to_tensor
   - zeros, ones, eye, etc.
   - reshape, transpose, etc.
3. Implement Python protocol methods for EmberTensor
4. Gradually migrate all remaining functions
5. Update tests to verify the new behavior

## Backward Compatibility Considerations

To maintain backward compatibility during the transition:

1. Keep the original functions available but mark them as deprecated
2. Add warnings when the deprecated functions are used
3. Provide clear migration guides for users

## Timeline and Milestones

1. **Phase 1 (1-2 days)**: Create the private/public API separation and update core functions
2. **Phase 2 (1-2 days)**: Implement Python protocol methods for EmberTensor
3. **Phase 3 (2-3 days)**: Update all remaining tensor operations
4. **Phase 4 (1-2 days)**: Update tests and documentation
5. **Phase 5 (1 day)**: Final testing and release

## Conclusion

This architectural change will significantly improve the consistency and usability of the EmberML tensor API while maintaining memory efficiency. By ensuring that all frontend functions return EmberTensor objects, implementing Python protocol methods, and leveraging Python's reference passing, we maintain a clean separation between the frontend and backend without doubling memory requirements.