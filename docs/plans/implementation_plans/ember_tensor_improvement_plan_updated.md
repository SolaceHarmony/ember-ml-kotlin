# EmberTensor Design Improvement Plan (Updated)

## Current Issues

1. **Inconsistent Dtype Handling**:
   - EmberTensor doesn't store dtype explicitly as a member variable (`_dtype`)
   - Instead relies on extracting dtype from the underlying tensor each time it's accessed
   - This can lead to inconsistencies if the underlying tensor's dtype changes

2. **Tensor Consistency Challenges**:
   - No clear mechanism to ensure consistent handling of `_tensor` across operations
   - New tensors created during operations may not maintain the same properties

3. **Missing Serialization Support**:
   - No explicit serialization methods (`__getstate__`, `__setstate__`, `__iter__`)
   - Makes saving/loading tensors difficult, especially with NumPy's pickle-based approach

4. **Lack of Backend Metadata**:
   - No mechanism to track which backend the tensor data originated from
   - This information would be valuable for save/load operations

5. **Confusion Between Public and Internal convert_to_tensor**:
   - The current implementation doesn't clearly distinguish between the public `convert_to_tensor` (which should return EmberTensor) and the internal `_convert_to_backend_tensor` (which should return backend-specific tensors)
   - This leads to inconsistent return types and potential bugs

6. **Backend Purity Violations**:
   - The `__array__` method uses `to_numpy()` which violates backend purity
   - The `__iter__` method also uses `to_numpy()` instead of iterating on the backend tensor directly

## Backend Architecture Analysis

After thoroughly examining the backend implementations, I've identified the following key architectural patterns:

1. **Backend-Specific Tensor Classes**:
   - Each backend (NumPy, PyTorch, MLX) has its own tensor class (`NumpyTensor`, `TorchTensor`, `MLXTensor`)
   - These classes implement the same interface but with backend-specific implementations
   - Each backend tensor class has a `_dtype_cls` instance for dtype handling

2. **Operation Implementation Pattern**:
   - Backend tensor classes delegate to specialized operation modules
   - Operations are implemented as standalone functions that take the tensor object as the first argument
   - This follows a consistent pattern across all backends

3. **IO Operations**:
   - Each backend has its own IO operations implementation
   - NumPy uses `np.save` and `np.load` with pickle
   - PyTorch uses `torch.save` and `torch.load`
   - MLX uses `mx.save` and `mx.load`
   - The ops module has a special case for NumPy IO operations (`NumPyIOOps` vs `NumpyIOOps`)

4. **Tensor Creation Flow**:
   - Frontend tensor operations are defined as lambda functions in `ember_ml/nn/tensor/common/__init__.py`
   - These functions delegate to the backend implementations through `_get_tensor_ops()`
   - EmberTensor methods call these lambda functions

5. **Backend Iteration Behavior**:
   - PyTorch tensors return `torch.Tensor` objects when iterating
   - MLX arrays return `mlx.core.array` objects when iterating
   - Both maintain their dtype and other properties
   - MLX uses shape (1,) for scalars, while PyTorch uses dimension 0

## Proposed Solutions

### 1. Separate Public and Internal Convert-to-Tensor Functions

Create a clear separation between public and internal tensor conversion functions:

```python
# In ember_ml/nn/tensor/common/__init__.py
# Rename the current function to indicate it's internal and don't export it
_convert_to_backend_tensor = lambda *args, **kwargs: _get_tensor_ops().convert_to_tensor(*args, **kwargs)

# Update __all__ to NOT include _convert_to_backend_tensor
__all__ = [
    # Implementations
    'EmberTensor',
    
    # Operations (excluding _convert_to_backend_tensor)
    'zeros',
    'ones',
    # ... other operations ...
]

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
    
    # Convert to backend tensor first using the internal function
    from ember_ml.nn.tensor.common import _convert_to_backend_tensor
    backend_tensor = _convert_to_backend_tensor(data, dtype=dtype)
    
    # Wrap in EmberTensor
    return EmberTensor(backend_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
```

### 2. Explicit Dtype and Backend Storage

```python
def __init__(
    self,
    data: Optional[Any] = None,
    *,
    dtype: Optional[Union[DType, str, Callable[[], Any]]] = None,
    device: Optional[str] = None,
    requires_grad: bool = False
) -> None:
    # Process dtype
    self._processed_dtype = None
    if dtype is not None:
        if callable(dtype):
            self._processed_dtype = dtype()
        elif isinstance(dtype, str):
            self._processed_dtype = dtype
        else:
            self._processed_dtype = dtype
    
    # Create tensor with processed dtype
    # Use the internal _convert_to_backend_tensor instead of convert_to_tensor
    # to avoid creating an EmberTensor inside the EmberTensor constructor
    self._tensor = _convert_to_backend_tensor(data, dtype=self._processed_dtype)
    
    # Store the dtype explicitly
    backend_dtype = dtype(self._tensor)
    if isinstance(backend_dtype, DType):
        self._dtype = backend_dtype
    else:
        dtype_name = str(backend_dtype).split('.')[-1]
        self._dtype = DType(dtype_name)
    
    # Store device and requires_grad
    from ember_ml.backend import get_device, get_backend
    self._device = device if device is not None else get_device()
    self._requires_grad = requires_grad
    
    # Store backend information
    self._backend = get_backend()
```

### 3. Consistent Property Access

```python
@property
def dtype(self) -> DType:
    """Get the dtype of the tensor."""
    return self._dtype

@property
def backend(self) -> str:
    """Get the backend the tensor was created with."""
    return self._backend
```

### 4. Backend-Agnostic Serialization Support

```python
def __getstate__(self):
    """Get the state for pickling."""
    return {
        'tensor_data': to_numpy(self._tensor),
        'dtype': str(self._dtype),
        'device': self._device,
        'requires_grad': self._requires_grad,
        'backend': self._backend
    }

def __setstate__(self, state):
    """Set the state during unpickling."""
    # Store the state values
    self._dtype = EmberDType(state['dtype'])
    self._device = state['device']
    self._requires_grad = state['requires_grad']
    self._backend = state['backend']
    
    # Recreate the tensor with the correct backend
    from ember_ml.backend import set_backend, get_backend
    current_backend = get_backend()
    try:
        # Temporarily switch to the original backend
        set_backend(self._backend)
        # Create the tensor with the original backend
        from ember_ml.nn.tensor.common import _convert_to_backend_tensor
        self._tensor = _convert_to_backend_tensor(state['tensor_data'], dtype=self._dtype)
    finally:
        # Restore the current backend
        set_backend(current_backend)
```

### 5. Backend-Agnostic Array Interface

```python
def __array__(self) -> Any:
    """Array interface."""
    return self.tolist()
```

This implementation uses the backend-agnostic `tolist()` method instead of converting to NumPy directly, avoiding any NumPy dependencies in the frontend.

### 6. Backend-Agnostic Iteration

```python
def __iter__(self):
    """
    Make the tensor iterable.
    
    Returns:
        Iterator over the tensor elements, where each element is an EmberTensor
    """
    # Iterate directly over the backend tensor
    for element in self._tensor:
        # Wrap each element in an EmberTensor to maintain backend purity
        yield EmberTensor(element, dtype=self._dtype, device=self.device, requires_grad=self._requires_grad)
```

This implementation:
1. Iterates directly over the backend tensor
2. For each element, wraps it in an EmberTensor
3. Preserves the dtype, device, and requires_grad properties
4. Returns EmberTensor objects, not raw backend tensors or NumPy arrays

This approach follows the pattern demonstrated in both PyTorch and MLX, where:
- PyTorch returns torch.Tensor objects when iterating
- MLX returns mlx.core.array objects when iterating
- Both maintain their dtype and other properties

The implementation aligns with MLX's approach for handling scalars (using shape (1,)) and ensures that iteration works consistently across all backends while maintaining backend purity.

### 7. Preserve Existing Method Structure

The current implementation already has a comprehensive set of methods for tensor operations. We should maintain this structure while ensuring consistent dtype and backend handling:

```python
def zeros(self, shape: Union[int, Sequence[int]], dtype: Optional[DType] = None, device: Optional[str] = None) -> 'EmberTensor':
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    # Use the existing zeros function from the common module
    tensor = zeros(shape, dtype=dtype if dtype is not None else self._dtype)
    # Create a new EmberTensor with consistent properties
    return EmberTensor(
        tensor, 
        dtype=dtype if dtype is not None else self._dtype,
        device=device if device is not None else self._device,
        requires_grad=self._requires_grad
    )
```

## Implementation Strategy

1. **Phase 1: Core Changes**
   - Separate public and internal convert_to_tensor functions
   - Ensure _convert_to_backend_tensor is not exported in __all__
   - Add explicit dtype and backend storage in `__init__`
   - Update property accessors
   - Implement serialization methods

2. **Phase 2: Backend Purity Improvements**
   - Update `__array__` to use `tolist()` instead of `to_numpy()`
   - Update `__iter__` to return EmberTensor objects
   - Ensure all methods maintain backend purity

3. **Phase 3: Method Updates**
   - Update existing methods to use the stored dtype and backend
   - Ensure consistent property propagation

4. **Phase 4: Testing**
   - Create comprehensive tests for dtype consistency
   - Test serialization across backends
   - Test iteration behavior with different backends
   - Verify backward compatibility

## Benefits

1. **Improved Consistency**: Explicit storage of dtype and backend ensures consistent behavior
2. **Better Serialization**: Proper serialization methods enable reliable save/load operations
3. **Enhanced Debugging**: Backend tracking makes it easier to diagnose issues
4. **Future Extensibility**: Cleaner architecture allows for easier addition of new features
5. **Clear API Boundaries**: Separation of public and internal functions improves code clarity
6. **Backend Purity**: Avoiding NumPy dependencies in the frontend ensures backend purity

## Immediate Next Steps

1. Implement the separation of public and internal convert_to_tensor functions
2. Update __all__ to not export _convert_to_backend_tensor
3. Add explicit dtype and backend storage in `__init__`
4. Update `__array__` to use `tolist()` instead of `to_numpy()`
5. Update `__iter__` to return EmberTensor objects
6. Add serialization methods (`__getstate__` and `__setstate__`)
7. Update the dtype property to use the stored dtype
8. Add backend property to expose backend information
9. Create tests to verify consistent behavior