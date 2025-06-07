# Ember ML Backend Switching Fix

This document describes a fix for backend switching issues in the Ember ML framework.

## The Problem

The Ember ML framework had an issue with backend switching where the ops module's cache was not properly cleared when switching backends. This resulted in tensor operations continuing to use the previous backend even after switching to a new one.

For example, if you switched from PyTorch to NumPy using `set_backend('numpy')`, the tensor operations might still return PyTorch tensors instead of NumPy arrays.

## The Solution

The solution was to clear the ops module's cache and reload the ops module when switching backends. This ensures that all tensor operations use the new backend after switching.

The fix was implemented by modifying the `set_backend` function in `ember_ml/backend/__init__.py` to:

1. Set the backend configuration (original functionality)
2. Clear the ops module's cache (`_CURRENT_INSTANCES`)
3. Reload the ops module to ensure it uses the new backend

```python
def set_backend(backend: str):
    """Set the current backend."""
    global _CURRENT_BACKEND, _CURRENT_BACKEND_MODULE
    
    # Original code...
    
    # Clear the ops module's cache and reload it
    try:
        import importlib
        ops_module = importlib.import_module('ember_ml.ops')
        
        # Clear the _CURRENT_INSTANCES cache
        if hasattr(ops_module, '_CURRENT_INSTANCES'):
            ops_module._CURRENT_INSTANCES = {}
        
        # Reload the ops module to ensure it uses the new backend
        importlib.reload(ops_module)
    except Exception as e:
        print(f"Warning: Error updating ops module after backend switch: {e}")
```

## Testing

The fix was tested using pytest in `tests/test_backend_switching.py`. The tests verify that:

1. When switching to NumPy, tensor operations return `numpy.ndarray` objects
2. When switching to PyTorch, tensor operations return `torch.Tensor` objects
3. When switching to MLX, tensor operations return `mlx.core.array` objects

## License

This fix is provided under the same license as the Ember ML framework.