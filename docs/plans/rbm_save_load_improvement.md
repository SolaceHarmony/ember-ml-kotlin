# RBM Module Save/Load Functionality Improvement Plan

## Current Status

During the process of moving tests from the main project folder to the tests/ folder and updating them to meet the project's testing standards, we encountered an issue with the save/load functionality for RBM modules.

The current implementation of `save_rbm` and `load_rbm` functions in `ember_ml/models/rbm/training.py` uses `ops.save` and `ops.load`, which in turn use NumPy's `save` and `load` functions. While NumPy's `save` function can save arbitrary Python objects using pickle, it doesn't preserve the class structure in a way that would allow for proper reconstruction of the `RBMModule` class.

Specifically, when loading an RBM module, the loaded object is a NumPy array, not an `RBMModule` instance, resulting in the following error:

```
AttributeError: 'numpy.ndarray' object has no attribute 'weights'
```

This issue was also present in the original `test_rbm_module.py` file, which skipped the save/load test with the comment "Skipping save/load test due to missing IO ops in NumPy backend".

## Proposed Improvement

A more robust approach to saving and loading RBM modules would be:

1. Save just the tensor data (weights, biases, etc.) to a file
2. When loading, create a new `RBMModule` instance with the same parameters as the original
3. Load the tensor data from the file and assign it to the new instance

This approach is commonly used in ML frameworks like PyTorch and TensorFlow, where model files typically store just the tensor data rather than the entire class instance with its methods.

## Implementation Plan

1. Modify the `save_rbm` function to save the RBM module's tensor data (weights, biases, etc.) to a file, along with any necessary metadata (e.g., number of visible and hidden units, learning rate, etc.)
2. Modify the `load_rbm` function to:
   - Load the tensor data and metadata from the file
   - Create a new `RBMModule` instance with the same parameters as the original
   - Assign the loaded tensor data to the new instance
3. Update the tests to verify that the save/load functionality works correctly

## Benefits

- More robust save/load functionality that works across different backends
- Better compatibility with standard ML practices
- Improved testability of the RBM module

## Timeline

This improvement should be prioritized as it affects the usability of the RBM module in production environments where model persistence is important.

## Related Issues

- The `NumPyIOOps` class in `ember_ml/backend/numpy/io_ops.py` is named with a capital 'P', while the code in `ember_ml/ops/__init__.py` looks for `NumpyIOOps` with a lowercase 'p'. This was fixed by adding a special case in the `_get_ops_instance` function, but a more consistent naming convention should be established.