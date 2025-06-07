# Gemini Implementation Plan

This document outlines the steps I will take as a coder to refactor the tensor operations in the Ember ML library, ensuring backend purity and adherence to the project's architectural principles.

## 1. Understanding the Current State

*   **[Already Completed] Review Existing Code:** Thoroughly examine the existing code in `ember_ml/ops/__init__.py`, `ember_ml/nn/tensor/common/ember_tensor.py`, and the backend-specific implementations to understand the current implementation of tensor operations.
*   **[Already Completed] Study `.clinerules-code`:** Carefully review the `.clinerules-code` file to ensure adherence to the project's coding standards and backend purity requirements.
*   **[Already Completed] Analyze Dependencies:** Identify dependencies between the operations to determine the correct implementation order.
*   **[Already Completed] Map NumPy to MLX:** Use the `numpy_to_mlx.py` script and the `numpyconversiondata.csv` file to identify the MLX equivalents of the NumPy functions used in the existing code.

## 2. Backend Implementation

*   **Create Directory Structure:** For each backend (NumPy, PyTorch, MLX), create the following directory structure:
    `ember_ml/backend/{backend_name}/tensor/ops/`
*   **Implement Operations:** Implement the tensor operations in the corresponding files within the directory structure (e.g., `math.py`, `random.py`).
    *   Ensure that the implementations take `EmberTensor` objects (or their underlying backend tensors) as input and return `EmberTensor` objects.
    *   Use the appropriate backend-specific functions (e.g., `np.add`, `torch.add`, `mx.add`).
    *   Handle device placement (CPU, GPU, MPS) using the backend-specific APIs.
    *   Implement gradient computation using the backend-specific APIs for operations that require automatic differentiation.

## 3. Frontend Abstraction

*   **Update `EmberTensor` Class:** Add methods to the `EmberTensor` class in `ember_ml/nn/tensor/common/ember_tensor.py` for each tensor operation.
    *   Delegate to the backend implementations using a mechanism like `_get_tensor_ops()`.
    *   Handle type conversions between `EmberTensor` and the backend-specific tensor types.
    *   Ensure that the operations are performed on the correct device.

## 4. API Cleanup

*   **[To Do] Remove Tensor-Related Entries:** Remove the tensor-related entries from the `__all__` list in `ember_ml/ops/__init__.py`.
*   **[To Do] Add Comment:** Add a comment to `ember_ml/ops/__init__.py` to clearly state that tensor operations are only available through `ember_ml.nn.tensor`.

## 5. Testing

*   **Unit Tests:** Create unit tests for each backend implementation to verify that the operations work correctly.
*   **Integration Tests:** Create integration tests to verify that the frontend `EmberTensor` class correctly delegates to the backend implementations.
*   **Regression Tests:** Run existing tests to ensure that the changes haven't broken any existing functionality.
*   **Performance Tests:** Benchmark the performance of the operations to ensure that there is no significant performance degradation.

## 6. Documentation

*   Update the documentation to reflect the new architecture and the location of the tensor operations.
*   Update any examples or tutorials that use the tensor operations to import from `ember_ml.nn.tensor`.

## 7. EmberLint Validation

*   Run `utils/emberlint.py` on all modified files to ensure code correctness and adherence to style guidelines.

## 8. Code Review

*   Submit the code for review and address any feedback.

## 9. Final Verification

*   Ensure that all tests pass.
*   Verify that the documentation is complete and accurate.
*   Confirm that EmberLint passes with no errors.

This plan will be executed iteratively, with each step carefully reviewed and tested before proceeding to the next.