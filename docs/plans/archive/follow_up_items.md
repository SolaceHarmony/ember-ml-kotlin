# Follow-Up Items

This document tracks discrepancies between the documentation and the actual implementation, as well as items that need further investigation or implementation.

## 1. Dynamic DType Properties Implementation

**Issue:** The dynamic dtype properties implementation described in `docs/plans/compatibility_plans/compatibility_plan.md` hasn't been fully implemented yet.

**Details:**
- The documentation describes adding a `get_available_dtypes()` method to each backend's DType class
- Our spot check of `ember_ml/backend/torch/tensor/dtype.py` shows that this method is not yet implemented
- This suggests that the dynamic dtype properties implementation is still a plan rather than an existing feature

**Follow-up Actions:**
- Implement the `get_available_dtypes()` method in each backend's DType class
- Update the dtypes module to dynamically create properties based on the available dtypes
- Update the tensor module to import and re-export these properties
- Add tests to verify the implementation

## 2. Function-First Approach Implementation

**Issue:** We need to verify that the tensor operations implementation follows the function-first approach described in the documentation.

**Details:**
- The documentation describes implementing each operation as a standalone function first, then adding method wrappers
- We need to verify that the actual implementation follows this approach

**Follow-up Actions:**
- Check the implementation of tensor operations in each backend
- Verify that operations are implemented as standalone functions
- Verify that methods in the tensor class are thin wrappers around these functions

## 3. Wiring Module Fixes

**Issue:** The wiring module fixes described in `docs/plans/compatibility_plans/compatibility_plan.md` need to be verified.

**Details:**
- The documentation describes updating the wiring module to use the new tensor API correctly
- We need to verify that these fixes have been implemented

**Follow-up Actions:**
- Check the implementation of the wiring module
- Verify that it uses the correct module paths for imports
- Verify that it uses module-level functions instead of static methods
- Verify that it doesn't reference the non-existent `data` property
- Verify that it uses dtype objects directly instead of strings

## 4. Frontend Compatibility

**Issue:** The frontend compatibility with the refactored backend needs to be verified.

**Details:**
- The documentation describes ensuring that the frontend tensor implementation remains compatible with the refactored backend
- We need to verify that this compatibility has been maintained

**Follow-up Actions:**
- Check the implementation of the frontend tensor implementation
- Verify that it works correctly with the refactored backend
- Verify that both function and method calling patterns are supported

## 5. API Cleanup

**Issue:** The API cleanup described in `docs/plans/implementation_plans/implementation_plan.md` needs to be verified.

**Details:**
- The documentation describes removing tensor-related imports from `ember_ml.ops/__init__.py`
- We need to verify that this cleanup has been performed

**Follow-up Actions:**
- Check the implementation of `ember_ml.ops/__init__.py`
- Verify that tensor-related imports have been removed
- Verify that tensor operations are only exposed through `ember_ml.nn.tensor`