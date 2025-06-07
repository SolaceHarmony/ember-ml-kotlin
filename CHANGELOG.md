# Ember ML Changelog

## Initial Setup and Major Changes

- **2025-03-08** (2daa231) - Initial commit
- **2025-03-08** (ae3cf88) - Add .gitignore file to exclude build artifacts, environment files, and IDE-specific files
- **2025-03-08** (07eebe7) - Massive initial check-in with framework foundation, including:
  - Core architecture for multi-backend ML operations
  - Support for NumPy, PyTorch, and MLX backends
  - Neural network modules (RNN, LSTM, GRU, CFC, LTC)
  - Feature extraction components
  - Attention mechanisms
  - Restricted Boltzmann Machines (RBM) implementation
  - Wave-based neural components
  - Comprehensive documentation and examples
- **2025-03-08** (c5d42e7) - Remove Notebook simulation content from README
- **2025-03-10** (3c481da) - Refactor references from emberharmony to ember_ml across the codebase

## Backend Framework and Tensor Operations

- **2025-03-10** (96e33c3) - Refactor dtype operations to use string representations and update memory usage functions
- **2025-03-11** (997eeb8) - Add Ember backend configuration, implement negative function, and update tensor operations
- **2025-03-12** (bd9204a) - Add cumulative sum and eigenvalue functions to math operations, and implement sort functionality in NumPy backend
- **2025-03-12** (2b73f0f) - Merge PR #1: Fix backend purity issues
- **2025-03-12** (3c8123d) - Fix backend purity issues:
  - Add missing functions (cumsum, eigh) to ops/__init__.py and backend implementations
  - Add proper type annotations and docstrings to all implementations
  - Ensure consistent implementation across all backends
- **2025-03-16** (8785413) - Refactor backend configurations and tensor interfaces; update device settings and add new test cases
- **2025-03-18** (8e450d7) - Add pi constant to math operations, update tensor slicing imports, and enhance dtype validation in MLX
- **2025-03-18** (9a61269) - Refactor import statements to replace deprecated config module with types module across multiple files
- **2025-03-18** (f32d8be) - Refactor tensor imports to use MLXTensor consistently across tensor operation modules
- **2025-03-18** (84bad18) - Rename tensor conversion function to convert_to_mlx_tensor for consistency across modules
- **2025-03-18** (761f9e9) - Implement NumPy linear algebra operations and enhance tensor indexing functionality
- **2025-03-18** (a3f9859) - Refactor tensor conversion calls to use the tensor module consistently across the codebase:
  - Implement linear algebra operations for PyTorch backend
  - Enhance tensor indexing and operations
  - Reorganize tensor utility functions
  - Improve tensor creation and manipulation functionality
- **2025-03-19** (d0faf57) - Refactor tensor slicing and dtype retrieval for improved compatibility and clarity
- **2025-03-19** (eee0744) - Refactor tensor operations and enhance error handling in Parameter representation
- **2025-03-19** (95302c8) - Add implementation and audit plans for MLX backend typing improvements

## Neural Network Modules and Functionality

- **2025-03-10** (ef275ca) - Enhance neural network module with new container implementations and abstract methods for device operations
- **2025-03-10** (85375a9) - Add activation function interfaces and implementations for neural networks
- **2025-03-10** (4c70e3a) - Refactor test_auto_ncp to support multiple backends and improve parameter output
- **2025-03-10** (da91fb3) - Remove deprecated Keras layer implementations and associated documentation
- **2025-03-18** (25c6372) - Add test packages for legacy modules and implement I/O operations interface - fix EmberTensor issues
- **2025-03-19** (1c02757) - Refactor and consolidate backend tests for EmberTensor operations

## Documentation and Code Quality

- **2025-03-18** (9cefc98) - Merge PR #6: Add compatibility tests for PyTorch backend and update documentation:
  - Update README and project documentation
  - Add documentation for backend compatibility
  - Create plans for sigmoid function consolidation
  - Improve API documentation and examples
- **2025-03-18** (06cd016) - Merge branch 'main' into cleanup_docs
- **2025-03-18** (eedcb8a) - Add compatibility tests for PyTorch backend and update documentation

## Testing Improvements

- **2025-03-10** (9c711cc) - Add new tests for NCP and StrideAwareCell modules, and remove outdated test files
- **2025-03-18** (ab08ff4) - Merge PR #4: Tensor improvements and cleanup
- **2025-03-12** (2738e87) - Merge PR #2: Fix backend purity

## Release Notes

### Major Improvements

1. **Backend Abstraction Layer**
   - Complete refactoring of tensor operations across NumPy, PyTorch, and MLX backends
   - Implementation of consistent API across all backends
   - Enhanced type safety and error handling

2. **Neural Network Modules**
   - Implementation of various RNN cell types (LSTM, GRU, CFC, LTC)
   - Support for Neural Circuit Policies (NCP)
   - Attention mechanisms for sequence modeling

3. **Feature Extraction**
   - Column-based feature extraction for tabular data
   - Temporal stride processing for time series

### Breaking Changes

- API changes in tensor conversion functions for better consistency
- Removal of deprecated Keras layer implementations
- Refactored dtype operations to use string representations

### Known Issues

- Some tensor operations may have different performance characteristics across backends
- Documentation for some newer features is still being improved

### Future Plans

- Further MLX backend typing improvements
- Continued refinement of tensor operations for cross-backend consistency
- Additional tests for edge cases and performance benchmarks