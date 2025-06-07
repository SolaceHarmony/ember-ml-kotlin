# ember_ml Tests

This directory contains tests for the ember_ml library. The tests are organized by functionality and are designed to ensure that the library works correctly across different backends. All tests have been converted to use pytest for better reporting and organization.

## Test Organization

The tests are organized into the following files:

- `test_plan.md`: A comprehensive test plan that outlines the testing strategy for the ember_ml library.
- `test_ops_tensor.py`: Tests for tensor operations across different backends.
- `test_ops_math.py`: Tests for math operations across different backends.
- `test_backend.py`: Tests for backend selection and switching functionality.
- `test_ncp.py`: Tests for Neural Circuit Policy (NCP) functionality.
- `test_ember_tensor.py`: Tests for the EmberTensor class.
- `test_ops_solver.py`: Tests for solver operations.
- `test_backend_auto_selection.py`: Tests for automatic backend selection.
- `test_detect_numpy_usage.py`: Tests for the NumPy usage detection tool.
- `test_compare_random_ops_purified.py`: Tests for purified random operations.
- `test_math_ops_comparison.py`: Tests for comparing math operations across backends.
- `test_pi_values.py`: Tests for pi values across backends.
- `test_terabyte_feature_extractor_purified.py` and `test_terabyte_feature_extractor_purified_v2.py`: Tests for the purified TerabyteFeatureExtractor.

## Running Tests

You can run the tests using the enhanced `run_tests.py` script:

```bash
# Run all tests
python run_tests.py

# Run specific test modules
python run_tests.py tensor    # Run all tensor tests
python run_tests.py math      # Run all math tests
python run_tests.py backend   # Run all backend tests

# Run specific test classes
python run_tests.py tensor.TestTensorCreation
python run_tests.py math.TestBasicArithmetic
python run_tests.py backend.TestBackendSelection

# Run specific test methods
python run_tests.py tensor.TestTensorCreation.test_zeros
python run_tests.py math.TestBasicArithmetic.test_add
python run_tests.py backend.TestBackendSelection.test_default_backend

# Generate HTML report
python run_tests.py --report

# Run with coverage
python run_tests.py --cov

# Run with verbose output
python run_tests.py --verbose
```

Alternatively, you can use pytest directly:

```bash
# Run all tests
pytest

# Run specific test modules
pytest test_ops_tensor.py
pytest test_ops_math.py
pytest test_backend.py

# Run specific test classes
pytest test_ops_tensor.py::TestTensorCreation
pytest test_ops_math.py::TestBasicArithmetic
pytest test_backend.py::TestBackendSelection

# Run specific test methods
pytest test_ops_tensor.py::TestTensorCreation::test_zeros
pytest test_ops_math.py::TestBasicArithmetic::test_add
pytest test_backend.py::TestBackendSelection::test_default_backend

# Generate HTML report
pytest --html=report.html --self-contained-html

# Run with coverage
pytest --cov=ember_ml --cov-report=html

# Run with verbose output
pytest -v
```

## Test Configuration

The tests are configured using the following files:

- `pytest.ini`: Contains pytest configuration settings.
- `conftest.py`: Contains common fixtures and configurations for pytest.

## Test Fixtures

The tests use pytest fixtures to set up the test environment. The main fixtures are:

### Backend Fixtures

```python
@pytest.fixture
def numpy_backend():
    """Set NumPy backend for tests."""
    original_backend = get_backend()
    set_backend('numpy')
    yield
    set_backend(original_backend)

@pytest.fixture
def torch_backend():
    """Set PyTorch backend for tests if available."""
    try:
        import torch
        original_backend = get_backend()
        set_backend('torch')
        yield
        set_backend(original_backend)
    except ImportError:
        pytest.skip("PyTorch not available")

@pytest.fixture
def mlx_backend():
    """Set MLX backend for tests if available."""
    try:
        import mlx.core
        original_backend = get_backend()
        set_backend('mlx')
        yield
        set_backend(original_backend)
    except ImportError:
        pytest.skip("MLX not available")

@pytest.fixture(params=['numpy', 'torch', 'mlx'])
def any_backend(request):
    """Parametrize tests with all available backends."""
    backend_name = request.param
    
    # Skip if backend is not available
    if backend_name == 'torch':
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    elif backend_name == 'mlx':
        try:
            import mlx.core
        except ImportError:
            pytest.skip("MLX not available")
    
    # Set backend
    original_backend = get_backend()
    set_backend(backend_name)
    
    yield backend_name
    
    # Restore original backend
    set_backend(original_backend)
```

### Tensor Fixtures

```python
@pytest.fixture
def sample_tensor_1d():
    """Create a 1D sample tensor."""
    return tensor.convert_to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def sample_tensor_2d():
    """Create a 2D sample tensor."""
    return tensor.convert_to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

@pytest.fixture
def sample_tensor_3d():
    """Create a 3D sample tensor."""
    return tensor.convert_to_tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])
```

## Backend Compatibility

The tests are designed to work with all available backends. The list of backends to test is determined dynamically based on the available libraries:

```python
@pytest.fixture(scope="session")
def available_backends():
    """Get list of available backends."""
    backends = ['numpy']
    
    try:
        import torch
        backends.append('torch')
    except ImportError:
        pass
    
    try:
        import mlx.core
        backends.append('mlx')
    except ImportError:
        pass
    
    return backends
```

This ensures that the tests will run even if some backends are not available.

## Test Coverage

The tests cover the following areas:

1. **Backend Selection and Switching**: Tests for backend selection, switching, and persistence.
2. **Tensor Operations**: Tests for tensor creation, manipulation, and information operations.
3. **Math Operations**: Tests for basic arithmetic, reduction, element-wise, and activation functions.
4. **Backend Compatibility**: Tests for tensor conversion and operation consistency across backends.
5. **NCP Functionality**: Tests for Neural Circuit Policy (NCP) and AutoNCP classes.
6. **Solver Operations**: Tests for solver operations like matrix inversion, determinant, etc.
7. **Feature Extraction**: Tests for feature extraction components.
8. **Code Quality**: Tests for detecting NumPy usage and ensuring backend purity.

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Test Organization**: Add tests to the appropriate file based on functionality.
2. **Backend Compatibility**: Use the backend fixtures to test with different backends.
3. **Test Coverage**: Ensure that all operations and edge cases are covered.
4. **Test Documentation**: Add docstrings to test classes and methods to explain what is being tested.
5. **Test Independence**: Each test should be independent and not rely on the state of other tests.
6. **Fixtures**: Use fixtures for common setup and teardown operations.
7. **Parametrization**: Use pytest's parametrization for testing multiple cases.

## Test Plan

For a comprehensive testing strategy, refer to the `test_plan.md` file, which outlines the testing approach for the ember_ml library.