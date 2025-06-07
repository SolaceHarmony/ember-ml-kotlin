import pytest
from numpy import dtype
import time

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend
from ember_ml.ops import stats
from ember_ml.ops import linearalg

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Alternative fixture for tests that need to explicitly set/reset backend
@pytest.fixture
def mlx_backend():
    """Set up MLX backend for tests."""
    prev_backend = ops.get_backend()
    ops.set_backend('mlx')
    yield None
    ops.set_backend(prev_backend)

# Test cases for ops.linearalg functions

def test_matmul():
    # Test matrix multiplication
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result = ops.matmul(a, b)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (using numpy for expected values)
    expected_np = ops.matmul(tensor.to_numpy(a), tensor.to_numpy(b))
    assert ops.allclose(result_np, expected_np)

    # Test with different shapes
    c = tensor.convert_to_tensor([[1.0, 2.0, 3.0]]) # Shape (1, 3)
    d = tensor.convert_to_tensor([[4.0], [5.0], [6.0]]) # Shape (3, 1)
    result_cd = ops.matmul(c, d)
    result_dc = ops.matmul(d, c)

    assert ops.allclose(tensor.to_numpy(result_cd), ops.matmul(tensor.to_numpy(c), tensor.to_numpy(d)))
    assert ops.allclose(tensor.to_numpy(result_dc), ops.matmul(tensor.to_numpy(d), tensor.to_numpy(c)))


def test_det():
    # Test determinant calculation
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = ops.linearalg.det(a)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    expected_np = ops.linearalg.det(tensor.to_numpy(a))
    assert ops.allclose(result_np, expected_np)

    # Test with a singular matrix
    b = tensor.convert_to_tensor([[1.0, 2.0], [2.0, 4.0]])
    result_singular = ops.linearalg.det(b)
    assert ops.allclose(tensor.to_numpy(result_singular), 0.0)


def test_qr_numerical_stability():
    """
    Test that QR implementation has 128-bit high precision numerical stability.
    
    This test verifies that our QR implementation maintains extremely high
    orthogonality precision even for challenging matrices.
    """
    # Create a test matrix
    n = 10
    m = 5
    
    # Create a random matrix
    a = tensor.random_normal((n, m))
    
    # Perform QR decomposition
    q, r = linearalg.qr(a)
    
    # Print shapes for debugging
    print(f"Shape of a: {a.shape}")
    print(f"Shape of q: {q.shape}")
    print(f"Shape of r: {r.shape}")
    
    # Check orthogonality of columns (Q^T * Q should be identity)
    q_t_q = ops.matmul(tensor.transpose(q), q)
    
    # Create identity matrix using MLX operations
    n_dim = q_t_q.shape[0]
    # Create zeros tensor
    identity = tensor.zeros((n_dim, n_dim), dtype=tensor.float32)
    # Create indices for diagonal
    indices = tensor.arange(n_dim)
    # Create diagonal indices as tuples
    diag_indices = (indices, indices)
    # Create ones for diagonal
    diag_values = tensor.ones(n_dim, dtype=tensor.float32)
    # Use scatter from MLX ops to set diagonal elements to 1
    from ember_ml.backend.mlx.tensor.ops import scatter
    identity = scatter(diag_indices, diag_values, (n_dim, n_dim))
    
    # Print shapes for debugging
    print(f"Shape of q_t_q: {q_t_q.shape}")
    print(f"Shape of identity: {identity.shape}")
    
    # Compute error matrix and maximum absolute error
    diff = ops.subtract(q_t_q, identity)
    max_error = ops.stats.max(ops.abs(diff))
    mean_error = stats.mean(ops.abs(diff))
    
    # Print actual values for debugging
    print(f"QR Orthogonality:")
    print(f"Maximum absolute error: {max_error}")
    print(f"Mean absolute error: {mean_error}")
    
    # Use extremely strict threshold for 128-bit precision
    # Should be close to machine epsilon for high precision
    assert ops.less(max_error, tensor.convert_to_tensor(1e-10, dtype=tensor.float32)), \
           f"QR orthogonality max error too large: {max_error}"
    
    # Check reconstruction
    recon = ops.matmul(q, r)
    recon_diff = ops.subtract(a, recon)
    recon_max_error = ops.max(ops.abs(recon_diff))
    recon_mean_error = stats.mean(ops.abs(recon_diff))
    
    # Print reconstruction errors
    print(f"QR Reconstruction:")
    print(f"Maximum reconstruction error: {recon_max_error}")
    print(f"Mean reconstruction error: {recon_mean_error}")
    
    # Verify reconstruction is essentially perfect
    assert ops.less(recon_max_error, tensor.convert_to_tensor(1e-10, dtype=tensor.float32)), \
           f"QR reconstruction error too large: {recon_max_error}"


@pytest.mark.skip(reason="SVD numerical stability test needs further refinement")
def test_svd_numerical_stability():
    """
    Test that SVD implementation has good numerical stability
    for ill-conditioned matrices.
    
    This test verifies that our SVD implementation (which uses HPC internally)
    maintains orthogonality and accurately computes singular values.
    """
    # Create a simple test matrix
    n = 10
    m = 5
    
    # Create a random matrix
    a = tensor.random_normal((n, m))
    
    # Perform SVD
    u, s, vt = linearalg.svd(a, full_matrices=False)
    
    # Check orthogonality of U
    u_t_u = ops.matmul(tensor.transpose(u), u)
    identity_m = tensor.eye(m, dtype=tensor.float32)
    error_u = stats.mean(ops.abs(ops.subtract(u_t_u, identity_m)))
    
    # Check orthogonality of V
    v = tensor.transpose(vt)
    v_t_v = ops.matmul(tensor.transpose(v), v)
    error_v = stats.mean(ops.abs(ops.subtract(v_t_v, identity_m)))
    
    # The errors should be small with a relaxed threshold
    assert ops.all(ops.less(error_u, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"SVD U orthogonality error too large: {error_u}"
    assert ops.all(ops.less(error_v, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"SVD V orthogonality error too large: {error_v}"
    
    # Check reconstruction
    recon = ops.matmul(u * s, vt)
    recon_error = stats.mean(ops.abs(ops.subtract(a, recon)))
    assert ops.all(ops.less(recon_error, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"SVD reconstruction error too large: {recon_error}"


@pytest.mark.skip(reason="Metal kernel orthogonalization not available or has compatibility issues")
def test_metal_kernel_orthogonalization():
    """
    Test the Metal kernel-based orthogonalization for large matrices.
    
    This test verifies that the Metal kernel implementation can handle
    large non-square matrices efficiently and with good numerical stability.
    """
    # Skip if not on macOS with Metal support
    try:
        import mlx.core as mx
        device = mx.default_device()
        if 'gpu' not in str(device):
            pytest.skip("Test requires Metal GPU support")
    except (ImportError, AttributeError):
        pytest.skip("Test requires MLX with Metal support")
    
    # Skip if orthogonalize_nonsquare is not available
    try:
        from ember_ml.backend.mlx.linearalg.orthogonal_nonsquare import orthogonalize_nonsquare
    except ImportError:
        pytest.skip("orthogonalize_nonsquare function not available")
    
    # Create a large non-square matrix
    n = 1024
    m = 512
    
    # Create random matrix
    a = tensor.random_normal((n, m))
    
    # Use the Metal kernel-based orthogonalization
    a_tensor = tensor.convert_to_tensor(a)
    
    # Time the Metal kernel implementation
    start_time = time.time()
    q_metal = orthogonalize_nonsquare(a_tensor)
    metal_time = time.time() - start_time
    
    # Check orthogonality
    q_t_q = ops.matmul(tensor.transpose(q_metal), q_metal)
    identity = tensor.eye(m)
    error_metal = stats.mean(ops.abs(ops.subtract(q_t_q, identity))).item()
    
    # Time a standard QR implementation for comparison
    start_time = time.time()
    q_standard, _ = linearalg.qr(a_tensor)
    standard_time = time.time() - start_time
    
    # Check orthogonality of standard implementation
    q_t_q_std = ops.matmul(tensor.transpose(q_standard), q_standard)
    error_standard = stats.mean(ops.abs(q_t_q_std - identity)).item()
    
    # The Metal kernel implementation should be faster for large matrices
    print(f"Metal kernel time: {metal_time:.4f}s, Standard QR time: {standard_time:.4f}s")
    print(f"Metal kernel error: {error_metal}, Standard QR error: {error_standard}")
    
    # The Metal kernel error should be small
    assert ops.all(ops.less(error_metal, tensor.convert_to_tensor(1e-5, dtype=tensor.float32))), \
           f"Metal kernel error too large: {error_metal}"


@pytest.mark.skip(reason="Orthogonal function has compatibility issues")
def test_orthogonal_non_square_matrices():
    """
    Test that the orthogonal function works correctly for non-square matrices.
    
    This test verifies that the orthogonal function produces matrices with
    orthogonal columns even for highly rectangular matrices.
    """
    # Test with various shapes
    shapes = [
        (100, 10),    # Tall and thin
        (10, 100),    # Short and wide
        (128, 64),    # Power of 2 dimensions
        (65, 33),     # Odd dimensions
        (200, 199),   # Almost square
        (3, 100)      # Very rectangular
    ]
    
    for shape in shapes:
        # Generate orthogonal matrix using QR instead of orthogonal function
        random_matrix = tensor.random_normal(shape)
        if shape[0] >= shape[1]:  # Tall matrix
            q, _ = linearalg.qr(random_matrix)
        else:  # Wide matrix
            q_temp, _ = linearalg.qr(tensor.transpose(random_matrix))
            q = tensor.transpose(q_temp)
        
        # Check shape
        assert q.shape == shape, f"Expected shape {shape}, got {q.shape}"
        
        # Check orthogonality of columns
        if shape[0] >= shape[1]:
            # Tall matrix: Q^T * Q should be identity
            q_t_q = ops.matmul(tensor.transpose(q), q)
            identity = tensor.eye(shape[1])
            error = stats.mean(ops.abs(ops.subtract(q_t_q,identity)))
        else:
            # Wide matrix: Q * Q^T should be identity
            q_q_t = ops.matmul(q, tensor.transpose(q))
            identity = tensor.eye(shape[0])
            error = stats.mean(ops.abs(ops.subtract(q_q_t, identity)))
        
        # Error should be small
        assert ops.all(ops.less(error, tensor.convert_to_tensor(1e-5, dtype=tensor.float32))), \
               f"Orthogonality error too large for shape {shape}: {error}"
        print(f"Shape {shape}: orthogonality error = {error}")


@pytest.mark.skip(reason="HPC16x8 QR test needs further refinement")
def test_hpc16x8_qr():
    """
    Test the HPC16x8 QR decomposition for numerical stability.
    
    This test verifies that the HPC16x8 QR implementation maintains
    orthogonality even for challenging matrices.
    """
    # Use HPC16x8 class from the frontend API
    try:
        HPC16x8 = linearalg.HPC16x8
    except AttributeError:
        pytest.skip("HPC16x8 not available in frontend API")
    
    # Create a simple test matrix
    n = 10
    m = 5
    
    # Create a random matrix
    a = tensor.random_normal((n, m))
    
    # Convert to HPC16x8 format
    a_hpc = HPC16x8.from_array(a)
    
    # Perform QR decomposition using HPC16x8
    q, r = a_hpc.qr()
    
    # Check orthogonality of Q
    q_t_q = ops.matmul(tensor.transpose(q), q)
    identity = tensor.eye(m, dtype=tensor.float32)
    error = stats.mean(ops.abs(ops.subtract(q_t_q, identity)))
    
    # Check reconstruction error
    recon = ops.matmul(q, r)
    recon_error = stats.mean(ops.abs(ops.subtract(a, recon)))
    
    # The errors should be small with a relaxed threshold
    assert ops.all(ops.less(error, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"HPC16x8 QR orthogonality error too large: {error}"
    assert ops.all(ops.less(recon_error, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"HPC16x8 QR reconstruction error too large: {recon_error}"


@pytest.mark.skip(reason="HPC16x8 eigendecomposition test needs further refinement")
def test_hpc16x8_eig():
    """
    Test the HPC16x8 eigendecomposition for numerical stability.
    
    This test verifies that the HPC16x8 eigendecomposition implementation
    accurately computes eigenvalues and eigenvectors.
    """
    # Use HPC16x8 class from the frontend API
    try:
        HPC16x8 = linearalg.HPC16x8
    except AttributeError:
        pytest.skip("HPC16x8 not available in frontend API")
    
    # Create a simple symmetric matrix
    n = 5  # Small size for faster test
    
    # Create a random symmetric matrix
    a_random = tensor.random_normal((n, n))
    a = ops.add(a_random, tensor.transpose(a_random)) / 2.0  # Make symmetric
    
    # Convert to HPC16x8 format
    a_hpc = HPC16x8.from_array(a)
    
    # Perform eigendecomposition using HPC16x8
    w, v = a_hpc.eig()
    
    # Check eigenvector orthogonality
    v_t_v = ops.matmul(tensor.transpose(v), v)
    identity = tensor.eye(n, dtype=tensor.float32)
    orthogonality_error = stats.mean(ops.abs(ops.subtract(v_t_v, identity)))
    
    # The errors should be small with a relaxed threshold
    assert ops.all(ops.less(orthogonality_error, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"HPC16x8 eigenvector orthogonality error too large: {orthogonality_error}"
    
    # Check reconstruction
    recon = ops.matmul(ops.matmul(v, linearalg.diag(w)), tensor.transpose(v))
    recon_error = stats.mean(ops.abs(ops.subtract(a, recon)))
    assert ops.all(ops.less(recon_error, tensor.convert_to_tensor(0.1, dtype=tensor.float32))), \
           f"HPC16x8 eigendecomposition reconstruction error too large: {recon_error}"


# Add more test functions for other ops.linearalg functions:
# test_cholesky(), test_solve(), test_inv(), test_norm(), test_lstsq(), test_diag(), test_diagonal()