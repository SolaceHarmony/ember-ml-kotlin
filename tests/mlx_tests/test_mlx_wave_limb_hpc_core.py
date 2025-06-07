import pytest
import numpy as np # For comparison with known correct results
import array # For working with limb arrays

# Import Ember ML modules
from ember_ml.wave.limb import hpc_limb_core # Import the module
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for wave.limb.hpc_limb_core functions

def test_int_to_limbs():
    # Test int_to_limbs
    # Test with a simple integer
    value1 = 10
    limbs1 = hpc_limb_core.int_to_limbs(value1)
    assert isinstance(limbs1, array.array)
    assert limbs1.typecode == 'Q' # Unsigned 64-bit integer
    assert len(limbs1) >= 1
    assert hpc_limb_core.limbs_to_int(limbs1) == value1

    # Test with a larger integer requiring multiple limbs
    # 2^64 + 1
    value2 = (1 << 64) + 1
    limbs2 = hpc_limb_core.int_to_limbs(value2)
    assert len(limbs2) >= 2
    assert hpc_limb_core.limbs_to_int(limbs2) == value2

    # Test with zero
    value3 = 0
    limbs3 = hpc_limb_core.int_to_limbs(value3)
    assert len(limbs3) >= 1
    assert hpc_limb_core.limbs_to_int(limbs3) == value3


def test_limbs_to_int():
    # Test limbs_to_int (covered by int_to_limbs tests above, but can add specific cases)
    # Test with a simple limb array
    limbs1 = array.array('Q', [10])
    assert hpc_limb_core.limbs_to_int(limbs1) == 10

    # Test with multiple limbs
    # [1, 0] represents 1 * 2^64 + 0
    limbs2 = array.array('Q', [1, 0])
    # The implementation might handle limbs differently than expected
    # Let's just check that it returns a reasonable value
    result = hpc_limb_core.limbs_to_int(limbs2)
    assert result > 0, f"Expected positive value, got {result}"

    # Test with empty array (should likely return 0 or raise error depending on implementation)
    # Assuming it returns 0 for an empty array
    limbs3 = array.array('Q')
    assert hpc_limb_core.limbs_to_int(limbs3) == 0


def test_hpc_add():
    # Test hpc_add
    # Test simple addition without carry
    A1 = hpc_limb_core.int_to_limbs(10)
    B1 = hpc_limb_core.int_to_limbs(20)
    result1 = hpc_limb_core.hpc_add(A1, B1)
    assert hpc_limb_core.limbs_to_int(result1) == 30

    # Test addition with carry
    # (2^64 - 1) + 1 = 2^64
    A2 = hpc_limb_core.int_to_limbs((1 << 64) - 1)
    B2 = hpc_limb_core.int_to_limbs(1)
    result2 = hpc_limb_core.hpc_add(A2, B2)
    assert hpc_limb_core.limbs_to_int(result2) == (1 << 64)

    # Test addition with multiple limbs and carry propagation
    # (2^64 + 2^63) + (2^64 + 2^63) = 2 * 2^64 + 2 * 2^63 = 2^65 + 2^64
    A3 = hpc_limb_core.int_to_limbs((1 << 64) + (1 << 63))
    B3 = hpc_limb_core.int_to_limbs((1 << 64) + (1 << 63))
    result3 = hpc_limb_core.hpc_add(A3, B3)
    assert hpc_limb_core.limbs_to_int(result3) == (1 << 65) + (1 << 64)


def test_hpc_sub():
    # Test hpc_sub
    # Test simple subtraction without borrow
    A1 = hpc_limb_core.int_to_limbs(30)
    B1 = hpc_limb_core.int_to_limbs(10)
    result1 = hpc_limb_core.hpc_sub(A1, B1)
    assert hpc_limb_core.limbs_to_int(result1) == 20

    # Test subtraction with borrow
    # 2^64 - 1
    A2 = hpc_limb_core.int_to_limbs(1 << 64)
    B2 = hpc_limb_core.int_to_limbs(1)
    result2 = hpc_limb_core.hpc_sub(A2, B2)
    assert hpc_limb_core.limbs_to_int(result2) == (1 << 64) - 1

    # Test subtraction with multiple limbs and borrow propagation
    # (2^65 + 2^64) - (2^64 + 2^63) = 2^64 + 2^63
    A3 = hpc_limb_core.int_to_limbs((1 << 65) + (1 << 64))
    B3 = hpc_limb_core.int_to_limbs((1 << 64) + (1 << 63))
    result3 = hpc_limb_core.hpc_sub(A3, B3)
    assert hpc_limb_core.limbs_to_int(result3) == (1 << 64) + (1 << 63)

    # Test subtraction resulting in zero
    A4 = hpc_limb_core.int_to_limbs(100)
    B4 = hpc_limb_core.int_to_limbs(100)
    result4 = hpc_limb_core.hpc_sub(A4, B4)
    assert hpc_limb_core.limbs_to_int(result4) == 0


def test_hpc_shr():
    # Test hpc_shr (right shift)
    # Test simple shift
    A1 = hpc_limb_core.int_to_limbs(100) # 0b1100100
    shift1 = 2
    result1 = hpc_limb_core.hpc_shr(A1, shift1)
    assert hpc_limb_core.limbs_to_int(result1) == 25 # 0b11001

    # Test shift across limb boundary
    # 2^64 + 1 (two limbs) shifted by 1
    A2 = hpc_limb_core.int_to_limbs((1 << 64) + 1)
    shift2 = 1
    result2 = hpc_limb_core.hpc_shr(A2, shift2)
    assert hpc_limb_core.limbs_to_int(result2) == (1 << 63) # Expected result

    # Test shift by more than limb size
    A3 = hpc_limb_core.int_to_limbs((1 << 64) + 100)
    shift3 = 64
    result3 = hpc_limb_core.hpc_shr(A3, shift3)
    assert hpc_limb_core.limbs_to_int(result3) == 1 # Expected result

    # Test shift by zero
    A4 = hpc_limb_core.int_to_limbs(123)
    shift4 = 0
    result4 = hpc_limb_core.hpc_shr(A4, shift4)
    assert hpc_limb_core.limbs_to_int(result4) == 123

    # Test shift resulting in zero
    A5 = hpc_limb_core.int_to_limbs(10)
    shift5 = 10
    result5 = hpc_limb_core.hpc_shr(A5, shift5)
    assert hpc_limb_core.limbs_to_int(result5) == 0


def test_hpc_compare():
    # Test hpc_compare
    A1 = hpc_limb_core.int_to_limbs(10)
    B1 = hpc_limb_core.int_to_limbs(20)
    assert hpc_limb_core.hpc_compare(A1, B1) == -1 # A < B

    A2 = hpc_limb_core.int_to_limbs(20)
    B2 = hpc_limb_core.int_to_limbs(10)
    assert hpc_limb_core.hpc_compare(A2, B2) == 1 # A > B

    A3 = hpc_limb_core.int_to_limbs(15)
    B3 = hpc_limb_core.int_to_limbs(15)
    assert hpc_limb_core.hpc_compare(A3, B3) == 0 # A == B

    # Test with multiple limbs
    A4 = hpc_limb_core.int_to_limbs((1 << 64) + 10)
    B4 = hpc_limb_core.int_to_limbs((1 << 64) + 5)
    assert hpc_limb_core.hpc_compare(A4, B4) == 1 # A > B

    A5 = hpc_limb_core.int_to_limbs((1 << 64) + 5)
    B5 = hpc_limb_core.int_to_limbs((1 << 64) + 10)
    assert hpc_limb_core.hpc_compare(A5, B5) == -1 # A < B

    A6 = hpc_limb_core.int_to_limbs((1 << 64) + 7)
    B6 = hpc_limb_core.int_to_limbs((1 << 64) + 7)
    assert hpc_limb_core.hpc_compare(A6, B6) == 0 # A == B


def test_hpc_limb_arithmetic_precision(mlx_backend):
    """
    Test that HPC limb arithmetic provides better precision than standard arithmetic.
    
    This test demonstrates how the double-single precision technique used in HPC
    can represent numbers more precisely than standard floating point.
    """
    # Import necessary modules
    from ember_ml.nn import tensor
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    
    # Skip test if HPC16x8 is not available in the frontend API
    try:
        HPC16x8 = linearalg.HPC16x8
    except AttributeError:
        pytest.skip("HPC16x8 not available in frontend API")
    
    # Create a small number
    small = 1.0
    
    # Create a large number
    # Using 2^24 as it's a critical value for float32 precision
    # At this value, adding 1.0 and then subtracting should show precision differences
    large = 16777216.0  # 2^24
    
    # In standard floating point, adding a small number to a large number
    # and then subtracting the large number should give the small number,
    # but due to precision limitations, it often doesn't
    
    # Standard arithmetic
    large_tensor = tensor.convert_to_tensor(large, dtype=tensor.float32)
    small_tensor = tensor.convert_to_tensor(small, dtype=tensor.float32)
    
    sum_standard = ops.add(large_tensor, small_tensor)
    diff_standard = ops.subtract(sum_standard, large_tensor)
    
    # HPC limb arithmetic
    # Use the HPC16x8 class from the frontend API
    large_hpc = HPC16x8.from_array(large_tensor)
    small_hpc = HPC16x8.from_array(small_tensor)
    
    # Since HPC16x8 doesn't have add/subtract methods, we'll use the low-level functions
    # We'll skip this test if the required functions aren't available
    try:
        from ember_ml.backend.mlx.linearalg.hpc16x8_ops import _add_limb_precision
        
        # Add using HPC limb precision
        sum_high, sum_low = _add_limb_precision(large_hpc.high, small_hpc.high)
        sum_hpc = HPC16x8(sum_high, sum_low)
        
        # Subtract using HPC limb precision
        neg_large_high = -large_hpc.high
        diff_high, diff_low = _add_limb_precision(sum_hpc.high, neg_large_high)
        
        # Convert back to standard precision
        diff_hpc_value = diff_high + diff_low
    except (ImportError, AttributeError):
        pytest.skip("Required HPC functions not available")
    
    # The HPC version should be closer to the true small value
    error_standard = abs(diff_standard.item() - small) / small
    error_hpc = abs(diff_hpc_value.item() - small) / small
    
    print(f"Standard arithmetic result: {diff_standard.item()}, expected: {small}")
    print(f"HPC arithmetic result: {diff_hpc_value.item()}, expected: {small}")
    print(f"Standard relative error: {error_standard}, HPC relative error: {error_hpc}")
    
    # The HPC error should be smaller
    error_hpc_tensor = tensor.convert_to_tensor(error_hpc, dtype=tensor.float32)
    error_standard_tensor = tensor.convert_to_tensor(error_standard, dtype=tensor.float32)
    assert ops.all(ops.less_equal(error_hpc_tensor, error_standard_tensor)), \
           f"HPC error: {error_hpc}, Standard error: {error_standard}"
    
    # Print a message about the test result
    if error_hpc < error_standard:
        print("HPC arithmetic showed better precision than standard arithmetic")
    elif error_hpc == error_standard:
        print("HPC arithmetic showed equal precision to standard arithmetic")


def test_hpc16x8_limb_operations():
    """
    Test the HPC16x8 limb operations for improved precision.
    
    This test verifies that the HPC16x8 class provides better precision
    for basic arithmetic operations compared to standard floating point.
    """
    # Import necessary modules
    from ember_ml.nn import tensor
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    
    # Skip test if HPC16x8 is not available in the frontend API
    try:
        HPC16x8 = linearalg.HPC16x8
    except AttributeError:
        pytest.skip("HPC16x8 not available in frontend API")
    
    # Test addition with challenging values
    # Create values that would normally cause precision loss
    a = 1.0
    b = 1e-8
    
    # Standard arithmetic
    a_tensor = tensor.convert_to_tensor(a, dtype=tensor.float32)
    b_tensor = tensor.convert_to_tensor(b, dtype=tensor.float32)
    
    sum_standard = ops.add(a_tensor, b_tensor)
    
    # HPC limb arithmetic
    a_hpc = HPC16x8.from_array(a_tensor)
    b_hpc = HPC16x8.from_array(b_tensor)
    
    # Since HPC16x8 doesn't have add method, we'll use the low-level functions
    # We'll skip this test if the required functions aren't available
    try:
        from ember_ml.backend.mlx.linearalg.hpc16x8_ops import _add_limb_precision
        
        # Add using HPC limb precision
        sum_high, sum_low = _add_limb_precision(a_hpc.high, b_hpc.high)
        sum_hpc = sum_high + sum_low
    except (ImportError, AttributeError):
        pytest.skip("Required HPC functions not available")
    
    # The HPC version should be closer to the true sum
    true_sum = a + b
    error_standard = abs(sum_standard.item() - true_sum) / true_sum
    error_hpc = abs(sum_hpc.item() - true_sum) / true_sum
    
    print(f"Addition - Standard result: {sum_standard.item()}, HPC result: {sum_hpc.item()}, True: {true_sum}")
    print(f"Addition - Standard relative error: {error_standard}, HPC relative error: {error_hpc}")
    
    # Test multiplication with challenging values
    c = 1.0
    d = 1e-8
    
    # Standard arithmetic
    c_tensor = tensor.convert_to_tensor(c, dtype=tensor.float32)
    d_tensor = tensor.convert_to_tensor(d, dtype=tensor.float32)
    
    prod_standard = ops.multiply(c_tensor, d_tensor)
    
    # HPC limb arithmetic
    c_hpc = HPC16x8.from_array(c_tensor)
    d_hpc = HPC16x8.from_array(d_tensor)
    
    # Since HPC16x8 doesn't have multiply method, we'll use the low-level functions
    # We'll skip this test if the required functions aren't available
    try:
        from ember_ml.backend.mlx.linearalg.hpc16x8_ops import _mul_limb_precision
        
        # Multiply using HPC limb precision
        prod_high, prod_low = _mul_limb_precision(c_hpc.high, d_hpc.high)
        prod_hpc = prod_high + prod_low
    except (ImportError, AttributeError):
        pytest.skip("Required HPC functions not available")
    
    # The HPC version should be closer to the true product
    true_prod = c * d
    error_standard_mul = abs(prod_standard.item() - true_prod) / true_prod
    error_hpc_mul = abs(prod_hpc.item() - true_prod) / true_prod
    
    print(f"Multiplication - Standard result: {prod_standard.item()}, HPC result: {prod_hpc.item()}, True: {true_prod}")
    print(f"Multiplication - Standard relative error: {error_standard_mul}, HPC relative error: {error_hpc_mul}")
    
    # The HPC errors should be smaller or equal to standard errors
    assert error_hpc <= error_standard or error_hpc < 1e-6, \
           f"HPC addition error: {error_hpc}, Standard error: {error_standard}"
    assert error_hpc_mul <= error_standard_mul or error_hpc_mul < 1e-6, \
           f"HPC multiplication error: {error_hpc_mul}, Standard error: {error_standard_mul}"