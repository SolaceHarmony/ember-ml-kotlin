import pytest
import numpy as np # For comparison with known correct results
import array # For working with limb arrays
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.limb import hpc_limb_core # Import the module
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
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
    assert hpc_limb_core.limbs_to_int(limbs2) == (1 << 64)

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