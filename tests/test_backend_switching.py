"""
Test the backend switching mechanism.

This test verifies that the backend switching mechanism works correctly,
and that operations are correctly updated when the backend changes.
"""

import unittest
import numpy as np

from ember_ml.ops import (
    set_backend, get_backend, add, subtract, multiply, divide,
    stats, linearalg, bitwise
)

class TestBackendSwitching(unittest.TestCase):
    """Test the backend switching mechanism."""
    
    def setUp(self):
        """Save the original backend to restore it after the test."""
        self.original_backend = get_backend()
    
    def tearDown(self):
        """Restore the original backend."""
        set_backend(self.original_backend)
    
    def test_backend_switching(self):
        """Test that backend switching works correctly."""
        # Get the available backends
        from ember_ml.backend import _AVAILABLE_BACKENDS
        
        # Skip the test if there's only one backend available
        if len(_AVAILABLE_BACKENDS) < 2:
            self.skipTest("Need at least 2 backends to test switching")
        
        # Choose two different backends to test
        backend1 = _AVAILABLE_BACKENDS[0]
        backend2 = _AVAILABLE_BACKENDS[1]
        
        # Set the backend to the first one
        set_backend(backend1)
        self.assertEqual(get_backend(), backend1)
        
        # Create some test data
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        # Perform some operations with the first backend
        result1_add = add(a, b)
        result1_subtract = subtract(a, b)
        result1_multiply = multiply(a, b)
        result1_divide = divide(a, b)
        
        # Get some stats operations
        result1_mean = stats.mean(a)
        result1_std = stats.std(a)
        
        # Get some linearalg operations
        result1_norm = linearalg.norm(a)
        
        # Switch to the second backend
        set_backend(backend2)
        self.assertEqual(get_backend(), backend2)
        
        # Perform the same operations with the second backend
        result2_add = add(a, b)
        result2_subtract = subtract(a, b)
        result2_multiply = multiply(a, b)
        result2_divide = divide(a, b)
        
        # Get some stats operations
        result2_mean = stats.mean(a)
        result2_std = stats.std(a)
        
        # Get some linearalg operations
        result2_norm = linearalg.norm(a)
        
        # The results should be the same regardless of the backend
        np.testing.assert_allclose(result1_add, result2_add)
        np.testing.assert_allclose(result1_subtract, result2_subtract)
        np.testing.assert_allclose(result1_multiply, result2_multiply)
        np.testing.assert_allclose(result1_divide, result2_divide)
        np.testing.assert_allclose(result1_mean, result2_mean)
        np.testing.assert_allclose(result1_std, result2_std)
        np.testing.assert_allclose(result1_norm, result2_norm)
        
        # Switch back to the first backend
        set_backend(backend1)
        self.assertEqual(get_backend(), backend1)
        
        # Perform the operations again
        result3_add = add(a, b)
        result3_subtract = subtract(a, b)
        result3_multiply = multiply(a, b)
        result3_divide = divide(a, b)
        
        # Get some stats operations
        result3_mean = stats.mean(a)
        result3_std = stats.std(a)
        
        # Get some linearalg operations
        result3_norm = linearalg.norm(a)
        
        # The results should be the same as the first time
        np.testing.assert_allclose(result1_add, result3_add)
        np.testing.assert_allclose(result1_subtract, result3_subtract)
        np.testing.assert_allclose(result1_multiply, result3_multiply)
        np.testing.assert_allclose(result1_divide, result3_divide)
        np.testing.assert_allclose(result1_mean, result3_mean)
        np.testing.assert_allclose(result1_std, result3_std)
        np.testing.assert_allclose(result1_norm, result3_norm)
        
        # Print a success message
        print(f"Successfully switched between backends {backend1} and {backend2}")

if __name__ == "__main__":
    unittest.main()