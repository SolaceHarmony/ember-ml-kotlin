"""
Test the asyncml module.

This test verifies that the asyncml module works correctly, particularly with backend switching.
"""

import unittest
import asyncio
import numpy as np
import sys
import importlib.util

# Check if ray is available
RAY_AVAILABLE = importlib.util.find_spec("ray") is not None

# Skip all tests if ray is not available
if not RAY_AVAILABLE:
    print("Ray is not installed, skipping asyncml tests")
    sys.exit(0)

# Import asyncml only if ray is available
try:
    import ember_ml.asyncml.ops
except ImportError as e:
    print(f"Error importing ember_ml.asyncml.ops: {e}")
    RAY_AVAILABLE = False
    sys.exit(0)
class TestAsyncML(unittest.TestCase):
    """Test the asyncml module."""

    def setUp(self):
        """Save the original backend to restore it after the test."""
        from ember_ml.ops import get_backend
        self.original_backend = get_backend()

    def tearDown(self):
        """Restore the original backend."""
        from ember_ml.ops import set_backend
        set_backend(self.original_backend)

    async def test_async_ops(self):
        """Test that async operations work correctly."""
        # Import the asyncml module
        from ember_ml.asyncml import ops

        # Create some test data
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        # Test some basic operations
        result_add = await ops.add(a, b)
        np.testing.assert_array_equal(result_add, np.array([5, 7, 9]))

        result_multiply = await ops.multiply(a, b)
        np.testing.assert_array_equal(result_multiply, np.array([4, 10, 18]))

        # Test some math operations
        result_sqrt = await ops.sqrt(a)
        np.testing.assert_allclose(result_sqrt, np.sqrt(a))

        # Test some stats operations
        result_mean = await ops.stats.mean(a)
        self.assertEqual(result_mean, 2.0)

        # Test some linearalg operations
        result_norm = await ops.linearalg.norm(a)
        self.assertAlmostEqual(result_norm, np.sqrt(14))

    async def test_backend_switching(self):
        """Test that backend switching works correctly with asyncml."""
        # Import the asyncml module
        from ember_ml.asyncml import ops
        from ember_ml.backend import _AVAILABLE_BACKENDS

        # Skip the test if there's only one backend available
        if len(_AVAILABLE_BACKENDS) < 2:
            self.skipTest("Need at least 2 backends to test switching")

        # Choose two different backends to test
        backend1 = _AVAILABLE_BACKENDS[0]
        backend2 = _AVAILABLE_BACKENDS[1]

        # Set the backend to the first one
        ops.set_backend(backend1)
        self.assertEqual(ops.get_backend(), backend1)

        # Create some test data
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        # Perform some operations with the first backend
        result1_add = await ops.add(a, b)
        result1_multiply = await ops.multiply(a, b)
        result1_mean = await ops.stats.mean(a)

        # Switch to the second backend
        ops.set_backend(backend2)
        self.assertEqual(ops.get_backend(), backend2)

        # Perform the same operations with the second backend
        result2_add = await ops.add(a, b)
        result2_multiply = await ops.multiply(a, b)
        result2_mean = await ops.stats.mean(a)

        # The results should be the same regardless of the backend
        np.testing.assert_array_equal(result1_add, result2_add)
        np.testing.assert_array_equal(result1_multiply, result2_multiply)
        self.assertEqual(result1_mean, result2_mean)

        # Switch back to the first backend
        ops.set_backend(backend1)
        self.assertEqual(ops.get_backend(), backend1)

        # Perform the operations again
        result3_add = await ops.add(a, b)
        result3_multiply = await ops.multiply(a, b)
        result3_mean = await ops.stats.mean(a)

        # The results should be the same as the first time
        np.testing.assert_array_equal(result1_add, result3_add)
        np.testing.assert_array_equal(result1_multiply, result3_multiply)
        self.assertEqual(result1_mean, result3_mean)

        # Print a success message
        print(f"Successfully switched between backends {backend1} and {backend2} with asyncml")

class AsyncioTestCase(unittest.TestCase):
    """Base class for asyncio test cases."""

    def run_async(self, coro):
        """Run a coroutine in the event loop."""
        return asyncio.run(coro)

    def run(self, result=None):
        """Run the test, wrapping async test methods."""
        orig_result = result
        if result is None:
            result = self.defaultTestResult()
            startTestRun = getattr(result, 'startTestRun', None)
            if startTestRun is not None:
                startTestRun()

        testMethod = getattr(self, self._testMethodName)
        if asyncio.iscoroutinefunction(testMethod):
            # If the test method is a coroutine function, run it with asyncio
            self._outcome = result
            try:
                self.setUp()
                try:
                    self.run_async(testMethod())
                except Exception:
                    result.addError(self, sys.exc_info())
                finally:
                    try:
                        self.tearDown()
                    except Exception:
                        result.addError(self, sys.exc_info())
            finally:
                result.stopTest(self)
            return result
        else:
            # Otherwise, use the standard unittest behavior
            return super().run(result)

# Make TestAsyncML inherit from AsyncioTestCase
TestAsyncML.__bases__ = (AsyncioTestCase,)

if __name__ == "__main__":
    unittest.main()
