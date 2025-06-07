"""
Test that ops.stats imports work in various ways.

This test verifies that the ops.stats module can be imported and used in various ways.
"""

import unittest
import numpy as np

class TestStatsImports(unittest.TestCase):
    """Test that ops.stats imports work in various ways."""
    
    def test_import_from_ops(self):
        """Test importing stats from ops."""
        from ember_ml.ops import stats
        
        # Create a test array
        test_array = np.array([1, 2, 3, 4, 5])
        
        # Test some stats functions
        mean = stats.mean(test_array)
        self.assertEqual(mean, 3.0)
        
        std = stats.std(test_array)
        self.assertAlmostEqual(std, 1.4142135623730951)
        
        median = stats.median(test_array)
        self.assertEqual(median, 3.0)
        
        # Test min and max
        min_val = stats.min(test_array)
        self.assertEqual(min_val, 1)
        
        max_val = stats.max(test_array)
        self.assertEqual(max_val, 5)
    
    def test_import_stats_directly(self):
        """Test importing stats functions directly."""
        from ember_ml.ops.stats import mean, std, median, min, max
        
        # Create a test array
        test_array = np.array([1, 2, 3, 4, 5])
        
        # Test the imported functions
        self.assertEqual(mean(test_array), 3.0)
        self.assertAlmostEqual(std(test_array), 1.4142135623730951)
        self.assertEqual(median(test_array), 3.0)
        self.assertEqual(min(test_array), 1)
        self.assertEqual(max(test_array), 5)
    
    def test_import_stats_with_alias(self):
        """Test importing stats with an alias."""
        import ember_ml.ops.stats as stats_ops
        
        # Create a test array
        test_array = np.array([1, 2, 3, 4, 5])
        
        # Test some stats functions
        mean = stats_ops.mean(test_array)
        self.assertEqual(mean, 3.0)
        
        std = stats_ops.std(test_array)
        self.assertAlmostEqual(std, 1.4142135623730951)
        
        median = stats_ops.median(test_array)
        self.assertEqual(median, 3.0)
        
        # Test min and max
        min_val = stats_ops.min(test_array)
        self.assertEqual(min_val, 1)
        
        max_val = stats_ops.max(test_array)
        self.assertEqual(max_val, 5)

if __name__ == "__main__":
    unittest.main()