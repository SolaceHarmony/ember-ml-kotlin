"""
Tests for the BigQueryFeatureExtractor.

This module contains tests for the BigQueryFeatureExtractor class and its
underlying components from the bigquery package.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.features.bigquery_feature_extractor import BigQueryFeatureExtractor


class TestBigQueryFeatureExtractor(unittest.TestCase):
    """
    Test case for the BigQueryFeatureExtractor class.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Create a mock client
        self.mock_client = MagicMock()
        
        # Create a sample extractor
        self.extractor = BigQueryFeatureExtractor(
            project_id='test-project',
            dataset_id='test-dataset',
            table_id='test-table',
            numeric_columns=['num1', 'num2'],
            categorical_columns=['cat1', 'cat2'],
            datetime_columns=['date1']
        )
        
        # Replace the client with the mock
        self.extractor._client = self.mock_client
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            'date1': pd.date_range(start='2023-01-01', periods=5)
        })
    
    @patch('ember_ml.features.bigquery.initialize_client')
    def test_initialize_client(self, mock_init_client):
        """
        Test initializing the BigQuery client.
        """
        # Set up the mock
        mock_init_client.return_value = self.mock_client
        
        # Initialize the client
        self.extractor.initialize_client()
        
        # Check that the mock was called with the correct arguments
        mock_init_client.assert_called_once_with(
            'test-project',
            None
        )
        
        # Check that the client was set
        self.assertEqual(self.extractor._client, self.mock_client)
    
    @patch('ember_ml.features.bigquery.execute_query')
    def test_execute_query(self, mock_execute_query):
        """
        Test executing a query.
        """
        # Set up the mock
        mock_execute_query.return_value = self.sample_data
        
        # Execute a query
        result = self.extractor.execute_query('SELECT * FROM test-table')
        
        # Check that the mock was called with the correct arguments
        mock_execute_query.assert_called_once_with(
            self.mock_client,
            'SELECT * FROM test-table'
        )
        
        # Check the result
        pd.testing.assert_frame_equal(result, self.sample_data)
    
    @patch('ember_ml.features.bigquery.fetch_table_schema')
    def test_fetch_table_schema(self, mock_fetch_schema):
        """
        Test fetching the table schema.
        """
        # Set up the mock
        schema = {
            'num1': 'FLOAT',
            'num2': 'FLOAT',
            'cat1': 'STRING',
            'cat2': 'STRING',
            'date1': 'DATETIME'
        }
        mock_fetch_schema.return_value = schema
        
        # Fetch the schema
        result = self.extractor.fetch_table_schema()
        
        # Check that the mock was called with the correct arguments
        mock_fetch_schema.assert_called_once_with(
            self.mock_client,
            'test-dataset',
            'test-table'
        )
        
        # Check the result
        self.assertEqual(result, schema)
    
    def test_auto_detect_column_types(self):
        """
        Test auto-detecting column types.
        """
        # Set up the mock
        schema = {
            'num1': 'FLOAT',
            'num2': 'INTEGER',
            'cat1': 'STRING',
            'cat2': 'STRING',
            'date1': 'DATETIME',
            'date2': 'DATE'
        }
        self.extractor.fetch_table_schema = MagicMock(return_value=schema)
        
        # Auto-detect column types
        self.extractor.auto_detect_column_types()
        
        # Check the detected column types
        self.assertEqual(set(self.extractor.numeric_columns), {'num1', 'num2'})
        self.assertEqual(set(self.extractor.categorical_columns), {'cat1', 'cat2'})
        self.assertEqual(set(self.extractor.datetime_columns), {'date1', 'date2'})
    
    @patch('ember_ml.features.bigquery.process_numeric_features')
    @patch('ember_ml.features.bigquery.process_categorical_features')
    @patch('ember_ml.features.bigquery.process_datetime_features')
    def test_extract_features(
        self,
        mock_process_datetime,
        mock_process_categorical,
        mock_process_numeric
    ):
        """
        Test extracting features.
        """
        # Set up the mocks
        numeric_tensor = tensor.ones((5, 2))
        numeric_names = ['num1', 'num2']
        mock_process_numeric.return_value = (numeric_tensor, numeric_names)
        
        categorical_tensor = tensor.ones((5, 3))
        categorical_names = ['cat1_A', 'cat1_B', 'cat1_C']
        mock_process_categorical.return_value = (categorical_tensor, categorical_names)
        
        datetime_tensor = tensor.ones((5, 1))
        datetime_names = ['date1_year']
        mock_process_datetime.return_value = (datetime_tensor, datetime_names)
        
        # Extract features
        features, names = self.extractor.extract_features(
            data=self.sample_data,
            create_samples=False,
            capture_processing=False
        )
        
        # Check the calls to the mocks
        mock_process_numeric.assert_called_once()
        mock_process_categorical.assert_called_once()
        mock_process_datetime.assert_called_once()
        
        # Check the results
        self.assertEqual(tensor.shape(features)[0], 5)  # 5 samples
        self.assertEqual(tensor.shape(features)[1], 6)  # 2 + 3 + 1 features
        self.assertEqual(
            names,
            ['num1', 'num2', 'cat1_A', 'cat1_B', 'cat1_C', 'date1_year']
        )
    
    def test_extract_features_with_no_features(self):
        """
        Test extracting features when no features are available.
        """
        # Set up an extractor with no columns
        extractor = BigQueryFeatureExtractor(
            project_id='test-project',
            dataset_id='test-dataset',
            table_id='test-table',
            numeric_columns=[],
            categorical_columns=[],
            datetime_columns=[]
        )
        
        # Test that extracting features raises a ValueError
        with self.assertRaises(ValueError):
            extractor.extract_features(data=self.sample_data)


if __name__ == '__main__':
    unittest.main()