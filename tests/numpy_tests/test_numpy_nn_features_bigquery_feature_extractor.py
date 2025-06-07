import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Import Ember ML modules
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.nn.features.bigquery_feature_extractor import BigQueryFeatureExtractor
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("mlx")

@pytest.fixture
def bigquery_extractor_setup(mocker):
    """
    Pytest fixture to set up the BigQueryFeatureExtractor and mock dependencies.
    """
    # Create a mock client
    mock_client = MagicMock()

    # Create a sample extractor
    extractor = BigQueryFeatureExtractor(
        project_id='test-project',
        dataset_id='test-dataset',
        table_id='test-table',
        numeric_columns=['num1', 'num2'],
        categorical_columns=['cat1', 'cat2'],
        datetime_columns=['date1']
    )

    # Replace the client with the mock
    extractor._client = mock_client

    # Create sample data
    sample_data = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'num2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'cat1': ['A', 'B', 'A', 'C', 'B'],
        'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
        'date1': pd.date_range(start='2023-01-01', periods=5)
    })

    # Yield the extractor and sample data
    yield extractor, sample_data, mock_client

def test_initialize_client(bigquery_extractor_setup, mocker, numpy_backend):
    """
    Test initializing the BigQuery client with NumPy backend.
    """
    extractor, _, mock_client = bigquery_extractor_setup

    # Set up the mock
    mock_init_client = mocker.patch('ember_ml.features.bigquery.initialize_client')
    mock_init_client.return_value = mock_client

    # Initialize the client
    extractor.initialize_client()

    # Check that the mock was called with the correct arguments
    mock_init_client.assert_called_once_with(
        'test-project',
        None
    )

    # Check that the client was set
    assert extractor._client == mock_client

def test_execute_query(bigquery_extractor_setup, mocker, numpy_backend):
    """
    Test executing a query with NumPy backend.
    """
    extractor, sample_data, mock_client = bigquery_extractor_setup

    # Set up the mock
    mock_execute_query = mocker.patch('ember_ml.features.bigquery.execute_query')
    mock_execute_query.return_value = sample_data

    # Execute a query
    result = extractor.execute_query('SELECT * FROM test-table')

    # Check that the mock was called with the correct arguments
    mock_execute_query.assert_called_once_with(
        mock_client,
        'SELECT * FROM test-table'
    )

    # Check the result
    pd.testing.assert_frame_equal(result, sample_data)

def test_fetch_table_schema(bigquery_extractor_setup, mocker, numpy_backend):
    """
    Test fetching the table schema with NumPy backend.
    """
    extractor, _, mock_client = bigquery_extractor_setup

    # Set up the mock
    mock_fetch_schema = mocker.patch('ember_ml.features.bigquery.fetch_table_schema')
    schema = {
        'num1': 'FLOAT',
        'num2': 'INTEGER',
        'cat1': 'STRING',
        'cat2': 'STRING',
        'date1': 'DATETIME',
        'date2': 'DATE'
    }
    mock_fetch_schema.return_value = schema

    # Fetch the schema
    result = extractor.fetch_table_schema()

    # Check that the mock was called with the correct arguments
    mock_fetch_schema.assert_called_once_with(
        mock_client,
        'test-dataset',
        'test-table'
    )

    # Check the result
    assert result == schema

def test_auto_detect_column_types(bigquery_extractor_setup, numpy_backend):
    """
    Test auto-detecting column types with NumPy backend.
    """
    extractor, _, _ = bigquery_extractor_setup

    # Set up the mock for fetch_table_schema within the test function
    schema = {
        'num1': 'FLOAT',
        'num2': 'INTEGER',
        'cat1': 'STRING',
        'cat2': 'STRING',
        'date1': 'DATETIME',
        'date2': 'DATE'
    }
    extractor.fetch_table_schema = MagicMock(return_value=schema)

    # Auto-detect column types
    extractor.auto_detect_column_types()

    # Check the detected column types
    assert set(extractor.numeric_columns) == {'num1', 'num2'}
    assert set(extractor.categorical_columns) == {'cat1', 'cat2'}
    assert set(extractor.datetime_columns) == {'date1', 'date2'}

def test_extract_features(bigquery_extractor_setup, mocker, numpy_backend):
    """
    Test extracting features with NumPy backend.
    """
    extractor, sample_data, _ = bigquery_extractor_setup

    # Set up the mocks
    mock_process_numeric = mocker.patch('ember_ml.features.bigquery.process_numeric_features')
    numeric_tensor = tensor.ones((5, 2))
    numeric_names = ['num1', 'num2']
    mock_process_numeric.return_value = (numeric_tensor, numeric_names)

    mock_process_categorical = mocker.patch('ember_ml.features.bigquery.process_categorical_features')
    categorical_tensor = tensor.ones((5, 3))
    categorical_names = ['cat1_A', 'cat1_B', 'cat1_C']
    mock_process_categorical.return_value = (categorical_tensor, categorical_names)

    mock_process_datetime = mocker.patch('ember_ml.features.bigquery.process_datetime_features')
    datetime_tensor = tensor.ones((5, 1))
    datetime_names = ['date1_year']
    mock_process_datetime.return_value = (datetime_tensor, datetime_names)

    # Extract features
    features, names = extractor.extract_features(
        data=sample_data,
        create_samples=False,
        capture_processing=False
    )

    # Check the calls to the mocks
    mock_process_numeric.assert_called_once()
    mock_process_categorical.assert_called_once()
    mock_process_datetime.assert_called_once()

    # Check the results
    assert tensor.shape(features)[0] == 5  # 5 samples
    assert tensor.shape(features)[1] == 6  # 2 + 3 + 1 features
    assert names == ['num1', 'num2', 'cat1_A', 'cat1_B', 'cat1_C', 'date1_year']

def test_extract_features_with_no_features(bigquery_extractor_setup, numpy_backend):
    """
    Test extracting features when no features are available with NumPy backend.
    """
    _, sample_data, _ = bigquery_extractor_setup

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
    with pytest.raises(ValueError):
        extractor.extract_features(data=sample_data)