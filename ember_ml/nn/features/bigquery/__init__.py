"""
BigQuery feature extraction utilities for Ember ML.

This package provides functionality for extracting features from Google BigQuery
datasets for use in Ember ML models.
"""

from ember_ml.nn.features.bigquery.client import (
    initialize_client,
    execute_query,
    fetch_table_schema
)

from ember_ml.nn.features.bigquery.feature_processing import (
    process_numeric_features,
    process_categorical_features,
    process_datetime_features,
    handle_missing_values,
    remove_outliers,
    normalize_robust
)

from ember_ml.nn.features.bigquery.encoding import (
    hash_encode,
    one_hot_encode
)

from ember_ml.nn.features.bigquery.visualization import (
    create_sample_table,
    create_sample_table_from_tensor,
    capture_frame,
    generate_processing_animation
)

__all__ = [
    'initialize_client',
    'execute_query',
    'fetch_table_schema',
    'process_numeric_features',
    'process_categorical_features',
    'process_datetime_features',
    'handle_missing_values',
    'remove_outliers',
    'normalize_robust',
    'hash_encode',
    'one_hot_encode',
    'create_sample_table',
    'create_sample_table_from_tensor',
    'capture_frame',
    'generate_processing_animation'
]