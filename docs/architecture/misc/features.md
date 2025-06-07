# Feature Processing and Engineering

This section documents the components within Ember ML responsible for preprocessing, transforming, and engineering features from various data types, including handling missing values, scaling, encoding, and temporal processing.

## Core Concepts

Feature engineering is a critical step in preparing data for machine learning models. Ember ML provides a backend-agnostic framework for feature processing, allowing consistent data preparation regardless of the underlying computation backend. This includes tools for:

*   Automatic type detection and analysis.
*   Handling missing values and outliers.
*   Scaling and normalization.
*   Encoding categorical and datetime features.
*   Temporal processing for time series data.
*   Dimensionality reduction (PCA).
*   Specialized processors for specific data sources (e.g., BigQuery).

## Components

### Type Detection and Analysis

*   **`ember_ml.data.type_detector.GenericTypeDetector`**: Detects column types (numeric, datetime, categorical, boolean) in a pandas DataFrame based on dtypes and simple heuristics.
    *   `detect_column_types(df)`: Performs type detection.
    *   `_is_numeric_type`, `_is_datetime_type`, `_is_boolean_type`: Internal helpers for type checking.
*   **`ember_ml.nn.features.enhanced_type_detector.EnhancedTypeDetector`**: Enhanced type detection with visualization and sample table capabilities, designed for BigQuery data.
    *   `__init__(visualization_enabled, sample_tables_enabled, cardinality_threshold, high_cardinality_threshold, null_ratio_threshold)`: Initializes with visualization/sampling flags and thresholds for cardinality and null ratios.
    *   `detect_column_types(df)`: Detects column types with detailed statistics and visualization data collection.
    *   `_detect_column_type(df, col)`: Detects type and collects stats for a single column.
    *   `_detect_basic_type(col_data)`: Detects basic type based on dtype.
    *   `_is_likely_text(col_data)`: Heuristically checks if a column is likely text.
    *   `_is_likely_identifier(col_name)`: Heuristically checks if a column name suggests an identifier.
    *   `_calculate_cardinality(col_data)`: Calculates the number of unique values.
    *   `_calculate_null_ratio(col_data)`: Calculates the ratio of null values.
    *   `_calculate_avg_text_length(col_data)`: Calculates the average length of text entries.
    *   `_collect_visualization_data(...)`: Collects data for visualizations.
    *   `_generate_type_detection_visualizations()`: Placeholder for visualization generation.
    *   `_generate_sample_tables(df, types)`: Generates sample data tables.
    *   `get_type_details_table()`: Returns detailed type detection results as a DataFrame.
    *   `get_sample_table(type_name)`: Returns a sample table for a specific type.

### General Feature Engineering

*   **`ember_ml.nn.features.generic_feature_engineer.GenericFeatureEngineer`**: Creates features based on detected column types.
    *   `__init__(max_categories, handle_unknown)`: Initializes with parameters for categorical encoding.
    *   `engineer_features(df, column_types, drop_original)`: Applies feature engineering based on provided column types.
    *   `_create_datetime_features(df, col)`: Creates cyclical features (sin/cos for hour, day, month, dayofweek) from a datetime column using `ops`.
    *   `_encode_categorical(df, col)`: One-hot encodes a categorical column, handling missing values and respecting `max_categories`. Stores mappings.
    *   `get_feature_names(df, column_types)`: Gets the names of features after engineering.
*   **`ember_ml.data.csv_loader.GenericCSVLoader`**: Loads CSV data with optional compression support and automatic type detection (though type detection is delegated to `GenericTypeDetector`).
    *   `__init__(compression_support)`: Initializes with compression support flag.
    *   `load_csv(file_path, header_file, index_col, datetime_cols, encoding)`: Loads CSV into a pandas DataFrame.
    *   `_parse_header_file(header_file)`: Parses a header file to get column names (basic implementation).

### Scaling and Normalization

*   **`ember_ml.nn.features.standardize_features.Standardize`**: Backend-agnostic standardization (removes mean, scales to unit variance).
    *   `__init__()`: Initializes internal state (mean, scale).
    *   `fit(X, with_mean, with_std, axis)`: Computes mean and standard deviation from data using `ops`.
    *   `transform(X)`: Applies centering and scaling using `ops`.
    *   `fit_transform(X, ...)`: Fits and then transforms.
    *   `inverse_transform(X)`: Scales data back to original representation using `ops`.
*   **`ember_ml.nn.features.normalize_features.Normalize`**: Backend-agnostic normalization (scales to unit norm).
    *   `__init__()`: Initializes internal state (norm type, axis).
    *   `fit(X, norm, axis)`: Computes the norm of the data using `ops.stats.sum`, `ops.sqrt`, `ops.stats.max`.
    *   `transform(X)`: Divides by the computed norm using `ops`.
    *   `fit_transform(X, ...)`: Fits and then transforms.

### Dimensionality Reduction (PCA)

*   **`ember_ml.nn.features.pca_features.PCA`**: Backend-agnostic Principal Component Analysis.
    *   `__init__()`: Initializes internal state (components, explained variance, mean, etc.).
    *   `fit(X, n_components, whiten, center, svd_solver)`: Fits the PCA model. Centers data using `ops.stats.mean`, performs SVD using `ops.svd` (or `_randomized_svd` or `ops.eigh` for covariance method), determines number of components (`_find_ncomponents`, `_infer_dimensions`), and stores results. Uses `_svd_flip` for sign correction.
    *   `_svd_flip(u, v)`: Helper for sign correction of SVD results using `ops.abs`, `ops.sign`, `ops.stats.argmax`, `tensor.stack`, `tensor.reshape`.
    *   `_find_ncomponents(...)`: Helper to determine the number of components based on user input or explained variance using `ops.cumsum`, `ops.less`, `ops.stats.sum`, `ops.add`, `ops.stats.argmax`.
    *   `_infer_dimensions(...)`: Helper for Minka's MLE for dimensionality selection using `ops.stats.mean`, `ops.stats.sum`, `ops.log`, `ops.multiply`, `ops.add`, `ops.divide`, `tensor.tensor_scatter_nd_update`.
    *   `_randomized_svd(...)`: Helper for randomized SVD using `ops.matmul`, `ops.qr`, `tensor.random_normal`, `tensor.transpose`.
    *   `transform(X)`: Applies centering and matrix multiplication with components using `ops.subtract`, `ops.matmul`, `tensor.transpose`. Applies whitening if enabled using `ops.sqrt`, `ops.clip`, `ops.divide`.
    *   `fit_transform(X, ...)`: Fits and then transforms.
    *   `inverse_transform(X)`: Transforms data back to original space using `ops.matmul`, `ops.add`, `ops.sqrt`, `ops.clip`, `ops.multiply`.

### Temporal Processing

*   **`ember_ml.nn.features.temporal_processor.TemporalStrideProcessor`**: Processes data into multi-stride temporal representations using sliding windows and PCA.
    *   `__init__(window_size, stride_perspectives, pca_components)`: Initializes window size, stride lengths, and PCA components.
    *   `process_batch(data)`: Processes a batch of data, creating strided sequences and applying PCA blend for each stride.
    *   `_create_strided_sequences(data, stride)`: Creates sliding windows with a given stride.
    *   `_apply_pca_blend(window_batch, stride)`: Applies PCA to flattened windows for a specific stride. Fits PCA if not already fit.
    *   `get_explained_variance(stride)`: Returns the sum of explained variance ratios for a stride's PCA model.
    *   `get_feature_importance(stride)`: Calculates feature importance based on PCA components.
*   **`ember_ml.nn.features.terabyte_feature_extractor.TerabyteTemporalStrideProcessor`**: Extends `TemporalStrideProcessor` with optimizations for terabyte-scale data using incremental PCA and batch processing.
    *   `__init__(..., batch_size, use_incremental_pca, verbose, preferred_backend)`: Initializes base parameters and adds batch size, incremental PCA flag, verbosity, and backend preference.
    *   `process_large_dataset(data_generator, maintain_state)`: Processes data from a generator in batches, optionally maintaining state between batches. Uses incremental PCA (`_apply_incremental_pca`) or standard PCA (`_apply_pca_blend`).
    *   `create_windows(data, stride)`: Creates windows of data with a specific stride (tensor-based implementation).
    *   `_apply_incremental_pca(window_batch, stride)`: Applies incremental PCA (partial fit and transform) using scikit-learn.
    *   `get_explained_variance(stride)`: Returns the sum of explained variance ratios for a stride's PCA model.
    *   `get_feature_importance(stride)`: Calculates feature importance based on PCA components.

### Specialized Extractors

*   **`ember_ml.nn.features.bigquery.initialize_client(...)`**: Placeholder for initializing a BigQuery client.
*   **`ember_ml.nn.features.bigquery.execute_query(...)`**: Placeholder for executing a BigQuery query.
*   **`ember_ml.nn.features.bigquery.fetch_table_schema(...)`**: Placeholder for fetching a BigQuery table schema.
*   **`ember_ml.nn.features.bigquery.process_numeric_features(...)`**: Placeholder for processing numeric features from BigQuery.
*   **`ember_ml.nn.features.bigquery.process_categorical_features(...)`**: Placeholder for processing categorical features from BigQuery.
*   **`ember_ml.nn.features.bigquery.process_datetime_features(...)`**: Placeholder for processing datetime features from BigQuery.
*   **`ember_ml.nn.features.bigquery.create_sample_table(...)`**: Placeholder for creating a sample table.
*   **`ember_ml.nn.features.bigquery.create_sample_table_from_tensor(...)`**: Placeholder for creating a sample table from a tensor.
*   **`ember_ml.nn.features.bigquery.capture_frame(...)`**: Placeholder for capturing a processing step frame.
*   **`ember_ml.nn.features.bigquery.generate_processing_animation(...)`**: Placeholder for generating a processing animation.
*   **`ember_ml.nn.features.bigquery_feature_extractor.BigQueryFeatureExtractor(ColumnFeatureExtractor)`**: Feature extractor for BigQuery data, leveraging the placeholder functions from `ember_ml.nn.features.bigquery`.
    *   `__init__(project_id, dataset_id, table_id, credentials_path, numeric_columns, categorical_columns, datetime_columns, target_column, device)`: Initializes with BigQuery connection details and column lists.
    *   `initialize_client()`: Initializes the BigQuery client (calls placeholder).
    *   `execute_query(query)`: Executes a query (calls placeholder).
    *   `fetch_table_schema()`: Fetches schema (calls placeholder).
    *   `auto_detect_column_types()`: Automatically detects column types from schema.
    *   `fetch_data(...)`: Fetches data from the table.
    *   `extract_features(...)`: Extracts features by calling placeholder processing functions for each type and combining results.
    *   `_apply_forced_column_types(...)`: Internal helper to apply user-specified column type overrides.
    *   `_move_columns_to_type(...)`: Internal helper to move columns between type lists.
    *   `_process_all_feature_types(...)`: Internal helper to orchestrate processing of all feature types.
    *   `_combine_features()`: Internal helper to concatenate processed feature tensors.
    *   `_prepare_metadata(...)`: Internal helper to prepare metadata about the processing.
    *   `_create_result()`: Internal helper to format the final result dictionary.
    *   `_empty_result()`: Internal helper to create an empty result on failure.
    *   `_track_memory_usage(...)`: Internal helper to track memory usage using `psutil`.
    *   `prepare_for_rbm(...)`: Prepares features for RBM training (ensures [0,1] range, optionally binarizes).
    *   `get_memory_usage_table()`: Returns memory usage data as a DataFrame.
*   **`ember_ml.nn.features.terabyte_feature_extractor.TerabyteFeatureExtractor(BaseTerabyteFeatureExtractor)`**: Feature extractor for terabyte-scale data from BigQuery using BigFrames.
    *   `__init__(project_id, location, chunk_size, max_memory_gb, verbose, preferred_backend)`: Initializes with BigQuery/BigFrames parameters and backend preference.
    *   `setup_bigquery_connection(credentials_path)`: Sets up BigFrames options and BigQuery client.
    *   `optimize_bigquery_query(...)`: Creates an optimized BigQuery query string.
    *   `get_table_row_count(...)`: Gets the approximate row count of a table.
    *   `process_bigquery_in_chunks(...)`: Processes a BigQuery table in chunks using BigFrames.
    *   `_monitor_memory()`: Internal helper to monitor memory usage using `psutil`.
    *   `_detect_column_types(schema)`: Detects column types from BigFrames schema.
    *   `_is_numeric_type(col_type)` / `_is_datetime_type(col_type)`: Internal helpers for type checking BigFrames dtypes.
    *   `prepare_data(...)`: Orchestrates data preparation from BigQuery, including splitting, processing, scaling, and imputation.
    *   `_split_data(df, index_col)`: Splits BigFrames data into train/val/test sets (temporal or random split).
    *   `_prepare_dataframe(...)`: Prepares a dataframe split with feature engineering (flattening structs, datetime features, one-hot encoding), scaling, and imputation.
    *   `_flatten_struct(df, struct_col_name)`: Internal helper to flatten a struct column using `json_normalize`.
    *   `_create_datetime_features(df, col)`: Internal helper to create cyclical datetime features using `backend_utils.sin_cos_transform`.
*   **`ember_ml.nn.features.test_feature_extraction.test_feature_extraction(...)`**: A test function demonstrating the feature extraction pipeline (loading, type detection, engineering, temporal processing).
*   **`ember_ml.nn.features.wave_extractor.WaveFeatureExtractor`**: Extracts features from waveform data using sliding windows and frequency-domain transformations (`ops.frame`, `ops.hann_window`, `ops.stft`).
    *   `__init__(window_size, hop_length)`: Initializes window size and hop length.
    *   `extract(waveform)`: Extracts features from a waveform tensor.