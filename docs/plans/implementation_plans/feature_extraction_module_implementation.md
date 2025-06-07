# Feature Extraction Module Implementation Plan

This document provides a detailed implementation plan for refactoring the Feature Extraction components to use the ember_ml Module system.

## Current Implementation

The current feature extraction implementation includes:

1. `TerabyteFeatureExtractor`: For extracting features from BigQuery datasets
2. `TerabyteTemporalStrideProcessor`: For processing features with different stride lengths

These components are implemented as standalone classes and don't integrate with the ember_ml Module system.

## Refactoring Goals

1. **Module Integration**: Implement feature extraction as subclasses of `Module`
2. **Backend Agnosticism**: Use `ops` for all operations to support any backend
3. **Parameter Management**: Use the `Parameter` class for trainable parameters
4. **Training Separation**: Separate model definition from training logic
5. **Maintain Functionality**: Preserve all existing functionality

## Implementation Details

### 1. Base Feature Extractor Module

```python
class BaseFeatureExtractorModule(Module):
    """
    Base class for feature extraction modules using the ember_ml Module system.
    
    This module provides a foundation for building feature extractors
    with different architectures and processing methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base feature extractor module.
        
        Args:
            **kwargs: Additional arguments
        """
        super().__init__()
    
    def forward(self, data, **kwargs):
        """
        Forward pass through the feature extractor.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Extracted features
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def fit(self, data, **kwargs):
        """
        Fit the feature extractor to the data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        raise NotImplementedError("Subclasses must implement fit method")
```

### 2. Terabyte Feature Extractor Module

```python
class TerabyteFeatureExtractorModule(BaseFeatureExtractorModule):
    """
    Feature extractor module for terabyte-scale data.
    
    This module provides functionality for extracting features from
    large datasets, with support for chunked processing and memory management.
    """
    
    def __init__(
        self,
        chunk_size: int = 100000,
        max_memory_gb: float = 16.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the terabyte feature extractor module.
        
        Args:
            chunk_size: Number of rows to process per chunk
            max_memory_gb: Maximum memory usage in GB
            verbose: Whether to print progress information
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        # For tracking processing
        self.register_buffer('feature_names', None)
        self.register_buffer('categorical_features', None)
        self.register_buffer('numerical_features', None)
        self.register_buffer('target_column', None)
        
        # For data preprocessing
        self.register_buffer('scaler', None)
        self.register_buffer('imputer', None)
    
    def setup_connection(self, credentials_path=None, **kwargs):
        """
        Set up connection to data source.
        
        Args:
            credentials_path: Path to credentials file
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        # Implementation depends on data source
        # For BigQuery, this would set up the client
        # For other sources, this would set up the appropriate connection
        return self
    
    def fit(self, data, target_column=None, **kwargs):
        """
        Fit the feature extractor to the data.
        
        Args:
            data: Input data
            target_column: Target column name
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        # Identify feature types
        if isinstance(data, pd.DataFrame):
            self._fit_dataframe(data, target_column, **kwargs)
        else:
            raise ValueError("Unsupported data type")
        
        return self
    
    def _fit_dataframe(self, df, target_column=None, **kwargs):
        """
        Fit the feature extractor to a DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        # Identify feature types
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if col == target_column:
                continue
            
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Store feature information
        self.feature_names = tensor.convert_to_tensor(df.columns.tolist())
        self.categorical_features = tensor.convert_to_tensor(categorical_features)
        self.numerical_features = tensor.convert_to_tensor(numerical_features)
        self.target_column = tensor.convert_to_tensor(target_column) if target_column else None
        
        # Fit preprocessing components
        if len(numerical_features) > 0:
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            
            self.scaler = StandardScaler()
            self.scaler.fit(df[numerical_features])
            
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(df[numerical_features])
        
        return self
    
    def forward(self, data, **kwargs):
        """
        Extract features from the data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Extracted features
        """
        if isinstance(data, pd.DataFrame):
            return self._process_dataframe(data, **kwargs)
        else:
            raise ValueError("Unsupported data type")
    
    def _process_dataframe(self, df, **kwargs):
        """
        Process a DataFrame to extract features.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional arguments
            
        Returns:
            Extracted features
        """
        # Process numerical features
        numerical_features = ops.to_numpy(self.numerical_features).tolist()
        if len(numerical_features) > 0:
            # Impute missing values
            df_numerical = pd.DataFrame(
                self.imputer.transform(df[numerical_features]),
                columns=numerical_features
            )
            
            # Scale features
            df_numerical = pd.DataFrame(
                self.scaler.transform(df_numerical),
                columns=numerical_features
            )
        else:
            df_numerical = pd.DataFrame()
        
        # Process categorical features
        categorical_features = ops.to_numpy(self.categorical_features).tolist()
        if len(categorical_features) > 0:
            # One-hot encode categorical features
            df_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
        else:
            df_categorical = pd.DataFrame()
        
        # Combine features
        if not df_numerical.empty and not df_categorical.empty:
            df_features = pd.concat([df_numerical, df_categorical], axis=1)
        elif not df_numerical.empty:
            df_features = df_numerical
        elif not df_categorical.empty:
            df_features = df_categorical
        else:
            df_features = pd.DataFrame()
        
        # Convert to tensor
        features = tensor.convert_to_tensor(df_features.values, dtype=ops.float32)
        
        return features
    
    def prepare_data(self, data, target_column=None, test_size=0.2, val_size=0.25, **kwargs):
        """
        Prepare data for training, validation, and testing.
        
        Args:
            data: Input data
            target_column: Target column name
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (train_data, val_data, test_data, train_features, val_features, test_features)
        """
        from sklearn.model_selection import train_test_split
        
        # Fit feature extractor
        self.fit(data, target_column, **kwargs)
        
        # Split data into train, validation, and test sets
        if target_column:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, random_state=42
            )
            
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
        else:
            train_val_data, test_data = train_test_split(
                data, test_size=test_size, random_state=42
            )
            
            train_data, val_data = train_test_split(
                train_val_data, test_size=val_size, random_state=42
            )
        
        # Extract features
        train_features = self.forward(train_data)
        val_features = self.forward(val_data)
        test_features = self.forward(test_data)
        
        return train_data, val_data, test_data, train_features, val_features, test_features
```

### 3. Temporal Stride Processor Module

```python
class TemporalStrideProcessorModule(BaseFeatureExtractorModule):
    """
    Temporal stride processor module.
    
    This module provides functionality for processing features with
    different stride lengths, enabling multi-scale temporal analysis.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        stride_perspectives: List[int] = [1, 3, 5],
        pca_components: int = 32,
        batch_size: int = 10000,
        use_incremental_pca: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the temporal stride processor module.
        
        Args:
            window_size: Size of the sliding window
            stride_perspectives: List of stride lengths
            pca_components: Number of PCA components
            batch_size: Batch size for processing
            use_incremental_pca: Whether to use incremental PCA
            verbose: Whether to print progress information
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.use_incremental_pca = use_incremental_pca
        self.verbose = verbose
        
        # For PCA components
        self.register_buffer('pca_models', {})
    
    def fit(self, data, **kwargs):
        """
        Fit the temporal stride processor to the data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        # Convert data to numpy if it's a tensor
        if ops.is_tensor(data):
            data = ops.to_numpy(data)
        
        # Initialize PCA models for each stride
        from sklearn.decomposition import PCA, IncrementalPCA
        
        for stride in self.stride_perspectives:
            # Create windowed data for this stride
            windowed_data = self._create_windows(data, stride)
            
            # Flatten windows
            flattened_data = windowed_data.reshape(windowed_data.shape[0], -1)
            
            # Initialize PCA model
            if self.use_incremental_pca:
                pca = IncrementalPCA(n_components=self.pca_components)
                
                # Fit in batches
                for i in range(0, len(flattened_data), self.batch_size):
                    batch = flattened_data[i:i+self.batch_size]
                    pca.partial_fit(batch)
            else:
                pca = PCA(n_components=self.pca_components)
                pca.fit(flattened_data)
            
            # Store PCA model
            self.pca_models[stride] = pca
        
        return self
    
    def forward(self, data, **kwargs):
        """
        Process data with different stride lengths.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of processed data for each stride
        """
        # Convert data to numpy if it's a tensor
        if ops.is_tensor(data):
            data = ops.to_numpy(data)
        
        # Process data for each stride
        result = {}
        
        for stride in self.stride_perspectives:
            # Create windowed data for this stride
            windowed_data = self._create_windows(data, stride)
            
            # Flatten windows
            flattened_data = windowed_data.reshape(windowed_data.shape[0], -1)
            
            # Apply PCA
            pca_data = self.pca_models[stride].transform(flattened_data)
            
            # Store result
            result[stride] = tensor.convert_to_tensor(pca_data, dtype=ops.float32)
        
        return result
    
    def _create_windows(self, data, stride):
        """
        Create windowed data with the specified stride.
        
        Args:
            data: Input data
            stride: Stride length
            
        Returns:
            Windowed data
        """
        # Get number of samples and features
        n_samples, n_features = data.shape
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // stride + 1
        
        # Create windowed data
        windowed_data = np.zeros((n_windows, self.window_size, n_features))
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + self.window_size
            windowed_data[i] = data[start_idx:end_idx]
        
        return windowed_data
    
    def process_large_dataset(self, data_generator, **kwargs):
        """
        Process a large dataset in chunks.
        
        Args:
            data_generator: Generator yielding batches of data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of processed data for each stride
        """
        # Initialize result
        result = {stride: [] for stride in self.stride_perspectives}
        
        # Process each batch
        for batch in data_generator:
            # Convert batch to numpy if it's a tensor
            if ops.is_tensor(batch):
                batch = ops.to_numpy(batch)
            
            # Process batch
            batch_result = self.forward(batch)
            
            # Append results
            for stride, data in batch_result.items():
                result[stride].append(ops.to_numpy(data))
        
        # Combine results
        for stride in self.stride_perspectives:
            if result[stride]:
                result[stride] = tensor.convert_to_tensor(
                    np.vstack(result[stride]),
                    dtype=ops.float32
                )
            else:
                result[stride] = ops.zeros((0, self.pca_components), dtype=ops.float32)
        
        return result
```

### 4. BigQuery Feature Extractor Module

```python
class BigQueryFeatureExtractorModule(TerabyteFeatureExtractorModule):
    """
    Feature extractor module for BigQuery data.
    
    This module provides functionality for extracting features from
    BigQuery tables, with support for chunked processing and memory management.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "US",
        chunk_size: int = 100000,
        max_memory_gb: float = 16.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the BigQuery feature extractor module.
        
        Args:
            project_id: GCP project ID
            location: BigQuery location
            chunk_size: Number of rows to process per chunk
            max_memory_gb: Maximum memory usage in GB
            verbose: Whether to print progress information
            **kwargs: Additional arguments
        """
        super().__init__(
            chunk_size=chunk_size,
            max_memory_gb=max_memory_gb,
            verbose=verbose,
            **kwargs
        )
        
        self.project_id = project_id
        self.location = location
        self.client = None
    
    def setup_connection(self, credentials_path=None, **kwargs):
        """
        Set up connection to BigQuery.
        
        Args:
            credentials_path: Path to service account credentials
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            # Set up credentials
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location
                )
            else:
                self.client = bigquery.Client(
                    project=self.project_id,
                    location=self.location
                )
            
            if self.verbose:
                print(f"Connected to BigQuery project: {self.client.project}")
        except ImportError:
            raise ImportError("google-cloud-bigquery is required for BigQuery support")
        
        return self
    
    def prepare_data(
        self,
        table_id: str,
        target_column: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        """
        Prepare data from BigQuery for training, validation, and testing.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            target_column: Target column name
            limit: Optional row limit for testing
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (train_data, val_data, test_data, train_features, val_features, test_features)
        """
        # Check if client is initialized
        if self.client is None:
            self.setup_connection()
        
        # Build query
        query = f"SELECT * FROM `{table_id}`"
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query
        if self.verbose:
            print(f"Executing query: {query}")
        
        df = self.client.query(query).to_dataframe()
        
        # Prepare data
        return super().prepare_data(df, target_column, **kwargs)
```

### 5. Factory Functions

```python
def create_feature_extractor(
    data_source: str = "dataframe",
    **kwargs
) -> BaseFeatureExtractorModule:
    """
    Create a feature extractor module.
    
    Args:
        data_source: Data source type ('dataframe', 'bigquery', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Feature extractor module
    """
    if data_source == "bigquery":
        return BigQueryFeatureExtractorModule(**kwargs)
    else:
        return TerabyteFeatureExtractorModule(**kwargs)

def create_temporal_processor(
    window_size: int = 10,
    stride_perspectives: List[int] = [1, 3, 5],
    **kwargs
) -> TemporalStrideProcessorModule:
    """
    Create a temporal stride processor module.
    
    Args:
        window_size: Size of the sliding window
        stride_perspectives: List of stride lengths
        **kwargs: Additional arguments
        
    Returns:
        Temporal stride processor module
    """
    return TemporalStrideProcessorModule(
        window_size=window_size,
        stride_perspectives=stride_perspectives,
        **kwargs
    )
```

## Integration with Pipeline

The Feature Extraction Modules will be integrated into the pipeline as follows:

```python
# In pipeline_module.py
from ember_ml.features.feature_extractor_module import (
    create_feature_extractor,
    create_temporal_processor
)

class PipelineModule(Module):
    def __init__(self, feature_dim, rbm_hidden_units=64, ncp_units=128, **kwargs):
        super().__init__()
        
        # Initialize feature extractor
        self.feature_extractor = create_feature_extractor(
            data_source=kwargs.get('data_source', 'dataframe'),
            chunk_size=kwargs.get('chunk_size', 100000),
            max_memory_gb=kwargs.get('max_memory_gb', 16.0),
            verbose=kwargs.get('verbose', True)
        )
        
        # Initialize temporal processor
        self.temporal_processor = create_temporal_processor(
            window_size=kwargs.get('window_size', 10),
            stride_perspectives=kwargs.get('stride_perspectives', [1, 3, 5]),
            pca_components=kwargs.get('pca_components', 32),
            batch_size=kwargs.get('batch_size', 10000),
            use_incremental_pca=kwargs.get('use_incremental_pca', True),
            verbose=kwargs.get('verbose', True)
        )
        
        # ...
    
    def extract_features(self, data, target_column=None, **kwargs):
        """Extract features from data."""
        return self.feature_extractor.prepare_data(
            data,
            target_column=target_column,
            **kwargs
        )
    
    def apply_temporal_processing(self, features, **kwargs):
        """Apply temporal processing to features."""
        # Convert to generator if not already
        if not hasattr(features, '__next__'):
            def data_generator(df, batch_size=10000):
                for i in range(0, len(df), batch_size):
                    yield df.iloc[i:i+batch_size].values
            
            features_generator = data_generator(features, batch_size=10000)
        else:
            features_generator = features
        
        # Process features
        return self.temporal_processor.process_large_dataset(
            features_generator,
            **kwargs
        )
```

## Testing Plan

1. **Unit Tests**:
   - Test initialization of each module
   - Test feature extraction from different data sources
   - Test temporal processing with different stride lengths
   - Test handling of categorical and numerical features

2. **Integration Tests**:
   - Test integration with pipeline
   - Test end-to-end feature extraction and processing

3. **Performance Tests**:
   - Test memory usage with large datasets
   - Test processing speed with different chunk sizes

## Implementation Timeline

1. **Day 1**: Implement base feature extractor module
2. **Day 2**: Implement terabyte feature extractor module
3. **Day 3**: Implement temporal stride processor module
4. **Day 4**: Implement BigQuery feature extractor module
5. **Day 5**: Write tests and debug
6. **Day 6**: Integrate with pipeline and finalize

## Conclusion

This implementation plan provides a detailed roadmap for refactoring the Feature Extraction components to use the ember_ml Module system. The resulting implementation will be more modular, maintainable, and backend-agnostic, while preserving all the functionality of the original implementation.