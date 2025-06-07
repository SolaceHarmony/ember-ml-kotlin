# Feature Extraction Module (`nn.features`)

The `ember_ml.nn.features` module provides a comprehensive set of components for feature extraction, transformation, and processing. It combines stateful components (classes like `PCA`, `Standardize`, `Normalize`) with stateless, backend-agnostic operations (like `one_hot`), as well as specialized feature extractors for various data sources and formats.

## Importing

```python
from ember_ml.nn import features
from ember_ml.nn import tensor # For creating example tensors
```

## Core Stateful Components

These components maintain internal state and are typically used in a fit/transform pattern. They are instantiated via factory functions or directly using their class names.

### PCA

Performs Principal Component Analysis for dimensionality reduction.

**Instantiation:**
```python
# Using the factory function (recommended)
pca_instance = features.pca()

# Direct instantiation
from ember_ml.nn.features import PCA
pca_instance = PCA()
```

**Usage:**
```python
# Fit PCA to data
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pca_instance.fit(data, n_components=2)

# Transform data
transformed = pca_instance.transform(data)

# Inverse transform
reconstructed = pca_instance.inverse_transform(transformed)
```
**Key Methods:** `fit`, `transform`, `fit_transform`, `inverse_transform`

### Standardize

Standardizes features by removing the mean and scaling to unit variance.

**Instantiation:**
```python
# Using the factory function (recommended)
std_scaler = features.standardize()

# Direct instantiation
from ember_ml.nn.features import Standardize
std_scaler = Standardize()
```

**Usage:**
```python
# Fit the scaler
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
std_scaler.fit(data, with_mean=True, with_std=True)

# Transform data
standardized_data = std_scaler.transform(data)

# Inverse transform
original_data = std_scaler.inverse_transform(standardized_data)
```
**Key Methods:** `fit`, `transform`, `fit_transform`, `inverse_transform`

### Normalize

Normalizes features using various normalization techniques (e.g., L1, L2, max).

**Instantiation:**
```python
# Using the factory function (recommended)
normalizer = features.normalize()

# Direct instantiation
from ember_ml.nn.features import Normalize
normalizer = Normalize()
```

**Usage:**
```python
# Fit the normalizer
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
normalizer.fit(data, norm="l2", axis=1)

# Transform data
normalized_data = normalizer.transform(data)
```
**Key Methods:** `fit`, `transform`, `fit_transform`

## Temporal Feature Processing

### TemporalStrideProcessor

Processes data into multi-stride temporal representations, enabling multi-scale temporal analysis.

**Instantiation:**
```python
from ember_ml.nn.features import TemporalStrideProcessor

processor = TemporalStrideProcessor(
    window_size=5,
    stride_perspectives=[1, 3, 5],
    pca_components=None  # Auto-calculated if None
)
```

**Usage:**
```python
# Process data into multi-stride temporal representations
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
stride_results = processor.process_batch(data)

# Get explained variance for a specific stride
explained_variance = processor.get_explained_variance(stride=1)

# Get feature importance for a specific stride
feature_importance = processor.get_feature_importance(stride=1)
```
**Key Methods:** `process_batch`, `get_explained_variance`, `get_feature_importance`

### TerabyteTemporalStrideProcessor

Extended version of TemporalStrideProcessor designed for very large datasets.

**Instantiation:**
```python
from ember_ml.nn.features import TerabyteTemporalStrideProcessor

processor = TerabyteTemporalStrideProcessor(
    window_size=100,
    stride_perspectives=[1, 10, 50],
    chunk_size=1000  # Process data in chunks of this size
)
```

## Column-Based Feature Extraction

### ColumnFeatureExtractor

Extracts features from tabular data on a column-by-column basis, handling categorical, numerical, and text columns.

**Instantiation:**
```python
from ember_ml.nn.features import ColumnFeatureExtractor

extractor = ColumnFeatureExtractor(
    numeric_strategy='standard',
    categorical_strategy='onehot',
    datetime_strategy='cyclical',
    text_strategy='basic',
    max_categories=100
)
```

**Usage:**
```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'numeric_col': [1.0, 2.0, 3.0, 4.0],
    'categorical_col': ['A', 'B', 'A', 'C'],
    'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
})

# Extract features
features_tensor, feature_names = extractor.extract_features(
    df,
    numeric_columns=['numeric_col'],
    categorical_columns=['categorical_col'],
    datetime_columns=['datetime_col']
)
```
**Key Methods:** `extract_features`

### ColumnPCAFeatureExtractor

Extends ColumnFeatureExtractor with PCA dimensionality reduction.

**Instantiation:**
```python
from ember_ml.nn.features import ColumnPCAFeatureExtractor

extractor = ColumnPCAFeatureExtractor(
    n_components=10,
    whiten=True
)
```

### TemporalColumnFeatureExtractor

Extends ColumnFeatureExtractor with temporal feature extraction capabilities.

**Instantiation:**
```python
from ember_ml.nn.features import TemporalColumnFeatureExtractor

extractor = TemporalColumnFeatureExtractor(
    window_size=5,
    stride=1
)
```

## Data Source-Specific Extractors

### BigQueryFeatureExtractor

Feature extractor for Google BigQuery data sources.

**Instantiation:**
```python
from ember_ml.nn.features import BigQueryFeatureExtractor

extractor = BigQueryFeatureExtractor(
    project_id="your-gcp-project",
    dataset_id="your-dataset",
    table_id="your-table",
    credentials_path="/path/to/credentials.json",
    numeric_columns=["col1", "col2"],
    categorical_columns=["col3"],
    datetime_columns=["col4"]
)
```

**Usage:**
```python
# Auto-detect column types
extractor.auto_detect_column_types()

# Extract features
features_tensor, feature_names = extractor.extract_features(
    limit=1000,
    handle_missing=True,
    handle_outliers=True,
    normalize=True
)
```
**Key Methods:** `initialize_client`, `execute_query`, `fetch_table_schema`, `auto_detect_column_types`, `fetch_data`, `extract_features`

### TerabyteFeatureExtractor

Designed for extracting features from very large datasets, often involving chunking and out-of-core processing.

**Instantiation:**
```python
from ember_ml.nn.features import TerabyteFeatureExtractor

extractor = TerabyteFeatureExtractor(
    window_size=100,
    stride=10,
    feature_functions=['mean', 'std', 'min', 'max'],
    chunk_size=1000
)
```

## Visualization and Animation

### AnimatedFeatureProcessor

Feature processor with animated visualization and sample tables for different data types.

**Instantiation:**
```python
from ember_ml.nn.features import AnimatedFeatureProcessor

processor = AnimatedFeatureProcessor(
    visualization_enabled=True,
    sample_tables_enabled=True
)
```

**Usage:**
```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'numeric_col': [1.0, 2.0, 3.0, 4.0],
    'categorical_col': ['A', 'B', 'A', 'C']
})

# Process numeric features with animation
numeric_features = processor.process_numeric_features(
    df,
    columns=['numeric_col'],
    with_imputation=True,
    with_outlier_handling=True,
    with_normalization=True
)

# Process categorical features with animation
categorical_features = processor.process_categorical_features(
    df,
    columns=['categorical_col'],
    encoding='one_hot'
)

# Get processing frames for animation
frames = processor.get_processing_frames()

# Get sample tables
tables = processor.get_sample_tables()
```
**Key Methods:** `process_numeric_features`, `process_categorical_features`, `get_processing_frames`, `get_sample_tables`, `clear_artifacts`

## Utility Components

### GenericCSVLoader

Loads and preprocesses CSV files for feature extraction.

**Instantiation:**
```python
from ember_ml.nn.features import GenericCSVLoader

loader = GenericCSVLoader(
    delimiter=',',
    header=0,
    encoding='utf-8'
)
```

### GenericTypeDetector

Detects column types in tabular data.

**Instantiation:**
```python
from ember_ml.nn.features import GenericTypeDetector

detector = GenericTypeDetector()
```

**Usage:**
```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'numeric_col': [1.0, 2.0, 3.0, 4.0],
    'categorical_col': ['A', 'B', 'A', 'C'],
    'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
})

# Detect column types
column_types = detector.detect_column_types(df)
```

### EnhancedTypeDetector

Enhanced version of GenericTypeDetector with additional type detection capabilities.

**Instantiation:**
```python
from ember_ml.nn.features import EnhancedTypeDetector

detector = EnhancedTypeDetector()
```

### GenericFeatureEngineer

Generic feature engineering component for tabular data.

**Instantiation:**
```python
from ember_ml.nn.features import GenericFeatureEngineer

engineer = GenericFeatureEngineer()
```

### SpeedtestEventProcessor

Specialized processor for speedtest event data.

**Instantiation:**
```python
from ember_ml.nn.features import SpeedtestEventProcessor

processor = SpeedtestEventProcessor()
```

## Stateless Feature Operations

### `features.one_hot(indices, depth, **kwargs)`

Convert integer indices to a one-hot encoded representation.

**Parameters:**
- `indices`: Tensor containing indices to convert.
- `depth`: The number of classes (determines the length of the one-hot vector).
- `**kwargs`: Backend-specific arguments.

**Returns:**
- One-hot encoded tensor (native backend tensor).

**Example:**
```python
# Create indices
indices = tensor.convert_to_tensor([0, 2, 1, 0])

# Convert to one-hot encoding
one_hot_encoded = features.one_hot(indices, depth=3)
print(one_hot_encoded)
# [[1, 0, 0],
#  [0, 0, 1],
#  [0, 1, 0],
#  [1, 0, 0]]
```

## Backend Support

All feature extraction operations and components are designed to be backend-agnostic, leveraging the `ops` module internally where necessary. Stateful components like `PCA` manage their state independently of the backend, while stateless functions like `one_hot` rely on the dynamically aliased backend implementation.

## Notes

- For basic tensor creation and manipulation, use the `ember_ml.nn.tensor` module.
- For mathematical and statistical operations within custom feature functions, use the `ember_ml.ops` and `ember_ml.ops.stats` modules respectively.