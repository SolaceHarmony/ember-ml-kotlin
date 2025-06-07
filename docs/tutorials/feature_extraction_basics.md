# Feature Extraction Basics

This tutorial covers the basics of feature extraction using Ember ML.

## Introduction

Feature extraction is the process of transforming raw data into features that better represent the underlying problem to predictive models. Ember ML provides powerful tools for feature extraction, especially for large-scale datasets.

## Basic Feature Extraction

### Loading Data

First, let's load some data:

```python
import pandas as pd
import ember_ml as eh
from ember_ml import ops

# Load a CSV file
df = pd.read_csv('your_data.csv')
print(f"Loaded data with shape: {df.shape}")
```

### Detecting Column Types

Ember ML can automatically detect column types:

```python
from ember_ml.features import GenericTypeDetector

# Initialize the type detector
detector = GenericTypeDetector()

# Detect column types
column_types = detector.detect_column_types(df)

print("Detected column types:")
for type_name, cols in column_types.items():
    print(f"{type_name}: {len(cols)} columns")
    if cols:
        print(f"  Example: {cols[0]}")
```

### Feature Engineering

Next, let's engineer features based on the detected column types:

```python
from ember_ml.features import GenericFeatureEngineer

# Initialize the feature engineer
engineer = GenericFeatureEngineer()

# Engineer features
df_engineered = engineer.engineer_features(df, column_types)

print(f"Original dataframe shape: {df.shape}")
print(f"Engineered dataframe shape: {df_engineered.shape}")
```

### Converting to Tensors

Finally, let's convert the engineered features to tensors for use with machine learning models:

```python
# Get numeric features
numeric_features = [col for col in df_engineered.columns
                   if pd.api.types.is_numeric_dtype(df_engineered[col].dtype)]

# Convert to tensor
features_tensor = tensor.convert_to_tensor(df_engineered[numeric_features].values)
print(f"Features tensor shape: {features_tensor.shape}")
```

## Temporal Feature Extraction

For time series data, Ember ML provides the `TemporalStrideProcessor`:

```python
from ember_ml.features import TemporalStrideProcessor

# Initialize the temporal processor
processor = TemporalStrideProcessor(
    window_size=5,
    stride_perspectives=[1, 3, 5]
)

# Process data
stride_perspectives = processor.process_batch(features_tensor.numpy())

# Print information about stride perspectives
print("\nStride Perspectives:")
for stride, perspective_data in stride_perspectives.items():
    explained_variance = processor.get_explained_variance(stride)
    variance_str = f"{explained_variance:.2f}" if explained_variance is not None else "N/A"
    print(f"Stride {stride}: Shape {perspective_data.shape}, "
          f"Explained Variance: {variance_str}")
```

## Large-Scale Feature Extraction

For large-scale datasets, Ember ML provides the `TerabyteFeatureExtractor`:

```python
from ember_ml.features import TerabyteFeatureExtractor

# Initialize the extractor
extractor = TerabyteFeatureExtractor(
    chunk_size=1000,
    max_memory_gb=1.0
)

# Extract features
result = extractor.prepare_data(
    df,
    target_column='target',
    test_size=0.2,
    validation_size=0.1
)

# Unpack results
train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
```

## Next Steps

Now that you've learned the basics of feature extraction with Ember ML, you can:

1. Explore more advanced feature extraction techniques in the [API Reference](../api/index.md)
2. Learn about [Working with Large Datasets](large_datasets.md)
3. Check out the [Examples](../examples/index.md) for more complex use cases

Happy feature engineering with Ember ML!