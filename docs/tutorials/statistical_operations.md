# Statistical Operations in Ember ML

This tutorial introduces the statistical operations available in Ember ML and demonstrates how to use them effectively in your projects.

## Introduction

Ember ML provides a comprehensive set of statistical operations through the `ops.stats` module. These operations are backend-agnostic, meaning they work the same way regardless of whether you're using NumPy, PyTorch, or MLX as your backend.

## Importing Statistical Operations

To use statistical operations in Ember ML, import them from the `ops.stats` module:

```python
from ember_ml import ops
from ember_ml.nn import tensor

# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Use statistical operations
mean_value = ops.stats.mean(x)
print(f"Mean: {mean_value}")
```

## Basic Statistical Operations

### Mean and Sum

The most commonly used statistical operations are mean and sum:

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute mean
mean_all = ops.stats.mean(x)  # Mean of all elements
mean_rows = ops.stats.mean(x, axis=1)  # Mean of each row
mean_cols = ops.stats.mean(x, axis=0)  # Mean of each column

print(f"Mean (all): {mean_all}")
print(f"Mean (rows): {mean_rows}")
print(f"Mean (columns): {mean_cols}")

# Compute sum
sum_all = ops.stats.sum(x)  # Sum of all elements
sum_rows = ops.stats.sum(x, axis=1)  # Sum of each row
sum_cols = ops.stats.sum(x, axis=0)  # Sum of each column

print(f"Sum (all): {sum_all}")
print(f"Sum (rows): {sum_rows}")
print(f"Sum (columns): {sum_cols}")
```

### Min and Max

Finding minimum and maximum values:

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Find minimum values
min_all = ops.stats.min(x)  # Minimum of all elements
min_rows = ops.stats.min(x, axis=1)  # Minimum of each row
min_cols = ops.stats.min(x, axis=0)  # Minimum of each column

print(f"Min (all): {min_all}")
print(f"Min (rows): {min_rows}")
print(f"Min (columns): {min_cols}")

# Find maximum values
max_all = ops.stats.max(x)  # Maximum of all elements
max_rows = ops.stats.max(x, axis=1)  # Maximum of each row
max_cols = ops.stats.max(x, axis=0)  # Maximum of each column

print(f"Max (all): {max_all}")
print(f"Max (rows): {max_rows}")
print(f"Max (columns): {max_cols}")
```

### Variance and Standard Deviation

For measuring dispersion:

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute variance
var_all = ops.stats.var(x)  # Variance of all elements
var_rows = ops.stats.var(x, axis=1)  # Variance of each row
var_cols = ops.stats.var(x, axis=0)  # Variance of each column

print(f"Variance (all): {var_all}")
print(f"Variance (rows): {var_rows}")
print(f"Variance (columns): {var_cols}")

# Compute standard deviation
std_all = ops.stats.std(x)  # Standard deviation of all elements
std_rows = ops.stats.std(x, axis=1)  # Standard deviation of each row
std_cols = ops.stats.std(x, axis=0)  # Standard deviation of each column

print(f"Standard Deviation (all): {std_all}")
print(f"Standard Deviation (rows): {std_rows}")
print(f"Standard Deviation (columns): {std_cols}")
```

### Median and Percentiles

For robust statistics:

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute median
median_all = ops.stats.median(x)  # Median of all elements
median_rows = ops.stats.median(x, axis=1)  # Median of each row
median_cols = ops.stats.median(x, axis=0)  # Median of each column

print(f"Median (all): {median_all}")
print(f"Median (rows): {median_rows}")
print(f"Median (columns): {median_cols}")

# Compute percentiles
p25 = ops.stats.percentile(x, 25)  # 25th percentile
p50 = ops.stats.percentile(x, 50)  # 50th percentile (same as median)
p75 = ops.stats.percentile(x, 75)  # 75th percentile

print(f"25th percentile: {p25}")
print(f"50th percentile: {p50}")
print(f"75th percentile: {p75}")
```

## Cumulative Operations

### Cumulative Sum

For running totals:

```python
# Create a tensor
x = tensor.convert_to_tensor([1, 2, 3, 4, 5])

# Compute cumulative sum
cumsum_result = ops.stats.cumsum(x)

print(f"Input: {x}")
print(f"Cumulative sum: {cumsum_result}")
```

## Indices and Sorting

### ArgMax

Finding the indices of maximum values:

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 5, 3], [4, 2, 6], [7, 8, 0]])

# Find indices of maximum values
argmax_all = ops.stats.argmax(x)  # Index of maximum in flattened tensor
argmax_rows = ops.stats.argmax(x, axis=1)  # Index of maximum in each row
argmax_cols = ops.stats.argmax(x, axis=0)  # Index of maximum in each column

print(f"ArgMax (all): {argmax_all}")
print(f"ArgMax (rows): {argmax_rows}")
print(f"ArgMax (columns): {argmax_cols}")
```

### Sorting

Sorting values and getting sort indices:

```python
# Create a tensor
x = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4], [9, 7, 8]])

# Sort values
sorted_all = ops.stats.sort(x)  # Sort along the last axis
sorted_rows = ops.stats.sort(x, axis=1)  # Sort each row
sorted_cols = ops.stats.sort(x, axis=0)  # Sort each column
sorted_desc = ops.stats.sort(x, descending=True)  # Sort in descending order

print(f"Original: {x}")
print(f"Sorted (default): {sorted_all}")
print(f"Sorted (rows): {sorted_rows}")
print(f"Sorted (columns): {sorted_cols}")
print(f"Sorted (descending): {sorted_desc}")

# Get sort indices
argsort_all = ops.stats.argsort(x)  # Indices that would sort the tensor
argsort_desc = ops.stats.argsort(x, descending=True)  # Indices for descending sort

print(f"ArgSort (default): {argsort_all}")
print(f"ArgSort (descending): {argsort_desc}")
```

## Keeping Dimensions

By default, reduction operations like `mean`, `sum`, `min`, `max`, etc. reduce the specified dimensions. You can keep those dimensions by setting `keepdims=True`:

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute mean with and without keeping dimensions
mean_without_keepdims = ops.stats.mean(x, axis=1)  # Shape: (3,)
mean_with_keepdims = ops.stats.mean(x, axis=1, keepdims=True)  # Shape: (3, 1)

print(f"Mean without keepdims: {mean_without_keepdims}, shape: {tensor.shape(mean_without_keepdims)}")
print(f"Mean with keepdims: {mean_with_keepdims}, shape: {tensor.shape(mean_with_keepdims)}")
```

## Practical Examples

### Normalizing Data

Standardizing data (z-score normalization):

```python
# Create a tensor
x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Standardize along each column
mean = ops.stats.mean(x, axis=0, keepdims=True)
std = ops.stats.std(x, axis=0, keepdims=True)
z_scores = ops.divide(ops.subtract(x, mean), std)

print(f"Original data:\n{x}")
print(f"Standardized data (z-scores):\n{z_scores}")
```

### Computing Moving Averages

Simple moving average of time series data:

```python
# Create a time series
time_series = tensor.convert_to_tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

# Define window size
window_size = 3

# Compute moving averages
moving_avgs = []
for i in range(len(time_series) - window_size + 1):
    window = time_series[i:i+window_size]
    avg = ops.stats.mean(window)
    moving_avgs.append(tensor.item(avg))

print(f"Time series: {time_series}")
print(f"Moving averages (window={window_size}): {moving_avgs}")
```

### Anomaly Detection

Using z-scores to detect outliers:

```python
# Create a tensor with outliers
data = tensor.convert_to_tensor([2, 3, 3, 2, 1, 2, 3, 20, 2, 3])

# Compute z-scores
mean = ops.stats.mean(data)
std = ops.stats.std(data)
z_scores = ops.divide(ops.subtract(data, mean), std)

# Identify outliers (|z-score| > 2)
outliers = ops.abs(z_scores) > 2

print(f"Data: {data}")
print(f"Z-scores: {z_scores}")
print(f"Outliers: {outliers}")
```

## Best Practices

1. **Use the right axis**: Be careful with the `axis` parameter, as it determines which dimensions are reduced.
2. **Keep dimensions when needed**: Use `keepdims=True` when you need to maintain the tensor's shape for broadcasting.
3. **Backend consistency**: Statistical operations are consistent across backends, so your code will work the same way regardless of the backend.
4. **Type safety**: Statistical operations preserve the data type of the input tensor.

## Summary

The `ops.stats` module provides a comprehensive set of statistical operations for tensor analysis. These operations are backend-agnostic and follow a consistent API, making it easy to work with tensors regardless of the underlying backend.

For more information, refer to the [API documentation](../api/stats.md).