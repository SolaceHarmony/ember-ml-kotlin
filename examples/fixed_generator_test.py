# %% [markdown]
# # Advanced Anomaly Detection with Ember ML
#
# This notebook demonstrates how to use Restricted Boltzmann Machines (RBMs) for anomaly detection with the Ember ML framework. We'll explore how to:
#
# 1. Generate synthetic data with anomalies
# 2. Train an RBM-based anomaly detector
# 3. Evaluate detection performance
# 4. Visualize the results
#
# This example showcases Ember ML's backend-agnostic capabilities.

# %%
# Import necessary libraries
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Import Ember ML components
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.models.rbm_anomaly_detector import RBMBasedAnomalyDetector
from ember_ml.visualization.rbm_visualizer import RBMVisualizer

# Set a backend (choose 'numpy', 'torch', or 'mlx')
set_backend('torch')  # Using PyTorch for GPU operations
print(f"Using backend: {ops.get_backend()}")

# Set random seed for reproducibility
tensor.set_seed(42)

# Create output directories
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

# %% [markdown]
# ## 1. Generate Synthetic Data with Anomalies
#
# We'll create synthetic data with three types of anomalies:
# 1. **Spike anomalies**: Sudden spikes in individual features
# 2. **Correlation anomalies**: Breaking the normal correlation patterns
# 3. **Collective anomalies**: Unusual patterns across multiple features

# %%
def generate_data(n_samples=1000, n_features=10, anomaly_fraction=0.05) -> pd.DataFrame:
    """Generate synthetic data with anomalies."""
    # Import necessary modules
    from ember_ml.nn import tensor
    from ember_ml import ops

    # Generate normal data using Ember ML tensor operations
    normal_data = tensor.random_normal((n_samples, n_features), mean=0.0, stddev=1.0)

    # Add correlations between features
    # Perform operations directly on the EmberTensor
    data_tensor = normal_data
    for i in range(1, n_features):
        feature_i = tensor.slice_tensor(data_tensor, [0, i], [-1, 1])
        feature_0 = tensor.slice_tensor(data_tensor, [0, 0], [-1, 1])
        # Explicitly convert scalars to tensors
        weighted_i = ops.multiply(feature_i, tensor.convert_to_tensor(0.5, dtype=tensor.float32))
        weighted_0 = ops.multiply(feature_0, tensor.convert_to_tensor(0.5, dtype=tensor.float32))
        combined = ops.add(weighted_i, weighted_0)
        # Update the tensor using bracket assignment
        # Create updated tensor by combining slices before and after the modified column
        if i > 0:
            left_slice = tensor.slice_tensor(data_tensor, [0, 0], [-1, i])
            combined_reshaped = tensor.reshape(combined, [-1, 1])
            if i < n_features - 1:
                right_slice = tensor.slice_tensor(data_tensor, [0, i+1], [-1, -1])
                data_tensor = tensor.concatenate([left_slice, combined_reshaped, right_slice], axis=1)
            else:
                data_tensor = tensor.concatenate([left_slice, combined_reshaped], axis=1)
        else:
            combined_reshaped = tensor.reshape(combined, [-1, 1])
            right_slice = tensor.slice_tensor(data_tensor, [0, i+1], [-1, -1])
            data_tensor = tensor.concatenate([combined_reshaped, right_slice], axis=1)

    # Add temporal patterns
    for i in range(n_samples):
        # Removed the incorrect dtype argument from ops.divide
        time_value = ops.divide(tensor.convert_to_tensor(i, dtype=tensor.float32), tensor.convert_to_tensor(50.0, dtype=tensor.float32))
        # Removed the incorrect dtype argument from ops.multiply
        sin_value = ops.multiply(ops.sin(time_value), tensor.convert_to_tensor(0.5, dtype=tensor.float32))
        row = tensor.slice_tensor(data_tensor, [i, 0], [1, -1])
        updated_row = ops.add(row, sin_value)
        # Update the tensor using bracket assignment
        data_tensor[i, :] = tensor.squeeze(updated_row, axis=0)


    # Generate anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    # Use Ember ML tensor operations for random choice
    all_indices = tensor.arange(n_samples)
    shuffled_indices = tensor.random_permutation(all_indices)
    anomaly_indices = shuffled_indices[:n_anomalies]

    # Create different types of anomalies
    # Operate directly on data_tensor (EmberTensor)
    for idx in anomaly_indices:
        # Use Ember ML tensor operations for random choices
        anomaly_type = tensor.cast(tensor.random_uniform((), maxval=3), tensor.int64)

        if anomaly_type == 0:  # Spike anomaly
            feature_idx = tensor.cast(tensor.random_uniform((), maxval=n_features), tensor.int64)
            # Explicitly convert scalars to tensors
            spike_value = tensor.random_uniform((), minval=tensor.convert_to_tensor(3.0, dtype=tensor.float32), maxval=tensor.convert_to_tensor(5.0, dtype=tensor.float32))
            current_value = data_tensor[idx, feature_idx]
            updated_value = ops.add(current_value, spike_value)
            # Update the tensor using bracket assignment
            data_tensor[idx, feature_idx] = updated_value

        elif anomaly_type == 1:  # Correlation anomaly
            # Explicitly convert scalar to tensor
            random_values = tensor.random_normal((n_features,), mean=tensor.convert_to_tensor(0.0, dtype=tensor.float32), stddev=tensor.convert_to_tensor(1.0, dtype=tensor.float32))
            # Update the tensor using bracket assignment
            data_tensor[idx, :] = random_values

        else:  # Collective anomaly
            # Explicitly convert scalars to tensors
            random_values = tensor.random_uniform((n_features,), minval=tensor.convert_to_tensor(2.0, dtype=tensor.float32), maxval=tensor.convert_to_tensor(3.0, dtype=tensor.float32))
            current_row = data_tensor[idx, :]
            updated_row = ops.add(current_row, random_values)
            # Update the tensor using bracket assignment
            data_tensor[idx, :] = updated_row

    # Convert the final EmberTensor to numpy for pandas
    data = tensor.to_numpy(data_tensor)

    # Create DataFrame
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    # Add anomaly label
    # anomaly_indices is an EmberTensor, convert to numpy for pandas indexing
    df['anomaly'] = 0
    df.loc[tensor.to_numpy(anomaly_indices), 'anomaly'] = 1

    return df


# Make sure we're using the right data types when generating data
df = generate_data(n_samples=1000, n_features=10, anomaly_fraction=0.05)
# Plot a few features over time
plt.figure(figsize=(15, 8))

# Get normal and anomaly indices
normal_indices = df['anomaly'] == 0
anomaly_indices = df['anomaly'] == 1

# Plot the first 3 features
for i in range(3):
    feature_col = f"feature_{i+1}"
    plt.subplot(3, 1, i+1)

    # Plot normal data
    plt.plot(df.index[normal_indices],
             df.loc[normal_indices, feature_col],
             'b-', alpha=0.7, label='Normal')

    # Plot anomalies
    plt.scatter(df.index[anomaly_indices],
                df.loc[anomaly_indices, feature_col],
                color='red', marker='o', label='Anomaly')

    plt.title(f'{feature_col} Over Time')
    plt.ylabel('Value')
    if i == 2:  # Only show x-label for the bottom plot
        plt.xlabel('Time Index')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
## 3. Prepare Data for Anomaly Detection



# %%
# Get features and split data
features = df.drop('anomaly', axis=1).values
labels = df['anomaly'].values

# Split data into normal and anomalous
normal_indices = labels == 0
anomaly_indices = labels == 1

normal_features = features[normal_indices]
anomaly_features = features[anomaly_indices]

# Split normal data into training and validation sets (80/20)
n_normal = len(normal_features)
n_train = int(n_normal * 0.8)

train_features = tensor.convert_to_tensor(normal_features[:n_train])
val_features = tensor.convert_to_tensor(normal_features[n_train:])

print(f"Training set: {train_features.shape} samples")
print(f"Validation set: {val_features.shape} samples")
print(f"Anomaly set: {anomaly_features.shape} samples")

# %% [markdown]
# ## 4. Train the RBM-based Anomaly Detector
#
# Now we'll train an RBM-based anomaly detector using Ember ML. The detector will learn the normal patterns in the data and identify deviations as anomalies.

# %%
# Initialize RBM-based anomaly detector
print("Initializing RBM-based anomaly detector...")
detector = RBMBasedAnomalyDetector(
    n_hidden=5,
    learning_rate=0.01,
    momentum=0.5,
    weight_decay=0.0001,
    batch_size=10,
    anomaly_threshold_percentile=95.0,
    anomaly_score_method='reconstruction',
    track_states=True
)

# Train anomaly detector
print("\nTraining anomaly detector...")
start_time = time.time()
# Pass EmberTensors directly to the fit method
detector.fit(
    X=train_features,
    validation_data=val_features,
    epochs=30,
    k=1,
    early_stopping_patience=5,
    verbose=True
)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Print detector summary
print("\nAnomaly Detector Summary:")
print(detector.summary())

# %% [markdown]
# ## 5. Detect Anomalies and Evaluate Performance
#
# Let's use the trained detector to identify anomalies in our test data and evaluate its performance.

# %%
# Convert anomaly_features (NumPy) to EmberTensor before combining
anomaly_features_tensor = tensor.convert_to_tensor(anomaly_features)

# Combine validation and anomaly data for testing
test_features = ops.vstack([val_features, anomaly_features_tensor])
test_labels = ops.hstack([tensor.zeros(val_features.shape[0]), tensor.ones(anomaly_features_tensor.shape[0])]) # Use length of tensor

# Predict anomalies
print("Detecting anomalies...")
predicted_anomalies = detector.predict(test_features)
anomaly_scores = detector.anomaly_score(test_features)

# Compute metrics
true_positives = ops.logical_and(ops.equal(predicted_anomalies, 1), ops.equal(test_labels, 1))
false_positives = ops.logical_and(ops.equal(predicted_anomalies, 1), ops.equal(test_labels, 0))
true_negatives = ops.logical_and(ops.equal(predicted_anomalies, 0), ops.equal(test_labels, 0))
false_negatives = ops.logical_and(ops.equal(predicted_anomalies, 0), ops.equal(test_labels, 1))

# Convert boolean tensors to count
tp_sum = stats.sum(tensor.cast(true_positives, tensor.int32))
fp_sum = stats.sum(tensor.cast(false_positives, tensor.int32))
tn_sum = stats.sum(tensor.cast(true_negatives, tensor.int32))
fn_sum = stats.sum(tensor.cast(false_negatives, tensor.int32))

# Calculate precision, recall, and F1 score
precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nPerformance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Create confusion matrix
# Convert tensor counts to Python integers for the confusion matrix list
confusion_matrix = [
    [int(tn_sum), int(fp_sum)],
    [int(fn_sum), int(tp_sum)]
]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
# Use numpy array for imshow
plt.imshow(tensor.convert_to_tensor(confusion_matrix), interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Normal', 'Anomaly'])
plt.yticks([0, 1], ['Normal', 'Anomaly'])

# Add text annotations
# Use numpy array for max and indexing
thresh = tensor.convert_to_tensor(confusion_matrix).max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(confusion_matrix[i][j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i][j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Visualize Anomaly Detection Results
#
# Let's visualize the anomaly scores and how well they separate normal from anomalous data points.

# %%
# Initialize visualizer
visualizer = RBMVisualizer()

# Plot anomaly scores
print("Plotting anomaly scores...")
plt.figure(figsize=(10, 6))

# Split scores by label
# Convert anomaly_scores and test_labels to numpy for matplotlib
normal_scores = tensor.to_numpy(anomaly_scores)[tensor.to_numpy(test_labels) == 0]
anomaly_scores_filtered = tensor.to_numpy(anomaly_scores)[tensor.to_numpy(test_labels) == 1]

# Plot histograms
plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal')
plt.hist(anomaly_scores_filtered, bins=30, alpha=0.7, label='Anomaly')
# Convert anomaly_threshold to numpy for matplotlib
plt.axvline(tensor.to_numpy(detector.anomaly_threshold), color='r', linestyle='--', label='Threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.title('Anomaly Score Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Visualize RBM Internals
#
# Let's explore the internal representations learned by the RBM.

# %%
# Plot training curve
print("Plotting training curve...")
visualizer.plot_training_curve(detector.rbm, show=True)

# Plot weight matrix
print("\nPlotting weight matrix...")
visualizer.plot_weight_matrix(detector.rbm, show=True)

# Plot reconstructions
print("\nPlotting reconstructions...")
# Convert test_features to numpy for plotting
visualizer.plot_reconstructions(detector.rbm, tensor.to_numpy(test_features[:5]), show=True)

# %% [markdown]
# ## 8. Conclusion
#
# In this notebook, we've demonstrated how to use Ember ML's RBM-based anomaly detection capabilities to identify anomalies in multivariate data. The key advantages of this approach include:
#
# 1. **Unsupervised learning**: No labeled anomaly data required for training
# 2. **Capturing complex patterns**: RBMs can learn non-linear relationships in the data
# 3. **Backend agnosticism**: The same code works across different computational backends
# 4. **Interpretable results**: Visualizations help understand the model's behavior
#
# This approach can be extended to real-world anomaly detection tasks in various domains, including cybersecurity, industrial monitoring, and financial fraud detection.