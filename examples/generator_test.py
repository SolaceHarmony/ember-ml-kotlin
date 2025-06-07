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
import pandas as pd
import numpy as np
def generate_data(n_samples=1000, n_features=10, anomaly_fraction=0.05) -> pd.DataFrame:
    """Generate synthetic data with anomalies."""
    # Import necessary modules
    from ember_ml.nn import tensor
    from ember_ml import ops
    
    # Generate normal data
    normal_data = tensor.random_normal((n_samples, n_features), mean=0.0, stddev=1.0)
    normal_data = tensor.to_numpy(normal_data)
    
    # Add correlations between features
    normal_tensor = tensor.convert_to_tensor(normal_data)
    for i in range(1, n_features):
        feature_i = tensor.slice_tensor(normal_tensor, [0, i], [-1, 1])
        feature_0 = tensor.slice_tensor(normal_tensor, [0, 0], [-1, 1])
        weighted_i = ops.multiply(feature_i, 0.5)
        weighted_0 = ops.multiply(feature_0, 0.5)
        combined = ops.add(weighted_i, weighted_0)
        # Update the tensor using bracket assignment
        # Create updated tensor by combining slices before and after the modified column
        if i > 0:
            left_slice = tensor.slice_tensor(normal_tensor, [0, 0], [-1, i])
            combined_reshaped = tensor.reshape(combined, [-1, 1])
            if i < n_features - 1:
                right_slice = tensor.slice_tensor(normal_tensor, [0, i+1], [-1, -1])
                normal_tensor = tensor.concatenate([left_slice, combined_reshaped, right_slice], axis=1)
            else:
                normal_tensor = tensor.concatenate([left_slice, combined_reshaped], axis=1)
        else:
            combined_reshaped = tensor.reshape(combined, [-1, 1])
            right_slice = tensor.slice_tensor(normal_tensor, [0, i+1], [-1, -1])
            normal_tensor = tensor.concatenate([combined_reshaped, right_slice], axis=1)
    
    # Add temporal patterns
    for i in range(n_samples):
        time_value = ops.divide(tensor.convert_to_tensor(i, dtype=tensor.float32), 50.0, dtype=tensor.float32)
        sin_value = ops.multiply(ops.sin(time_value), 0.5, dtype=tensor.float32)
        row = tensor.slice_tensor(normal_tensor, [i, 0], [1, -1])
        updated_row = ops.add(row, sin_value)
        # Update the tensor using bracket assignment
        normal_tensor[i, :] = tensor.squeeze(updated_row, axis=0)
    
    
    # Generate anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = tensor.convert_to_tensor(tensor.random_choice(n_samples, n_anomalies, replace=False))
    
    # Create different types of anomalies
    normal_tensor = tensor.convert_to_tensor(normal_data)
    for idx in anomaly_indices:
        anomaly_type = tensor.convert_to_tensor(np.random.randint(0, 3))
        
        if anomaly_type == 0:  # Spike anomaly
            feature_idx = tensor.convert_to_tensor(np.random.randint(0, n_features))
            spike_value = tensor.convert_to_tensor(tensor.random_uniform(3.0, 5.0))
            current_value = normal_tensor[idx, feature_idx]
            updated_value = ops.add(current_value, tensor.convert_to_tensor(spike_value))
            # Update the tensor using bracket assignment
            normal_tensor[idx, feature_idx] = updated_value
            
        elif anomaly_type == 1:  # Correlation anomaly
            random_values = tensor.random_normal((n_features,), mean=0.0, stddev=1.0)
            # Update the tensor using bracket assignment
            normal_tensor[idx, :] = random_values
            
        else:  # Collective anomaly
            random_values = tensor.random_uniform((n_features,), minval=2.0, maxval=3.0)
            current_row = normal_tensor[idx, :]
            updated_row = ops.add(current_row, random_values)
            # Update the tensor using bracket assignment
            normal_tensor[idx, :] = updated_row
    
    # Convert back to numpy
    data = tensor.to_numpy(normal_tensor)
    
    # Create DataFrame
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    
    # Add anomaly label
    df['anomaly'] = 0
    df.loc[anomaly_indices, 'anomaly'] = 1
    
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
# Ensure train_features is in NumPy format
if hasattr(train_features, 'numpy'):
    validation_data=val_features if isinstance(val_features, TensorLike) else (val_features.numpy() if hasattr(val_features, 'numpy') else
                     ops.to_numpy(val_features) if hasattr(ops, 'to_numpy') else val_features),
elif hasattr(ops, 'to_numpy'):
    train_features_np = ops.to_numpy(train_features)
else:
    train_features_np = train_features  # Assume it's already NumPy

detector.fit(
    X=train_features_np,
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
# Combine validation and anomaly data for testing
test_features = ops.vstack([val_features, anomaly_features])
test_labels = ops.hstack([tensor.zeros(len(val_features)), tensor.ones(len(anomaly_features))])

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
confusion_matrix = tensor.convert_to_tensor([
    [true_negatives, false_positives],
    [false_negatives, true_positives]
])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Normal', 'Anomaly'])
plt.yticks([0, 1], ['Normal', 'Anomaly'])

# Add text annotations
thresh = confusion_matrix.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

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
normal_scores = anomaly_scores[test_labels == 0]
anomaly_scores_filtered = anomaly_scores[test_labels == 1]

# Plot histograms
plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal')
plt.hist(anomaly_scores_filtered, bins=30, alpha=0.7, label='Anomaly')
plt.axvline(detector.anomaly_threshold, color='r', linestyle='--', label='Threshold')
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
visualizer.plot_reconstructions(detector.rbm, test_features[:5], show=True)

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


