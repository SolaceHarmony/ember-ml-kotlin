"""
RBM-based Anomaly Detection Example

This script demonstrates how to use the RBM-based anomaly detector
with the generic feature extraction library to detect anomalies in data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Import our modules from ember_ml
from ember_ml.data import GenericCSVLoader, GenericTypeDetector
from ember_ml.nn.features.temporal_processor import TemporalStrideProcessor
from ember_ml.nn.features.feature_engineer import GenericFeatureEngineer
from ember_ml.models.rbm_anomaly_detector import RBMBasedAnomalyDetector
from ember_ml.visualization.rbm_visualizer import RBMVisualizer


def generate_telemetry_data(n_samples=1000, n_features=10, anomaly_fraction=0.05):
    """
    Generate synthetic telemetry data with anomalies.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        anomaly_fraction: Fraction of anomalous samples
        
    Returns:
        DataFrame with telemetry data and anomaly labels
    """
    # Generate normal data using tensor operations
    from ember_ml.nn import tensor
    from ember_ml import ops
    
    # Set random seed for reproducibility
    tensor.set_seed(42)
    
    # Generate normal data
    normal_data = tensor.random_normal((n_samples, n_features), mean=0.0, stddev=1.0)
    normal_data = tensor.to_numpy(normal_data)  # Convert to numpy for pandas compatibility
    
    # Add some correlations between features
    # Convert back to tensor for operations
    normal_tensor = tensor.convert_to_tensor(normal_data)
    for i in range(1, n_features):
        # Create indices for the operation
        feature_i = tensor.slice_tensor(normal_tensor, [0, i], [-1, 1])
        feature_0 = tensor.slice_tensor(normal_tensor, [0, 0], [-1, 1])
        
        # Compute the weighted sum
        weighted_i = ops.multiply(feature_i, 0.5)
        weighted_0 = ops.multiply(feature_0, 0.5)
        combined = ops.add(weighted_i, weighted_0)
        
        # Update the tensor using index_update
        normal_tensor = tensor.index_update(normal_tensor,
                                           tensor.index[:, i],
                                           tensor.squeeze(combined, axis=1))
    
    # Convert back to numpy
    normal_data = tensor.to_numpy(normal_tensor)
    
    # Add some temporal patterns
    # Convert back to tensor for operations
    normal_tensor = tensor.convert_to_tensor(normal_data)
    for i in range(n_samples):
        # Create time value and compute sin
        time_value = ops.divide(tensor.convert_to_tensor(i, dtype=tensor.float32), 50.0)
        sin_value = ops.multiply(ops.sin(time_value), 0.5)
        
        # Get the row and add the sin value
        row = tensor.slice_tensor(normal_tensor, [i, 0], [1, -1])
        updated_row = ops.add(row, sin_value)
        
        # Update the tensor using index_update
        normal_tensor = tensor.index_update(normal_tensor,
                                           tensor.index[i, :],
                                           tensor.squeeze(updated_row, axis=0))
    
    # Convert back to numpy
    normal_data = tensor.to_numpy(normal_tensor)
    
    # Generate anomalies
    n_anomalies = tensor.cast(ops.multiply(tensor.convert_to_tensor(n_samples), anomaly_fraction), tensor.int32)
    n_anomalies_np = tensor.item(n_anomalies)  # Get as Python scalar
    
    # Generate random indices for anomalies
    # Note: tensor.random_choice is not available, so we'll use numpy for this specific operation
    # In a real application, you would implement this using tensor operations
    anomaly_indices_np = ops.random_choice(n_samples, n_anomalies_np, replace=False)
    anomaly_indices = anomaly_indices_np.tolist()  # Convert to list for pandas indexing
    
    # Convert back to tensor for operations
    normal_tensor = tensor.convert_to_tensor(normal_data)
    
    # Create different types of anomalies
    for idx in anomaly_indices:
        # Generate random anomaly type (0, 1, or 2)
        anomaly_type_tensor = tensor.cast(ops.floor(ops.multiply(tensor.random_uniform(()), 3.0)), tensor.int32)
        anomaly_type = tensor.item(anomaly_type_tensor)  # Get as Python scalar
        
        if anomaly_type == 0:
            # Spike anomaly
            # Generate random feature index
            feature_idx_tensor = tensor.cast(ops.floor(ops.multiply(tensor.random_uniform(()), n_features)), tensor.int32)
            feature_idx = tensor.item(feature_idx_tensor)  # Get as Python scalar
            
            # Generate random spike value
            spike_value = tensor.random_uniform((), minval=3.0, maxval=5.0)
            
            # Get the current value
            current_value = normal_tensor[idx, feature_idx]
            
            # Add the spike
            updated_value = ops.add(current_value, spike_value)
            
            # Update the tensor
            normal_tensor = tensor.index_update(normal_tensor, tensor.index[idx, feature_idx], updated_value)
            
        elif anomaly_type == 1:
            # Correlation anomaly
            # Generate random values
            random_values = tensor.random_normal((n_features,), mean=0.0, stddev=1.0)
            
            # Update the tensor
            normal_tensor = tensor.index_update(normal_tensor, tensor.index[idx, :], random_values)
            
        else:
            # Collective anomaly
            # Generate random values
            random_values = tensor.random_uniform((n_features,), minval=2.0, maxval=3.0)
            
            # Get the current row
            current_row = normal_tensor[idx, :]
            
            # Add the random values
            updated_row = ops.add(current_row, random_values)
            
            # Update the tensor
            normal_tensor = tensor.index_update(normal_tensor, tensor.index[idx, :], updated_row)
    
    # Convert back to numpy
    normal_data = tensor.to_numpy(normal_tensor)
    
    # Create DataFrame
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(normal_data, columns=columns)
    
    # Add timestamp column
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='5min')
    df['timestamp'] = timestamps
    
    # Add anomaly label
    df['anomaly'] = 0
    df.loc[anomaly_indices, 'anomaly'] = 1
    
    return df


def save_to_csv(df, filename='telemetry_data.csv'):
    """Save DataFrame to CSV file."""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename


def main():
    """Main function to demonstrate RBM-based anomaly detection."""
    print("RBM-based Anomaly Detection Example")
    print("===================================")
    
    # Create output directories if they don't exist
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/animations', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # Set random seed for reproducibility
    from ember_ml.nn import tensor
    from ember_ml import ops
    tensor.set_seed(42)
    
    # Generate synthetic telemetry data
    print("\nGenerating synthetic telemetry data...")
    telemetry_df = generate_telemetry_data(n_samples=1000, n_features=10, anomaly_fraction=0.05)
    print(f"Generated {len(telemetry_df)} samples with {telemetry_df['anomaly'].sum()} anomalies")
    
    # Save data to CSV
    csv_file = save_to_csv(telemetry_df)
    
    # Load data using GenericCSVLoader
    print("\nLoading data using GenericCSVLoader...")
    loader = GenericCSVLoader()
    df = loader.load_csv(csv_file)
    
    # Detect column types
    print("\nDetecting column types...")
    detector = GenericTypeDetector()
    column_types = detector.detect_column_types(df)
    
    # Print column types
    for type_name, cols in column_types.items():
        print(f"{type_name.capitalize()} columns: {cols}")
    
    # Engineer features
    print("\nEngineering features...")
    engineer = GenericFeatureEngineer()
    df_engineered = engineer.engineer_features(df, column_types)
    
    # Get numeric features for anomaly detection
    numeric_features = column_types.get('numeric', [])
    numeric_features = [col for col in numeric_features if col != 'anomaly']  # Exclude anomaly label
    
    if not numeric_features:
        print("No numeric features available for anomaly detection")
        return
    
    # Extract features
    features_df = df_engineered[numeric_features]
    
    # Split data into normal and anomalous
    normal_indices = df_engineered['anomaly'] == 0
    anomaly_indices = df_engineered['anomaly'] == 1
    
    normal_features = features_df[normal_indices].values
    anomaly_features = features_df[anomaly_indices].values
    
    # Split normal data into training and validation sets
    n_normal = len(normal_features)
    n_train = tensor.cast(ops.multiply(tensor.convert_to_tensor(n_normal), 0.8), tensor.int32)
    n_train_np = tensor.item(n_train)  # Get as Python scalar
    
    # Convert to tensor for proper slicing
    normal_features_tensor = tensor.convert_to_tensor(normal_features)
    train_features = tensor.to_numpy(tensor.slice_tensor(normal_features_tensor, [0, 0], [n_train_np, -1]))
    val_features = tensor.to_numpy(tensor.slice_tensor(normal_features_tensor, [n_train_np, 0], [-1, -1]))
    
    # Initialize RBM-based anomaly detector
    print("\nInitializing RBM-based anomaly detector...")
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
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"outputs/models/rbm_anomaly_detector_{timestamp}"
    detector.save(model_path)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    
    # Combine validation and anomaly data for testing
    # Convert to tensors for concatenation
    val_features_tensor = tensor.convert_to_tensor(val_features)
    anomaly_features_tensor = tensor.convert_to_tensor(anomaly_features)
    
    # Concatenate along the first axis (equivalent to vstack)
    test_features_tensor = ops.concatenate([val_features_tensor, anomaly_features_tensor], axis=0)
    test_features = tensor.to_numpy(test_features_tensor)
    
    # Create label tensors
    val_labels = tensor.zeros((tensor.shape(val_features_tensor)[0],))
    anomaly_labels = tensor.ones((tensor.shape(anomaly_features_tensor)[0],))
    
    # Concatenate labels (equivalent to hstack)
    test_labels_tensor = ops.concatenate([val_labels, anomaly_labels], axis=0)
    test_labels = tensor.to_numpy(test_labels_tensor)
    
    # Predict anomalies
    predicted_anomalies = detector.predict(test_features)
    anomaly_scores = detector.anomaly_score(test_features)
    
    # Compute metrics using tensor operations
    pred_tensor = tensor.convert_to_tensor(predicted_anomalies)
    labels_tensor = tensor.convert_to_tensor(test_labels)
    
    # Compute metrics using tensor operations
    true_positives = stats.sum(ops.logical_and(ops.equal(pred_tensor, 1), ops.equal(labels_tensor, 1)))
    false_positives = stats.sum(ops.logical_and(ops.equal(pred_tensor, 1), ops.equal(labels_tensor, 0)))
    true_negatives = stats.sum(ops.logical_and(ops.equal(pred_tensor, 0), ops.equal(labels_tensor, 0)))
    false_negatives = stats.sum(ops.logical_and(ops.equal(pred_tensor, 0), ops.equal(labels_tensor, 1)))
    
    # Calculate metrics using ops
    precision_denom = ops.add(true_positives, false_positives)
    precision = ops.divide(true_positives, precision_denom) if tensor.item(precision_denom) > 0 else 0
    
    recall_denom = ops.add(true_positives, false_negatives)
    recall = ops.divide(true_positives, recall_denom) if tensor.item(recall_denom) > 0 else 0
    
    precision_recall_sum = ops.add(precision, recall)
    f1_score = ops.multiply(2.0, ops.divide(ops.multiply(precision, recall), precision_recall_sum)) if tensor.item(precision_recall_sum) > 0 else 0
    
    # Convert to Python scalars for printing
    precision_val = tensor.item(precision) if hasattr(precision, 'shape') else precision
    recall_val = tensor.item(recall) if hasattr(recall, 'shape') else recall
    f1_score_val = tensor.item(f1_score) if hasattr(f1_score, 'shape') else f1_score
    
    print(f"Precision: {precision_val:.4f}")
    print(f"Recall: {recall_val:.4f}")
    print(f"F1 Score: {f1_score_val:.4f}")
    
    # Initialize visualizer
    visualizer = RBMVisualizer()
    
    # Plot training curve
    print("\nPlotting training curve...")
    visualizer.plot_training_curve(detector.rbm, show=True)
    
    # Plot weight matrix
    print("\nPlotting weight matrix...")
    visualizer.plot_weight_matrix(detector.rbm, show=True)
    
    # Plot reconstructions
    print("\nPlotting reconstructions...")
    visualizer.plot_reconstructions(detector.rbm, test_features[:5], show=True)
    
    # Plot anomaly scores
    print("\nPlotting anomaly scores...")
    plt.figure(figsize=(10, 6))
    
    # Convert to tensors for filtering
    scores_tensor = tensor.convert_to_tensor(anomaly_scores)
    labels_tensor = tensor.convert_to_tensor(test_labels)
    
    # Filter scores for normal and anomaly samples
    normal_mask = ops.equal(labels_tensor, 0)
    anomaly_mask = ops.equal(labels_tensor, 1)
    
    normal_scores = tensor.to_numpy(tensor.boolean_mask(scores_tensor, normal_mask))
    anomaly_scores_filtered = tensor.to_numpy(tensor.boolean_mask(scores_tensor, anomaly_mask))
    
    # Plot histograms
    plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal')
    plt.hist(anomaly_scores_filtered, bins=30, alpha=0.7, label='Anomaly')
    plt.axvline(detector.anomaly_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"outputs/plots/anomaly_scores_{timestamp}.png"
    plt.savefig(plot_path)
    plt.show()
    
    # Animate weight evolution
    print("\nAnimating weight evolution...")
    visualizer.animate_weight_evolution(detector.rbm, show=True)
    
    # Animate dreaming
    print("\nAnimating dreaming process...")
    visualizer.animate_dreaming(detector.rbm, n_steps=50, show=True)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()