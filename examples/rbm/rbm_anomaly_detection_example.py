"""
RBM-based Anomaly Detection Example

This script demonstrates how to use the RBM-based anomaly detector
with the generic feature extraction library to detect anomalies in data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import modules from ember_ml
from ember_ml.nn.features.generic_feature_engineer import (

    GenericFeatureEngineer
)
from ember_ml.nn.features.generic_type_detector import GenericTypeDetector
from ember_ml.nn.features.generic_csv_loader import GenericCSVLoader
from ember_ml.nn.features.temporal_processor import TemporalStrideProcessor
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
    # Generate normal data
    from ember_ml.nn import tensor
    normal_data = tensor.random_normal(0, 1, (n_samples, n_features))
    
    # Add some correlations between features
    for i in range(1, n_features):
        normal_data[:, i] = normal_data[:, i] * 0.5 + normal_data[:, 0] * 0.5
    
    # Add some temporal patterns
    for i in range(n_samples):
        normal_data[i, :] += ops.sin(i / 50) * 0.5
    
    # Generate anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = ops.random_choice(n_samples, n_anomalies, replace=False)
    
    # Create different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.randint(0, 3)
        
        if anomaly_type == 0:
            # Spike anomaly
            feature_idx = np.random.randint(0, n_features)
            normal_data[idx, feature_idx] += tensor.random_uniform(3, 5)
        elif anomaly_type == 1:
            # Correlation anomaly
            normal_data[idx, :] = tensor.random_normal(0, 1, n_features)
        else:
            # Collective anomaly
            normal_data[idx, :] += tensor.random_uniform(2, 3, n_features)
    
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
    np.random.seed(42)
    
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
    # No need to exclude 'anomaly' as it's now in categorical columns, not numeric
    
    if not numeric_features:
        print("No numeric features available for anomaly detection")
        return
    
    # Extract features
    features_df = df_engineered[numeric_features]
    # After one-hot encoding, 'anomaly' column is replaced with 'anomaly_0' and 'anomaly_1' columns
    # Find the anomaly columns
    anomaly_cols = [col for col in df_engineered.columns if col.startswith('anomaly_')]
    
    if not anomaly_cols:
        print("Error: Could not find anomaly columns after feature engineering")
        return
    
    print(f"Found anomaly columns: {anomaly_cols}")
    
    # Split data into normal and anomalous based on one-hot encoded columns
    # 'anomaly_0' corresponds to normal samples (0)
    # 'anomaly_1' corresponds to anomalous samples (1)
    normal_indices = df_engineered['anomaly_0'] == 1.0
    anomaly_indices = df_engineered['anomaly_1'] == 1.0
    
    normal_features = features_df[normal_indices].values
    anomaly_features = features_df[anomaly_indices].values
    
    # Split normal data into training and validation sets
    n_normal = len(normal_features)
    n_train = int(n_normal * 0.8)
    
    train_features = normal_features[:n_train]
    val_features = normal_features[n_train:]
    
    # Initialize RBM-based anomaly detector
    print("\nInitializing RBM-based anomaly detector...")
    detector = RBMBasedAnomalyDetector(
        n_hidden=20,  # Increased from 5 to 20 for a deeper network
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
        epochs=100,  # Increased from 30 to 100 to ensure convergence
        k=1,
        early_stopping_patience=10,  # Increased patience for better convergence
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
    test_features = tensor.vstack([val_features, anomaly_features])
    test_labels = np.hstack([
        tensor.zeros(len(val_features)),
        tensor.ones(len(anomaly_features))
    ])
    
    # Predict anomalies
    predicted_anomalies = detector.predict(test_features)
    anomaly_scores = detector.anomaly_score(test_features)
    
    # Compute metrics
    true_positives = stats.sum((predicted_anomalies == 1) & (test_labels == 1))
    false_positives = stats.sum((predicted_anomalies == 1) & (test_labels == 0))
    true_negatives = stats.sum((predicted_anomalies == 0) & (test_labels == 0))
    false_negatives = stats.sum((predicted_anomalies == 0) & (test_labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Initialize visualizer
    visualizer = RBMVisualizer()
    # Plot convergence analysis
    print("\nPlotting convergence analysis...")
    visualizer.plot_convergence(detector.rbm, show=True)
    
    # Plot training curve
    print("\nPlotting training curve...")
    visualizer.plot_training_curve(detector.rbm, show=True)
    
    # Plot weight matrix
    print("\nPlotting weight matrix...")
    visualizer.plot_weight_matrix(detector.rbm, show=True)
    visualizer.plot_weight_matrix(detector.rbm, show=True)
    
    # Plot reconstructions
    print("\nPlotting reconstructions...")
    visualizer.plot_reconstructions(detector.rbm, test_features[:5], show=True)
    
    # Plot anomaly scores
    print("\nPlotting anomaly scores...")
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores[test_labels == 0], bins=30, alpha=0.7, label='Normal')
    plt.hist(anomaly_scores[test_labels == 1], bins=30, alpha=0.7, label='Anomaly')
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
    
    # Categorize anomalies
    print("\nCategorizing anomalies...")
    category_labels, cluster_info = detector.categorize_anomalies(
        test_features,
        anomaly_flags=predicted_anomalies,
        max_clusters=5,
        min_samples_per_cluster=3
    )
    
    # Count samples in each category
    unique_categories, category_counts = np.unique(category_labels[category_labels >= 0], return_counts=True)
    print(f"Found {len(unique_categories)} anomaly categories:")
    for cat_id, count in zip(unique_categories, category_counts):
        print(f"  Category {cat_id}: {count} samples")
    
    # Plot anomaly categories
    print("\nPlotting anomaly categories...")
    visualizer.plot_anomaly_categories(
        detector.rbm,
        test_features,
        category_labels,
        cluster_info,
        feature_names=numeric_features,
        show=True
    )
    
    # Get normal data for comparison
    normal_indices = ops.where(predicted_anomalies == 0)[0]
    normal_data = test_features[normal_indices]
    
    # Plot detailed statistical distributions for each category
    print("\nPlotting anomaly category statistics...")
    visualizer.plot_anomaly_category_statistics(
        test_features,
        normal_data,
        category_labels,
        cluster_info,
        feature_names=numeric_features,
        show=True
    )
    
    # Generate pandas-friendly tables with category statistics
    print("\nGenerating category statistics tables...")
    tables_dir = "outputs/tables"
    os.makedirs(tables_dir, exist_ok=True)
    
    category_dfs = visualizer.generate_category_statistics_tables(
        test_features,
        normal_data,
        category_labels,
        cluster_info,
        feature_names=numeric_features,
        save_dir=tables_dir,
        save=True
    )
    
    # Print summary information from the tables
    print("\nCategory Statistics Summary:")
    print(category_dfs['summary'].to_string(index=False))
    
    # For each category, print the top 3 most anomalous features by Z-score
    for cat_id, df in category_dfs.items():
        if cat_id != 'summary':
            category_num = cat_id.split('_')[1]
            print(f"\nTop features for Category {category_num} (by Z-score):")
            print(df[['Feature', 'Z_Score', 'Category_Mean', 'Normal_Mean']].head(3).to_string(index=False))
    
    # Plot feature-hidden correlations
    print("\nPlotting feature-hidden correlations...")
    visualizer.plot_feature_hidden_correlations(
        detector.rbm,
        test_features,
        feature_names=numeric_features,
        show=True
    )
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()