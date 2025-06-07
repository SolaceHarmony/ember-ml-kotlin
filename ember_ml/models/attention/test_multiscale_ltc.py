import matplotlib.pyplot as plt
import ember_ml as nl
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.attention.multiscale_ltc import (
    TemporalStrideProcessor,
    build_multiscale_ltc_model,
    visualize_feature_extraction,
    visualize_multiscale_dynamics
)
from ember_ml.nn.modules import AutoNCP # Updated import path


def generate_synthetic_data(num_samples=1000, num_features=10, seed=42):
    """Generate synthetic data for testing.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features to generate
        seed: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    nl.set_seed(seed)
    
    # Generate features with different frequencies
    t = tensor.linspace(0, 4 * ops.pi, num_samples)
    X = tensor.zeros((num_samples, num_features))
    
    for i in range(num_features):
        freq = 0.5 + i * 0.2  # Different frequency for each feature
        phase = ops.random.uniform(0, 2 * ops.pi)  # Random phase
        X = ops.index_update(X, ops.index[:, i],
                           ops.sin(freq * t + phase) + 0.1 * ops.random.normal(shape=(num_samples,)))
    
    # Generate target as a non-linear function of features
    y = ops.sin(X[:, 0] * X[:, 1]) + ops.cos(X[:, 2] + X[:, 3]) + 0.1 * ops.random.normal(shape=(num_samples,))
    y = tensor.reshape(y, (-1, 1))
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def test_temporal_stride_processor():
    """Test the TemporalStrideProcessor class."""
    print("\n=== Testing TemporalStrideProcessor ===")
    
    # Generate synthetic data
    X_train, _, _, _, _, _ = generate_synthetic_data(num_samples=100, num_features=5)
    
    # Create a TemporalStrideProcessor
    processor = TemporalStrideProcessor(
        window_size=5,
        stride_perspectives=[1, 2, 3],
        pca_components=2
    )
    
    # Process the data
    perspectives = processor.process_batch(X_train)
    
    # Print the shapes of the processed data
    for stride, data in perspectives.items():
        print(f"Stride {stride}: {data.shape}")
    
    return perspectives


def test_build_multiscale_ltc_model():
    """Test the build_multiscale_ltc_model function."""
    print("\n=== Testing build_multiscale_ltc_model ===")
    
    # Define input dimensions for each stride
    input_dims = {
        1: 50,
        2: 30,
        3: 20
    }
    
    # Build the model
    model = build_multiscale_ltc_model(
        input_dims=input_dims,
        output_dim=1,
        hidden_units=16,
        dropout_rate=0.2
    )
    
    # Print the model summary
    model.summary()
    
    return model


def test_end_to_end():
    """Test the entire pipeline end-to-end."""
    print("\n=== Testing End-to-End Pipeline ===")
    
    # Generate synthetic data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_synthetic_data(
        num_samples=500,
        num_features=10
    )
    
    # Define stride perspectives
    stride_perspectives = [1, 2, 3]
    window_size = 5
    pca_components = 2
    
    # Process the data
    processor = TemporalStrideProcessor(
        window_size=window_size,
        stride_perspectives=stride_perspectives,
        pca_components=pca_components
    )
    
    train_perspectives = processor.process_batch(X_train)
    val_perspectives = processor.process_batch(X_val)
    test_perspectives = processor.process_batch(X_test)
    
    # Convert to tensors
    train_inputs = tuple(tensor.convert_to_tensor(data, dtype='float32') for data in train_perspectives.values())
    val_inputs = tuple(tensor.convert_to_tensor(data, dtype='float32') for data in val_perspectives.values())
    test_inputs = tuple(tensor.convert_to_tensor(data, dtype='float32') for data in test_perspectives.values())
    
    train_y = tensor.convert_to_tensor(y_train, dtype='float32')
    val_y = tensor.convert_to_tensor(y_val, dtype='float32')
    test_y = tensor.convert_to_tensor(y_test, dtype='float32')
    
    # Build the model
    input_dims = {s: train_perspectives[s].shape[1] for s in train_perspectives.keys()}
    model = build_multiscale_ltc_model(
        input_dims=input_dims,
        output_dim=1,
        hidden_units=16,
        dropout_rate=0.2
    )
    
    # Train the model (with a small number of epochs for testing)
    history = model.fit(
        train_inputs,
        train_y,
        validation_data=(val_inputs, val_y),
        batch_size=16,
        epochs=5,
        verbose=1
    )
    
    # Evaluate on the test set
    loss, mae = model.evaluate(test_inputs, test_y, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Create metadata for visualization
    metadata = {
        'feature_counts': {
            'original': X_train.shape[1],
            'numeric': X_train.shape[1],
            'categorical': 0,
        },
        'temporal_compression': {
            stride: {
                "input_dim": train_perspectives[stride].shape[0] * window_size,
                "output_dim": train_perspectives[stride].shape[1],
                "compression_ratio": (train_perspectives[stride].shape[0] * window_size)/train_perspectives[stride].shape[1],
            }
            for stride in stride_perspectives if stride in train_perspectives
        },
        'dimensional_evolution': [
            {"stage": f"stride_{s}", "dimension": train_perspectives[s].shape[1]} for s in stride_perspectives if s in train_perspectives
        ]
    }
    
    # Visualize feature extraction
    feature_fig = visualize_feature_extraction(metadata)
    plt.savefig('feature_extraction_test.png')
    print("Feature extraction visualization saved to 'feature_extraction_test.png'")
    
    # Visualize multiscale dynamics
    dynamics_fig = visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives)
    plt.savefig('multiscale_dynamics_test.png')
    print("Multiscale dynamics visualization saved to 'multiscale_dynamics_test.png'")
    
    return history


if __name__ == "__main__":
    print("Testing multiscale_ltc.py module...")
    
    # Set the backend to numpy for testing
    nl.backend.set_backend('numpy')
    
    # Test the TemporalStrideProcessor
    perspectives = test_temporal_stride_processor()
    
    # Test building the model
    model = test_build_multiscale_ltc_model()
    
    # Test the end-to-end pipeline
    history = test_end_to_end()
    
    print("\nAll tests completed successfully!")