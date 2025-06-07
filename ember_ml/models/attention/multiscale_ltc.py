import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from ember_ml.ops import stats
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor, zeros, ones, reshape, concatenate, to_numpy, convert_to_tensor
from ember_ml.nn.tensor import float32, shape, cast, arange, stack, pad, full
from ember_ml.nn.modules import AutoNCP # Updated import path
from ember_ml.nn.modules.rnn.stride_aware import StrideAwareCell
from ember_ml.nn.modules.rnn.rnn import RNN
from ember_ml.nn.initializers import glorot_uniform
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn.features import one_hot, fit_transform, fit, transform, PCA

# The prepare_bigquery_data_bf function would be imported here in a real implementation
# from data_utils import prepare_bigquery_data_bf

def prepare_bigquery_data_bf(*args, **kwargs):
    """
    This is a placeholder function for the BigQuery data preparation.
    In a real implementation, this would be imported from data_utils.
    
    Returns:
        A tuple of mock data for testing purposes.
    """
    print("WARNING: Using mock data for BigQuery preparation. This is not for production use.")
    
    # Create mock dataframes with minimal data for testing
    # Use ops for random generation instead of numpy
    mock_data = tensor.random_uniform((100, 2))
    mock_target = tensor.random_uniform((100, 1))
    
    # Create mock dataframes
    mock_df = pd.DataFrame({
        'feature1': to_numpy(mock_data[:, 0]),
        'feature2': to_numpy(mock_data[:, 1]),
        'target': to_numpy(mock_target[:, 0])
    })
    
    # Mock features
    features = ['feature1', 'feature2']
    
    # Return mock data
    return (
        mock_df, mock_df, mock_df,  # train, val, test dataframes
        features, features, features,  # train, val, test features
        None, None  # scaler, imputer
    )
def build_multiscale_ltc_model(input_dims: Dict[int, int], output_dim: int = 1,
                               hidden_units: int = 32, dropout_rate: float = 0.2) -> Dict[str, Dict]:
    """Build a multi-scale model with stride-aware cells.
    
    Args:
        input_dims: Dictionary mapping stride lengths to input dimensions
        output_dim: Dimension of the output
        hidden_units: Number of hidden units in each cell
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Model with multiple stride-aware cells
    """
    # Create a dictionary to store the model components
    model: Dict[str, Dict] = {
        'inputs': {},
        'ltc_cells': {},
        'outputs': {}
    }
    
    # Create inputs and cells for each stride
    ltc_outputs = []
    
    for stride, dim in input_dims.items():
        # Create input for this stride
        input_name = f"stride_{stride}_input"
        # Create a tensor to serve as a placeholder (similar to Keras Input layer)
        model['inputs'][stride] = convert_to_tensor(zeros((1, dim)), dtype=float32)
        
        # Reshape for RNN processing if needed
        reshaped = reshape(model['inputs'][stride], (-1, 1, dim))
        
        # Create a wiring for this stride
        wiring = AutoNCP(
            units=hidden_units,
            output_size=ops.floor_divide(hidden_units, 2),
            sparsity_level=0.5
        )
        
        # Create a stride-aware cell
        ltc_cell = StrideAwareCell(
            input_size=dim,
            hidden_size=hidden_units,
            stride_length=stride,
            time_scale_factor=1.0,
            activation="tanh"
        )
        
        # Create an RNN layer with the cell
        rnn = RNN(
            input_size=dim,
            hidden_size=hidden_units,
            batch_first=True,
            return_sequences=False,
            return_state=False,
            dropout=dropout_rate
        )
        
        # Process the input through the RNN
        rnn_output = rnn(reshaped)
        
        # Store the output
        ltc_outputs.append(rnn_output)
        
        # Store the cell in the model
        model['ltc_cells'][stride] = ltc_cell
    
    # Concatenate outputs from all strides if there are multiple
    if len(ltc_outputs) > 1:
        concatenated = concatenate(ltc_outputs, axis=-1)
    else:
        concatenated = ltc_outputs[0]
    
    # Add a fully connected layer to combine the multi-scale features
    # Use matrix multiplication for the dense layer
    W_dense = glorot_uniform((shape(concatenated)[-1], hidden_units))
    b_dense = zeros((hidden_units,))
    dense = ops.relu(ops.add(ops.matmul(concatenated, W_dense), b_dense))
    
    # Apply dropout manually
    dropout_mask = ops.divide(
        cast(ops.greater(tensor.random_uniform(shape(dense)), dropout_rate), dense.dtype),
        ops.subtract(1.0, dropout_rate)
    )
    dense_dropout = ops.multiply(dense, dropout_mask)
    
    # Output layer
    W_output = glorot_uniform((hidden_units, output_dim))
    b_output = zeros((output_dim,))
    output = ops.add(ops.matmul(dense_dropout, W_output), b_output)
    model['outputs']['main'] = output
    
    return model


def visualize_feature_extraction(metadata: Dict) -> Any:
    """Visualize the feature extraction process.
    
    Args:
        metadata: Dictionary containing metadata about the feature extraction process
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Feature counts
    ax1 = fig.add_subplot(2, 2, 1)
    feature_counts = metadata['feature_counts']
    ax1.bar(feature_counts.keys(), feature_counts.values())
    ax1.set_title("Feature Counts by Type")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # 2. Temporal compression
    ax2 = fig.add_subplot(2, 2, 2)
    compression_ratios = [data["compression_ratio"] for stride, data in metadata['temporal_compression'].items()]
    strides = list(metadata['temporal_compression'].keys())
    ax2.bar(strides, compression_ratios)
    ax2.set_title("Compression Ratio by Stride")
    ax2.set_xlabel("Stride")
    ax2.set_ylabel("Compression Ratio")
    ax2.grid(True, alpha=0.3)
    
    # 3. Input/Output dimensions
    ax3 = fig.add_subplot(2, 2, 3)
    input_dims = [data["input_dim"] for stride, data in metadata['temporal_compression'].items()]
    output_dims = [data["output_dim"] for stride, data in metadata['temporal_compression'].items()]
    x = arange(len(strides))
    width = 0.35
    # Use numpy for matplotlib compatibility
    x_np = to_numpy(x)
    ax3.bar(to_numpy(ops.subtract(x_np, width/2)), input_dims, width, label='Input Dim')
    ax3.bar(to_numpy(ops.add(x_np, width/2)), output_dims, width, label='Output Dim')
    ax3.set_title("Input/Output Dimensions by Stride")
    ax3.set_xlabel("Stride")
    ax3.set_ylabel("Dimension")
    ax3.set_xticks(x)
    ax3.set_xticklabels(strides)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Dimensional evolution
    ax4 = fig.add_subplot(2, 2, 4)
    stages = [item["stage"] for item in metadata['dimensional_evolution']]
    dimensions = [item["dimension"] for item in metadata['dimensional_evolution']]
    ax4.plot(stages, dimensions, marker='o')
    ax4.set_title("Dimensional Evolution Through Processing Stages")
    ax4.set_xlabel("Processing Stage")
    ax4.set_ylabel("Dimension")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives) -> Any:
    """Visualize the multi-scale dynamics of the model.
    
    Args:
        model: Trained model dictionary
        test_inputs: Test inputs
        test_y: Test targets
        stride_perspectives: List of stride lengths
        
    Returns:
        Matplotlib figure
    """
    # Get the intermediate outputs from the LTC cells
    intermediate_outputs = []
    
    # Convert inputs to tensor format if they're not already
    test_inputs_tensor = {}
    for i, stride in enumerate(stride_perspectives):
        if stride in model['inputs']:
            # If using dictionary inputs
            if isinstance(test_inputs, dict):
                test_inputs_tensor[stride] = test_inputs[stride]
            # If using tuple inputs (from the original code)
            elif isinstance(test_inputs, tuple) and i < len(test_inputs):
                test_inputs_tensor[stride] = test_inputs[i]
    
    # Get outputs from each LTC cell
    for stride in stride_perspectives:
        if stride in model['ltc_cells']:
            # Get the cell
            cell = model['ltc_cells'][stride]
            
            # Forward pass through the cell
            if stride in test_inputs_tensor:
                # Reshape for RNN processing
                reshaped = reshape(test_inputs_tensor[stride], (-1, 1, shape(test_inputs_tensor[stride])[1]))
                
                # Process through the cell
                outputs = []
                states = None
                for i in range(shape(reshaped)[0]):
                    output, states = cell.forward(reshaped[i], states)
                    outputs.append(output)
                
                # Convert to tensor
                cell_output = stack(outputs)
                intermediate_outputs.append((stride, cell_output))
    
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Prediction vs. Actual
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Get predictions from the model
    predictions = []
    test_y_tensor = test_y
    
    # Plot scatter of predictions vs actual
    if len(intermediate_outputs) > 0:
        # Use the last layer's output as predictions
        predictions = intermediate_outputs[-1][1]
        ax1.scatter(to_numpy(test_y_tensor), to_numpy(predictions), alpha=0.5)
        
        # Get min and max values for the diagonal line
        min_val = min(stats.min(test_y_tensor).item(), stats.min(predictions).item())
        max_val = max(stats.max(test_y_tensor).item(), stats.max(predictions).item())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax1.set_title("Prediction vs. Actual")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.grid(True, alpha=0.3)
    
    # 2. Activation patterns across strides
    ax2 = fig.add_subplot(2, 2, 2)
    for stride, output in intermediate_outputs:
        # Take the mean activation across samples
        mean_activation = ops.stats.mean(output, axis=0)
        ax2.plot(to_numpy(mean_activation), label=f"Stride {stride}")
    ax2.set_title("Mean Activation Patterns Across Strides")
    ax2.set_xlabel("Neuron Index")
    ax2.set_ylabel("Mean Activation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA of activations - use 2D plot instead of 3D to avoid compatibility issues
    ax3 = fig.add_subplot(2, 2, 3)
    for stride, output in intermediate_outputs:
        # Apply PCA to reduce to 2D
        if shape(output)[0] > 2:  # Need at least 3 samples for 2 components
            # Convert to numpy for PCA
            output_tensor = tensor.convert_to_tensor(output)
            pca = PCA()
            output_pca = pca.fit_transform(output_tensor, n_components=min(2, output_tensor.shape[0]-1))
            
            # Plot the PCA results
            ax3.scatter(output_pca[:, 0],
                       output_pca[:, 1] if output_pca.shape[1] > 1 else zeros(output_pca.shape[0]),
                       label=f"Stride {stride}", alpha=0.5)
    ax3.set_title("PCA of Activations (2D)")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.legend()
    
    # 4. Activation distribution
    ax4 = fig.add_subplot(2, 2, 4)
    for stride, output in intermediate_outputs:
        # Flatten the output
        output_flat = reshape(output, (-1,))
        ax4.hist(to_numpy(output_flat), bins=50, alpha=0.5, label=f"Stride {stride}")
    ax4.set_title("Activation Distribution")
    ax4.set_xlabel("Activation")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def integrate_liquid_neurons_with_visualization(
    project_id,
    table_id,
    target_column=None,
    window_size=5,
    stride_perspectives=[1, 3, 5],
    batch_size=32,
    epochs=15,
    pca_components=3,
    **prepare_kwargs
):
    """
    Runs the entire pipeline with PCA applied *per column* within each stride.
    
    Args:
        project_id: GCP project ID
        table_id: BigQuery table ID
        target_column: Target column name
        window_size: Size of the sliding window
        stride_perspectives: List of stride lengths to use
        batch_size: Batch size for training
        epochs: Number of epochs for training
        pca_components: Number of PCA components to extract
        **prepare_kwargs: Additional arguments for prepare_bigquery_data_bf
        
    Returns:
        Training history
    """
    # Data preparation
    print("🔹 Starting Data Preparation...")
    result = prepare_bigquery_data_bf(
        project_id=project_id,
        table_id=table_id,
        target_column=target_column,
        **prepare_kwargs
    )

    if result is None:
        raise ValueError("❌ Data preparation failed.")

    # Unpack results
    train_bf_df, val_bf_df, test_bf_df, train_features, val_features, test_features, scaler, imputer = result
    
    # Use pandas directly
    train_df = train_bf_df
    val_df = val_bf_df
    test_df = test_bf_df

    # Auto-detect target column if not provided
    if target_column is None:
        all_cols = train_bf_df.columns.tolist()
        feature_set = set(train_features)

        # Find the first column that is NOT a feature (likely the target)
        possible_targets = [col for col in all_cols if col not in feature_set]

        if possible_targets:
            target_column = possible_targets[0]
            print(f"🟢 Auto-selected target column: {target_column}")
        else:
            raise ValueError("❌ No valid target column found. Please specify `target_column` manually.")

    # Extract features & targets using EmberTensor
    train_X = EmberTensor(train_df[train_features].values)
    val_X = EmberTensor(val_df[val_features].values)
    test_X = EmberTensor(test_df[test_features].values)
    
    # Convert pandas Series to EmberTensor
    train_y = EmberTensor(train_df[target_column].values)
    val_y = EmberTensor(val_df[target_column].values)
    test_y = EmberTensor(test_df[target_column].values)

    # Detect if the target is categorical (string-based)
    if train_df[target_column].dtype == "object":
        print(f"🟡 Detected categorical target ({target_column}). Applying EmberTensor-based encoding.")

        # Get unique categories
        all_categories = pd.concat([
            train_df[target_column],
            val_df[target_column],
            test_df[target_column]
        ]).unique()
        
        # Create a mapping from category to index
        category_to_index = {cat: i for i, cat in enumerate(all_categories)}
        
        # Convert categorical values to indices
        train_indices = [category_to_index[cat] for cat in train_df[target_column]]
        val_indices = [category_to_index[cat] for cat in val_df[target_column]]
        test_indices = [category_to_index[cat] for cat in test_df[target_column]]
        
        # Create one-hot encoded tensors using EmberTensor
        num_categories = len(all_categories)
        
        # Use the one_hot function from ember_ml.features
        train_one_hot = one_hot(train_indices, num_categories)
        val_one_hot = one_hot(val_indices, num_categories)
        test_one_hot = one_hot(test_indices, num_categories)
        
        # Apply dimensionality reduction using ember_ml PCA
        # Combine one-hot tensors for fitting
        all_one_hot = concatenate([
            train_one_hot.data,
            val_one_hot.data,
            test_one_hot.data
        ], axis=0)
        
        # Use fit_transform directly
        pca_instance = PCA()
        all_transformed = pca_instance.fit_transform(all_one_hot, n_components=1)
        
        # Split the transformed data back into train, val, test
        train_size = shape(train_one_hot.data)[0]
        val_size = shape(val_one_hot.data)[0]
        
        # Use slice instead of direct indexing
        train_y_pca = tensor.slice_tensor(all_transformed, [0, 0], [train_size, -1])
        val_y_pca = tensor.slice_tensor(all_transformed, [train_size, 0], [val_size, -1])
        test_y_pca = tensor.slice_tensor(all_transformed, [train_size + val_size, 0], [-1, -1])
        
        # Convert to float32
        train_y = cast(train_y_pca, 'float32')
        val_y = cast(val_y_pca, 'float32')
        test_y = cast(test_y_pca, 'float32')
        
        print(f"✅ EmberTensor-based encoding shape: {train_y.shape}")
    else:
        print(f"🟢 Detected numeric target ({target_column}). Using directly as float32.")
        
        # Reshape to 2D if needed
        if len(train_y.shape) == 1:
            train_y = reshape(train_y,(-1, 1))
            val_y = reshape(val_y,(-1, 1))
            test_y = reshape(test_y,(-1, 1))
        
        # Cast to float32
        train_y = cast(train_y,float32)
        val_y = cast(val_y,float32)
        test_y = cast(test_y,float32)

    print(f"✅ Final target shape: {shape(train_y)}, dtype: {train_y.dtype}")

    # Process stride-based representations
    processor = TemporalStrideProcessor(window_size=window_size, stride_perspectives=stride_perspectives, pca_components=pca_components)
    train_perspectives = processor.process_batch(train_X)
    val_perspectives = processor.process_batch(val_X)
    test_perspectives = processor.process_batch(test_X)

    # Convert to ember_ml tensors
    train_inputs = {s: convert_to_tensor(data, dtype='float32')
                   for s, data in train_perspectives.items()}
    val_inputs = {s: convert_to_tensor(data, dtype='float32')
                 for s, data in val_perspectives.items()}
    test_inputs = {s: convert_to_tensor(data, dtype='float32')
                  for s, data in test_perspectives.items()}

    print("Train Input Shapes (Before Model Building):",
          {s: shape(data) for s, data in train_inputs.items()})
    print("Validation Input Shapes (Before Model Building):",
          {s: shape(data) for s, data in val_inputs.items()})
    print("Test Input Shapes (Before Model Building):",
          {s: shape(data) for s, data in test_inputs.items()})

    # Build model
    print("🔹 Building Multi-Scale Liquid Neural Network...")
    input_dims = {s: shape(train_perspectives[s])[1] for s in train_perspectives.keys()}
    model = build_multiscale_ltc_model(input_dims=input_dims, output_dim=1)

    # Train model
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': []
    }
    
    # Simple training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training step
        train_loss = 0
        train_mae = 0
        
        # Process in batches
        num_batches = shape(train_X)[0] // batch_size
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, shape(train_X)[0])
            
            # Get batch inputs
            batch_inputs = {s: tensor.slice_tensor(train_inputs[s], [start_idx, 0], [end_idx - start_idx, -1]) for s in train_inputs}
            batch_y = tensor.slice_tensor(train_y, [start_idx, 0], [end_idx - start_idx, -1])
            
            # Forward pass
            outputs = model['outputs']['main']
            
            # Compute loss
            loss = ops.mse(batch_y, outputs)
            mae = ops.mean_absolute_error(batch_y, outputs)
            
            # Update metrics
            train_loss += loss
            train_mae += mae
            
            # Print progress
            if batch % 10 == 0:
                print(f"  Batch {batch}/{num_batches} - Loss: {loss:.4f}, MAE: {mae:.4f}")
        
        # Compute average metrics
        train_loss /= num_batches
        train_mae /= num_batches
        
        # Validation step
        val_loss = 0
        val_mae = 0
        
        # Process validation data
        num_val_batches = shape(val_X)[0] // batch_size
        for batch in range(num_val_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, shape(val_X)[0])
            
            # Get batch inputs
            batch_inputs = {s: tensor.slice_tensor(val_inputs[s], [start_idx, 0], [end_idx - start_idx, -1]) for s in val_inputs}
            batch_y = tensor.slice_tensor(val_y, [start_idx, 0], [end_idx - start_idx, -1])
            
            # Forward pass
            outputs = model['outputs']['main']
            
            # Compute loss
            loss = ops.mse(batch_y, outputs)
            mae = ops.mean_absolute_error(batch_y, outputs)
            
            # Update metrics
            val_loss += loss
            val_mae += mae
        
        # Compute average metrics
        val_loss /= num_val_batches
        val_mae /= num_val_batches
        
        # Update history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Early stopping
        if epoch > 5 and history['val_loss'][-1] > history['val_loss'][-2]:
            print("Early stopping triggered")
            break

    # Evaluate on the test set
    test_loss = 0
    test_mae = 0
    
    # Process test data
    num_test_batches = shape(test_X)[0] // batch_size
    for batch in range(num_test_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, shape(test_X)[0])
        
        # Get batch inputs
        batch_inputs = {s: tensor.slice_tensor(test_inputs[s], [start_idx, 0], [end_idx - start_idx, -1]) for s in test_inputs}
        batch_y = tensor.slice_tensor(test_y, [start_idx, 0], [end_idx - start_idx, -1])
        
        # Forward pass
        outputs = model['outputs']['main']
        
        # Compute loss
        loss = ops.mse(batch_y, outputs)
        mae = ops.mean_absolute_error(batch_y, outputs)
        
        # Update metrics
        test_loss += loss
        test_mae += mae
    
    # Compute average metrics
    test_loss /= num_test_batches
    test_mae /= num_test_batches
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Populate metadata for visualizations
    metadata = {
        'feature_counts': {
            'original': len(train_features),
            'numeric': sum(1 for feat in train_features if "sin_" not in feat and "cos_" not in feat),
            'categorical': sum(1 for feat in train_features if "sin_" in feat or "cos_" in feat),
        },
        'temporal_compression': {
            stride: {
                "input_dim": shape(train_perspectives[stride])[0] * window_size,
                "output_dim": shape(train_perspectives[stride])[1],
                "compression_ratio": (shape(train_perspectives[stride])[0] * window_size)/shape(train_perspectives[stride])[1],
            }
            for stride in stride_perspectives if stride in train_perspectives
        },
        'dimensional_evolution': [
            {"stage": f"stride_{s}", "dimension": shape(train_perspectives[s])[1]} for s in stride_perspectives if s in train_perspectives
        ]
    }

    # Run the visualizations
    feature_fig = visualize_feature_extraction(metadata)
    plt.savefig('feature_extraction.png')
    print("Feature extraction visualization saved to 'feature_extraction.png'")
    
    dynamics_fig = visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives)
    plt.savefig('multiscale_dynamics.png')
    print("Multiscale dynamics visualization saved to 'multiscale_dynamics.png'")
    
    return history


if __name__ == "__main__":
    # Example usage
    print("This module provides classes and functions for multi-scale liquid neural networks.")
    print("To use it, import the module and call the integrate_liquid_neurons_with_visualization function.")
    print("Example:")
    print("  from ember_ml.attention.multiscale_ltc import integrate_liquid_neurons_with_visualization")
    print("  history = integrate_liquid_neurons_with_visualization(")
    print("      project_id='your-project-id',")
    print("      table_id='your-dataset.your-table',")
    print("      window_size=5,")
    print("      stride_perspectives=[1, 3, 5],")
    print("      batch_size=32,")
    print("      epochs=15,")
    print("      pca_components=3")
    print("  )")

class TemporalStrideProcessor:
    """Processes temporal data with multiple stride perspectives."""
    
    def __init__(self, window_size: int, stride_perspectives: List[int], pca_components: int):
        """Initialize the TemporalStrideProcessor.
        
        Args:
            window_size: Size of the sliding window
            stride_perspectives: List of stride lengths to use
            pca_components: Number of PCA components to extract
        """
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives
        self.pca_components = pca_components

    def process_batch(self, data) -> Dict[int, Any]:
        """Process a batch of data with multiple stride perspectives.
        
        Args:
            data: Input data of shape (num_samples, num_features)
            
        Returns:
            Dictionary mapping stride lengths to processed data
        """
        # Convert to tensor for processing if it's not already
        data = convert_to_tensor(data)
            
        perspectives = {}
        for stride in self.stride_perspectives:
            if stride == 1:
                # Stride 1: No PCA, just create sliding windows
                strided_data = self._create_strided_sequences(data, stride)
                # Reshape to (num_windows, num_features * window_size)
                reduced_data = reshape(strided_data, (shape(strided_data)[0], -1))
            else:
                # Stride > 1: Apply PCA per column
                strided_data = self._create_strided_sequences(data, stride)
                reduced_data = self._apply_pca_per_column(strided_data)
            perspectives[stride] = reduced_data
            print(f"TemporalStrideProcessor: Stride {stride}, Output Shape: {shape(reduced_data)}")
        return perspectives

    def _create_strided_sequences(self, data: Any, stride: int) -> Any:
        """Create strided sequences from the input data.
        
        Args:
            data: Input data of shape (num_samples, num_features)
            stride: Stride length
            
        Returns:
            Strided sequences of shape (num_sequences, window_size, num_features)
        """
        num_samples = shape(data)[0]
        num_features = shape(data)[1]
        subsequences = []

        for i in range(0, num_samples - self.window_size + 1, stride):
            subsequence = data[i:i + self.window_size]
            subsequences.append(subsequence)
        # Pad with the last window to keep the sequence as long as possible.
        if (num_samples - self.window_size + 1) % stride != 0:
            last_index = max(0, num_samples - self.window_size)
            subsequences.append(data[last_index:last_index+self.window_size])

        return stack(subsequences)

    def _apply_pca_per_column(self, strided_data: Any) -> Any:
        """Apply PCA to each column of the strided data.
        
        Args:
            strided_data: Strided data of shape (num_sequences, window_size, num_features)
            
        Returns:
            PCA-reduced data of shape (num_sequences, num_features * pca_components)
        """
        # Convert to numpy for PCA processing (PCA requires numpy arrays)
        strided_data_tensor = tensor.convert_to_tensor(strided_data)
        num_sequences = strided_data_tensor.shape[0]
        num_features = strided_data_tensor.shape[2]  # Original number of features
        reduced_features = []

        for i in range(num_sequences):
            sequence = strided_data_tensor[i]  # (window_size, num_features)
            pca_results = []
            for j in range(num_features):
                column_data = sequence[:, j].reshape(-1, 1)  # Reshape for PCA
                # Check if all values are the same
                if (column_data == column_data[0]).all():  # Check for constant columns
                    if column_data.shape[0] < self.pca_components:
                        # Create padded array with constant values
                        padded_column = pad(
                            convert_to_tensor(column_data.flatten()),
                            [[0, self.pca_components - column_data.shape[0]]],
                            constant_values=column_data[0,0]
                        )
                        pca_results.append(to_numpy(padded_column))
                    else:
                        # Create array filled with constant value
                        pca_results.append(to_numpy(
                            full((self.pca_components,), column_data[0,0])
                        ))
                else:
                    pca = PCA()
                    try:
                        transformed = pca.fit_transform(column_data, n_components=self.pca_components)
                        pca_results.append(transformed.flatten())  # Flatten to 1D
                    except ValueError as e:
                        print(f"Error during PCA for sequence {i}, column {j}: {e}")
                        print("Input Data to PCA:", column_data)
                        raise

            # Concatenate PCA results
            reduced_features.append(to_numpy(
                concatenate([convert_to_tensor(arr) for arr in pca_results])
            ))

        # Convert back to tensor
        return convert_to_tensor(reduced_features)