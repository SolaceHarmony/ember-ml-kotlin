# Liquid Neuron Training Utilities

This section describes utilities specifically designed for processing data and monitoring the training of liquid neuron models.

## Components

### `ember_ml.models.liquid.liquidtrainer`

*   **`TelemetryProcessor`**: Handles loading, preprocessing, and sequencing of telemetry data for liquid network training.
    *   `__init__(seq_len, stride, test_size)`: Initializes sequence length, stride, test split size, and a `StandardScaler`.
    *   `load_data(csv_path)`: Loads data, extracts numeric columns, splits into train/test, and applies standardization (fitting scaler only on train data).
    *   `create_sequences(data)`: Converts time series data into overlapping sequences suitable for RNN input.
*   **`LiquidNeuralNetwork`**: A helper class (likely for demonstration) that builds a multi-scale liquid network using Keras layers.
    *   `__init__(input_shape, model_size)`: Initializes input shape and model size.
    *   `_build_model()`: Creates a `keras.Sequential` model with multiple `CfC` layers (using `AutoNCP` wiring) at different scales, followed by Dense layers for anomaly prediction. Compiles the model. *(Note: Uses Keras directly, violating backend purity).*
*   **`MultiScaleMonitor(keras.callbacks.Callback)`**: A Keras callback to monitor the mean and standard deviation of outputs from intermediate `CfC` layers during training. *(Note: Keras dependency).*
*   **`main()`**: Example function demonstrating the use of `TelemetryProcessor` and `LiquidNeuralNetwork` to load data, build a model, set up Keras callbacks (including `MultiScaleMonitor`), train the model using `tf.data.Dataset`, evaluate, save the model, and optionally convert to CoreML. *(Note: Uses Keras/TensorFlow directly).*

### `ember_ml.nn.modules.trainers.memory_optimized_trainer`

*   **`MemoryOptimizedTrainer(Module)`**: A trainer potentially optimized for memory usage, possibly targeting specific hardware like Apple Silicon. (Note: Specific implementation details are not in the `__init__.py`).