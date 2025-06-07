# Liquid Neuron Applications

This section describes specific applications and models built using liquid neuron architectures within Ember ML.

## Anomaly Detection

### `ember_ml.models.liquid.liquid_anomaly_detector`

*   **`generate_log_data(...)`**: Generates synthetic Splunk-like log data with embedded anomaly patterns for testing.
*   **`LiquidAnomalyDetector`**: Implements an anomaly detection system using a multi-scale CfC network.
    *   `__init__(total_neurons, motor_neurons, sequence_length)`: Initializes the detector, scaler, and builds the multi-scale CfC model (`_build_model`).
    *   `_build_model(...)`: Creates a `keras.Sequential` model with multiple `CfC` layers (using `AutoNCP` wiring) at different scales (fast, medium, slow), followed by Dense layers for anomaly probability prediction. Compiles the model with Adam optimizer and binary cross-entropy loss. *(Note: Uses Keras directly, violating backend purity).*
    *   `_encode_features(df)`: Encodes categorical log features (location, message, severity) into a numerical format suitable for the RNN, scales the features, and creates overlapping sequences.
    *   `_detect_anomalies(sequences, threshold)`: Uses the trained Keras model to predict anomaly probabilities and applies a threshold.
    *   `_generate_labels(df)`: Generates synthetic anomaly labels based on predefined rules (e.g., multiple critical events, repeated issues) for training purposes.
    *   `train(df, epochs, validation_split)`: Trains the Keras model using the encoded sequences and generated labels.
    *   `process_logs(df, threshold)`: Encodes logs, trains the model if not already trained, predicts anomaly probabilities, and prints detected anomalies with context.

### `ember_ml.nn.modules.anomaly.liquid_autoencoder`

*   **`LiquidAutoencoder(Module)`**: An autoencoder model potentially using liquid layers (like LTC or CfC) in its encoder and decoder components for reconstruction-based anomaly detection. (Note: The specific internal layers are not detailed in the `__init__.py` but it's grouped under anomaly detection).

## Forecasting

### `ember_ml.nn.modules.forecasting.liquid_forecaster`

*   **`LiquidForecaster(Module)`**: A forecasting model likely utilizing liquid layers (LTC or CfC) for time series prediction. It might incorporate mechanisms for uncertainty estimation. (Note: Specific implementation details are not in the `__init__.py`).