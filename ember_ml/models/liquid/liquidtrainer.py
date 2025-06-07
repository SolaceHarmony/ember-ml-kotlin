import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Keep StandardScaler import

from ember_ml import ops
from ember_ml.nn.modules import AutoNCP, Dense, Dropout # Keep AutoNCP import
from ember_ml.nn.modules.rnn import CfC # Keep CfC import
from ember_ml.nn import Module # Keep Module and Sequential imports
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn.container import Sequential,BatchNormalization
from ember_ml.training
# --------------------------
# 📊 Telemetry Data Pipeline
# --------------------------
class TelemetryProcessor:
    def __init__(self, seq_len=64, stride=16, test_size=0.2):
        self.seq_len = seq_len
        self.stride = stride
        self.test_size = test_size
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path):
        """Load and preprocess telemetry data."""
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # Extract numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.drop('timestamp', errors='ignore')
        data = df[numeric_cols]
        
        # Split and scale
        split_idx = int(ops.multiply(len(data), ops.subtract(1.0, self.test_size))) # Replaced * and - with ops
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Fit scaler on training data only
        train_norm = self.scaler.fit_transform(train_data)
        test_norm = self.scaler.transform(test_data)
        
        return train_norm, test_norm
        
    def create_sequences(self, data):
        """Convert raw data into overlapping sequences."""
        sequences = []
        for i in range(0, ops.add(ops.subtract(len(data), self.seq_len), 1), self.stride): # Replaced -, + with ops
            sequences.append(data[i:ops.add(i, self.seq_len)]) # Replaced + with ops
        return tensor.stack(sequences)

# --------------------------
# 🔥 Multi-Scale Liquid Neural Network
# --------------------------
class LiquidNeuralNetwork:
    def __init__(self, input_shape, model_size=128):
        self.input_shape = input_shape
        self.model_size = model_size
        self.model = self._build_model()
        
    def _build_model(self):
        # Fast timescale layer for immediate feature detection
        wiring_fast = AutoNCP(
            units=self.model_size,
            output_size=ops.floor_divide(self.model_size, 4), # Replaced // with ops
            sparsity_level=0.5
        )
        
        ltc_fast = CfC(
            wiring_fast,
            return_sequences=True,
            mixed_memory=True
        )
        
        # Medium timescale layer for pattern recognition
        wiring_med = AutoNCP(
            units=ops.floor_divide(self.model_size, 2), # Replaced // with ops
            output_size=ops.floor_divide(self.model_size, 8), # Replaced // with ops
            sparsity_level=0.4
        )
        
        ltc_med = CfC(
            wiring_med,
            return_sequences=True,
            mixed_memory=True
        )
        
        # Slow timescale layer for trend analysis
        wiring_slow = AutoNCP(
            units=ops.floor_divide(self.model_size, 4), # Replaced // with ops
            output_size=ops.floor_divide(self.model_size, 16), # Replaced // with ops
            sparsity_level=0.3
        )
        
        ltc_slow = CfC(
            wiring_slow,
            return_sequences=False,  # Only need final state for anomaly detection
            mixed_memory=True
        )
        # Build the model
        # Sequential model with multi-scale liquid neural layers
        model = Sequential([
            # Input layer
            EmberTensor(0.0,shape=self.input_shape),
            
            # Multi-scale liquid neural layers
            ltc_fast,
            BatchNormalization(),
            
            ltc_med,
            BatchNormalization(),
            
            ltc_slow,
            BatchNormalization(),
            
            # Anomaly detection head
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Optimizer with gradient clipping
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model

# --------------------------
# 📈 Training Callbacks
# --------------------------
class MultiScaleMonitor(keras.callbacks.Callback):
    """Monitor the outputs of multiple LTC layers during training"""
    def __init__(self):
        super().__init__()
        self.monitoring_model = None
    
    def on_train_batch_end(self, batch, logs=None):
        if self.monitoring_model is None:
            ltc_layers = [layer for layer in self.model.layers if isinstance(layer, CfC)]
            if ltc_layers:
                layer_outputs = []
                inputs = keras.layers.Input(shape=self.model.input_shape[1:])
                x = inputs
                for layer in ltc_layers:
                    x = layer(x)
                    layer_outputs.append(x)
                self.monitoring_model = keras.Model(inputs=inputs, outputs=layer_outputs)
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        if self.monitoring_model is not None and hasattr(self.model, 'validation_data'):
            try:
                data = self.model.validation_data[0][:1]
                layer_outputs = self.monitoring_model.predict(data, verbose=0)
                if not isinstance(layer_outputs, list):
                    layer_outputs = [layer_outputs]
                
                for i, output in enumerate(layer_outputs):
                    logs[f'ltc_{i}_output_mean'] = float(tf.reduce_mean(output))
                    logs[f'ltc_{i}_output_std'] = float(tf.math.reduce_std(output))
            except Exception as e:
                print(f"Warning: Error computing layer statistics: {str(e)}")

# --------------------------
# 🚀 Training & Deployment
# --------------------------
def main():
    # Initialize data processor
    processor = TelemetryProcessor(seq_len=64, stride=16)
    
    # Load and process data
    print("Loading telemetry data...")
    train_data, test_data = processor.load_data("network_telemetry.csv")
    
    # Create sequences
    print("Creating sequences...")
    train_seq = processor.create_sequences(train_data)
    test_seq = processor.create_sequences(test_data)
    
    # Create synthetic anomaly labels for demonstration
    # In practice, these would come from your labeled dataset
    train_labels = np.random.binomial(1, 0.1, size=(len(train_seq),))
    test_labels = np.random.binomial(1, 0.1, size=(len(test_seq),))
    
    # Prepare data loaders
    train_ds = tf.data.Dataset.from_tensor_slices((train_seq, train_labels))
    train_ds = train_ds.shuffle(1000).batch(64).prefetch(2)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_seq, test_labels))
    test_ds = test_ds.batch(64).prefetch(2)
    
    # Initialize model
    print("Building liquid neural network...")
    input_shape = (processor.seq_len, train_seq.shape[-1])
    model = LiquidNeuralNetwork(input_shape=input_shape, model_size=128).model
    
    # Setup callbacks
    callbacks = [
        MultiScaleMonitor(),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Results:")
    for metric, value in zip(model.metrics_names, results):
        print(f"{metric}: {value:.4f}")
    
    # Save model
    print("\nSaving model...")
    model.save("liquid_anomaly_detector.h5")
    
    try:
        import coremltools as ct
        print("\nConverting to CoreML format...")
        coreml_model = ct.convert(
            "liquid_anomaly_detector.h5",
            inputs=[ct.TensorType(name="input", shape=(1, *input_shape))],
            compute_precision=ct.precision.FLOAT32
        )
        coreml_model.save("LiquidAnomalyDetector.mlpackage")
        print("CoreML model saved successfully!")
    except ImportError:
        print("CoreML tools not available. Skipping CoreML conversion.")

if __name__ == "__main__":
    main()