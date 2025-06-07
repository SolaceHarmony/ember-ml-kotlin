# Restricted Boltzmann Machines (RBMs)

This section documents the implementations of Restricted Boltzmann Machines (RBMs) within Ember ML, including standard, optimized, and anomaly detection variants, as well as training and visualization components.

## Core Concepts

Restricted Boltzmann Machines are a type of generative stochastic artificial neural network that can learn a probability distribution over its set of inputs. They consist of a visible layer and a hidden layer with connections between neurons in different layers but no connections within a layer. RBMs can be used for dimensionality reduction, feature learning, and anomaly detection by measuring reconstruction error or free energy.

## Implementations

### `ember_ml.models.rbm.rbm_module`

*   **`RBMModule(Module)`**: A backend-agnostic RBM implementation using the Ember ML Module system.
    *   `__init__(n_visible, n_hidden, learning_rate, momentum, weight_decay, use_binary_states)`: Initializes the RBM with specified dimensions and training parameters. Initializes weights and biases as `Parameter` objects.
    *   `forward(visible_states)`: Transforms visible states to hidden probabilities (calls `compute_hidden_probabilities`).
    *   `compute_hidden_probabilities(visible_states)`: Computes probabilities of hidden units given visible states using matrix multiplication and sigmoid activation.
    *   `sample_hidden_states(hidden_probs)`: Samples binary hidden states from probabilities (if `use_binary_states` is True).
    *   `compute_visible_probabilities(hidden_states)`: Computes probabilities of visible units given hidden states.
    *   `sample_visible_states(visible_probs)`: Samples binary visible states from probabilities (if `use_binary_states` is True).
    *   `reconstruct(visible_states)`: Reconstructs visible states by sampling hidden states and then computing visible probabilities.
    *   `reconstruction_error(visible_states, per_sample)`: Computes the mean squared reconstruction error.
    *   `free_energy(visible_states)`: Computes the free energy of visible states.
    *   `anomaly_score(visible_states, method)`: Computes anomaly score based on reconstruction error or free energy.
    *   `is_anomaly(visible_states, method)`: Determines if samples are anomalous based on a threshold.

### `ember_ml.models.rbm.training`

*   **`contrastive_divergence_step(rbm, batch_data, k)`**: Performs one step of the Contrastive Divergence algorithm for an `RBMModule`. Computes positive and negative phase associations, calculates gradients, and returns gradients and reconstruction error.
*   **`train_rbm(rbm, data_generator, epochs, k, validation_data, early_stopping_patience, callback)`**: Trains an `RBMModule` using a data generator, supporting epochs, CD-k, validation, early stopping, and callbacks. Updates RBM parameters using momentum and weight decay. Computes and stores training and validation errors. Calculates anomaly thresholds after training.
*   **`transform_in_chunks(rbm, data_generator)`**: Transforms data to hidden representation in chunks using a generator.
*   **`save_rbm(rbm, filepath)`**: Saves an `RBMModule` to a JSON file, including parameters and metadata.
*   **`load_rbm(filepath)`**: Loads an `RBMModule` from a JSON file.

### `ember_ml.models.rbm.rbm`

*   **`RestrictedBoltzmannMachine`**: A CPU-friendly RBM implementation using NumPy.
    *   Includes methods for `sigmoid`, `compute_hidden_probabilities`, `sample_hidden_states`, `compute_visible_probabilities`, `sample_visible_states`, `contrastive_divergence`, `train`, `transform`, `reconstruct`, `reconstruction_error`, `free_energy`, `anomaly_score`, `is_anomaly`, `dream`, `save`, `load`, and `summary`.
*   **`RBM`**: A PyTorch implementation of an RBM.
    *   Includes methods for `__init__`, `forward`, `backward`, `free_energy`, `contrastive_divergence`, and `train`.

### `ember_ml.models.optimized_rbm`

*   **`OptimizedRBM`**: An optimized RBM implementation for large-scale feature learning with memory efficiency and chunked training, supporting optional GPU acceleration (PyTorch).
    *   Includes methods for `__init__`, `_to_gpu`, `_to_cpu`, `sigmoid`, `compute_hidden_probabilities`, `sample_hidden_states`, `compute_visible_probabilities`, `sample_visible_states`, `contrastive_divergence`, `train_in_chunks`, `transform`, `transform_in_chunks`, `reconstruct`, `reconstruction_error`, `free_energy`, `anomaly_score`, `is_anomaly`, `save`, `load`, and `summary`.
    *   **`data_generator(data, batch_size)`**: Helper function to yield data in batches.

### `ember_ml.models.rbm_anomaly_detector`

*   **`RBMBasedAnomalyDetector`**: An anomaly detection system based on RBMs, integrating with generic feature extraction.
    *   `__init__(n_hidden, learning_rate, momentum, weight_decay, batch_size, anomaly_threshold_percentile, anomaly_score_method, track_states)`: Initializes RBM parameters, preprocessing parameters, and anomaly detection parameters.
    *   `preprocess(X, fit, scaling_method)`: Preprocesses data (scaling) and optionally fits preprocessing parameters.
    *   `fit(X, validation_data, epochs, k, early_stopping_patience, scaling_method, verbose)`: Fits the anomaly detector by training the internal `RBMModule`. Computes anomaly thresholds after training.
    *   `predict(X)`: Predicts whether samples are anomalies.
    *   `anomaly_score(X)`: Computes anomaly scores.
    *   `anomaly_probability(X)`: Computes anomaly probability using a sigmoid on normalized scores.
    *   `reconstruct(X)`: Reconstructs input data using the RBM and inverse scaling.
    *   `save(filepath)` / `load(filepath)`: Saves and loads the detector's state (excluding the RBM module itself in the current implementation).
    *   `summary()`: Provides a summary of the detector.
    *   `categorize_anomalies(X, anomaly_flags, max_clusters, min_samples_per_cluster)`: Categorizes anomalies using KMeans clustering on hidden unit activations.
*   **`generate_telemetry_data(...)`**: Generates synthetic telemetry data with anomalies for example usage.
*   **`save_to_csv(df, filename)`**: Saves a DataFrame to a CSV file.
    *   **`detect_anomalies_from_features(...)`**: Example function demonstrating end-to-end anomaly detection from features using the `RBMBasedAnomalyDetector`.