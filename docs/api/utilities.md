# Utilities

This section documents various utility functions and classes that support the Ember ML framework, often providing backend-agnostic helpers or tools for common tasks like performance analysis and visualization.

## Backend Utilities (`ember_ml.utils.backend_utils`)

Provides utility functions for working with Ember ML's backend system.

*   **`get_current_backend()`**: Returns the name of the currently active backend.
*   **`set_preferred_backend(backend_name)`**: Sets the preferred backend if available, otherwise uses the default. Returns the name of the backend that was actually set.
*   **`initialize_random_seed(seed)`**: Initializes the random seed for reproducibility across backends (if supported by the backend).
*   **`convert_to_tensor_safe(data)`**: Safely converts input data to a tensor using the current backend, handling various input types.
*   **`tensor_to_numpy_safe(tensor)`**: Safely converts a tensor from any backend to a NumPy array, handling different backend tensor types and devices (like PyTorch MPS).
*   **`random_uniform(shape, low, high)`**: Generates uniform random values using the current backend.
*   **`sin_cos_transform(values, period)`**: Applies sine and cosine transformations for cyclical features using `ops`.
*   **`vstack_safe(arrays)`**: Safely stacks a list of arrays vertically using the current backend, handling potential shape mismatches.
*   **`get_backend_info()`**: Gets information about the current backend (name, device).
*   **`print_backend_info()`**: Prints information about the current backend and performs a simple test operation.

## Fraction Utilities (`ember_ml.utils.fraction`)

Provides utilities for working with fractions, primarily using Python's built-in `fractions.Fraction`.

*   **`to_fraction(value)`**: Converts a float, int, or string to a `Fraction`.
*   **`to_float(frac)`**: Converts a `Fraction` to a float.
*   **`simplify_fraction(frac, max_denominator)`**: Simplifies a fraction to a given maximum denominator.
*   **`fraction_to_ratio(frac)`**: Converts a fraction to a tuple of (numerator, denominator).
*   **`ratio_to_fraction(numerator, denominator)`**: Converts a ratio of integers to a `Fraction`.
*   **`continued_fraction(value, max_terms)`**: Computes the continued fraction representation of a float.
*   **`from_continued_fraction(terms)`**: Converts a continued fraction representation (list of terms) back to a `Fraction`.

## Performance Utilities (`ember_ml.utils.performance`)

Provides tools for measuring and analyzing the performance of code.

*   **`timeit(func)`**: Decorator to measure the execution time of a function.
*   **`benchmark(func, *args, **kwargs)`**: Benchmarks a function over multiple runs and returns statistics (mean, std, min, max, times) and the last result.
*   **`compare_functions(funcs, args_list, kwargs_list, labels)`**: Compares the performance of multiple functions using `benchmark`.
*   **`plot_benchmark_results(results, title)`**: Plots benchmark results as a bar chart with error bars.
*   **`memory_usage(func)`**: Decorator to measure the memory usage of a function using `psutil`.
*   **`profile_function(func, *args, **kwargs)`**: Profiles a function using `cProfile` and prints the top function calls by cumulative time.

## Visualization Utilities (`ember_ml.utils.visualization`)

Provides general plotting functions using Matplotlib, often for visualizing wave-related data or model outputs.

*   **`plot_wave(wave, sample_rate, title)`**: Plots a 1D wave signal over time.
*   **`plot_spectrogram(wave, sample_rate, window_size, hop_length, title)`**: Plots a spectrogram using STFT.
*   **`plot_confusion_matrix(cm, class_names, title)`**: Plots a confusion matrix.
*   **`plot_roc_curve(fpr, tpr, roc_auc, title)`**: Plots a Receiver Operating Characteristic (ROC) curve.
*   **`plot_precision_recall_curve(precision, recall, title)`**: Plots a precision-recall curve.
*   **`plot_learning_curve(train_scores, val_scores, title)`**: Plots training and validation scores over epochs.
*   **`fig_to_image(fig)`**: Converts a Matplotlib figure to a PIL Image.
*   **`plot_to_numpy(fig)`**: Converts a Matplotlib figure to a NumPy array.

*(Note: Some wave-specific visualization functions are located in `ember_ml.wave.utils.wave_visualization` and are documented under the Wave Architecture section.)*