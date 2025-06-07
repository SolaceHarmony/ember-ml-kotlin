# Wave-Based Neural Network Models

This section describes various neural network architectures built upon wave processing principles within Ember ML.

## Core Concepts

These models leverage wave representations (binary, harmonic, etc.) or quantum-inspired concepts within their architectures, often combined with standard neural network layers like RNNs, Transformers, or Autoencoders.

## Models

### `ember_ml.wave.models.wave_autoencoder`

*   **`WaveEncoder(nn.Module)`**: A standard feed-forward encoder (Linear layers, activation, dropout).
*   **`WaveDecoder(nn.Module)`**: A standard feed-forward decoder (Linear layers, activation, dropout).
*   **`WaveVariationalEncoder(nn.Module)`**: A variational encoder that outputs mean and log variance for the latent space. Includes the reparameterization trick.
*   **`WaveAutoencoder(nn.Module)`**: Combines `WaveEncoder` (or `WaveVariationalEncoder`) and `WaveDecoder`.
    *   `forward(x)`: Encodes input `x` and returns reconstruction, latent representation `z`, and optionally `mean` and `log_var` (if variational).
    *   `encode(x)`: Encodes input `x` to latent `z`.
    *   `decode(z)`: Decodes latent `z` to reconstruction.
*   **`WaveConvolutionalAutoencoder(nn.Module)`**: An autoencoder using 1D convolutional layers for encoding and transposed convolutions for decoding. Supports variational mode.
    *   `_calculate_conv_output_size()`: Helper to determine flattened size after convolutions.
    *   `_build_encoder_conv()` / `_build_decoder_conv()`: Helper methods to build convolutional layers.
    *   `encode(x)` / `decode(z)` / `forward(x)`: Similar to `WaveAutoencoder` but using convolutional layers.
*   **`create_wave_autoencoder(...)`**: Factory function to create a `WaveAutoencoder`.
*   **`create_wave_conv_autoencoder(...)`**: Factory function to create a `WaveConvolutionalAutoencoder`.

### `ember_ml.wave.models.wave_transformer`

*   **`WaveMultiHeadAttention(nn.Module)`**: Implements standard multi-head self-attention.
*   **`WaveTransformerEncoderLayer(nn.Module)`**: A standard transformer encoder layer combining multi-head self-attention and a feed-forward network with layer normalization and dropout.
*   **`WaveTransformerEncoder(nn.Module)`**: Stacks multiple `WaveTransformerEncoderLayer` instances.
*   **`WaveTransformer(nn.Module)`**: A standard transformer encoder model including positional encoding.
    *   `_init_positional_encoding()`: Initializes sinusoidal positional encodings.
    *   `forward(x, mask)`: Adds positional encoding and processes input through the encoder stack.
*   **`create_wave_transformer(...)`**: Factory function to create a `WaveTransformer`.

### `ember_ml.wave.models.wave_rnn`

*   **`WaveGRUCell(nn.Module)`**: Implements a standard GRU cell.
*   **`WaveGRU(nn.Module)`**: A GRU RNN layer built using `WaveGRUCell`. Supports multiple layers, dropout, and bidirectionality.
*   **`WaveRNN(nn.Module)`**: A general RNN model that can wrap either `WaveGRU` or `nn.LSTM`.
    *   `forward(x, h)`: Processes input sequence `x` through the chosen RNN type.
*   **`create_wave_rnn(...)`**: Factory function to create a `WaveRNN`.

### `ember_ml.wave.models.multi_sphere`

*   **`SphereProjection(nn.Module)`**: Projects data onto a unit sphere and optionally applies rotation.
*   **`MultiSphereProjection(nn.Module)`**: Applies `SphereProjection` independently to multiple spheres (segments) of the input data.
*   **`MultiSphereEncoder(nn.Module)`**: An encoder using `MultiSphereProjection` and linear layers.
*   **`MultiSphereDecoder(nn.Module)`**: A decoder using linear layers and `MultiSphereProjection`.
*   **`MultiSphereModel(nn.Module)`**: An autoencoder-like model combining `MultiSphereEncoder` and `MultiSphereDecoder`.
*   **`SphericalHarmonics(nn.Module)`**: Computes spherical harmonics for input data projected onto spheres.
*   **`MultiSphereHarmonicModel(nn.Module)`**: Combines multi-sphere projection with spherical harmonics and linear layers.
*   **`MultiSphereWaveModel(nn.Module)`**: A complex model integrating multi-sphere projections, spherical harmonics, wave dynamics (potentially simulated via RNN or similar), and linear layers for encoding and decoding.
*   **`create_multi_sphere_model(...)`**: Factory function for `MultiSphereModel`.
*   **`create_multi_sphere_harmonic_model(...)`**: Factory function for `MultiSphereHarmonicModel`.
*   **`create_multi_sphere_wave_model(...)`**: Factory function for `MultiSphereWaveModel`.