# MLX Examples

This directory contains examples and implementations using Apple's MLX framework, which is optimized for Apple Silicon.

## Files

- `mlxncps.py`: Implementation of Neural Circuit Policies and related components using MLX

## Usage

These examples can be run directly to see MLX implementations in action:

```bash
# Run the MLX NCPS example
python examples/mlx/mlxncps.py
```

## Background

### Apple MLX Framework

MLX is a framework for machine learning on Apple Silicon, designed to take advantage of the Metal Performance Shaders (MPS) and the Neural Engine. Key features include:

- Optimized for Apple Silicon (M1/M2/M3)
- Array-based computation similar to NumPy
- Automatic differentiation
- GPU acceleration via Metal
- Neural Engine support for specific operations

### Neural Circuit Policies in MLX

The `mlxncps.py` file provides implementations of:

- Closed-form Continuous-time (CfC) cells
- Liquid Time Constant (LTC) cells
- ConvLSTM2D cells
- Multi-scale CfC networks

These implementations are optimized for Apple Silicon and provide similar functionality to the TensorFlow/PyTorch versions, but with native Metal acceleration.

## Requirements

- macOS 12.0 or later
- Apple Silicon (M1/M2/M3) processor
- MLX framework (`pip install mlx`)