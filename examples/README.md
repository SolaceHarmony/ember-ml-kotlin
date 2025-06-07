# Examples

This directory contains example implementations, demonstrations, and applications of the ember_ml framework.

## Subdirectories

- **demos/**: Demonstration scripts showcasing various features and capabilities
- **ltc/**: Examples of Liquid Time Constant (LTC) neurons and Closed-form Continuous-time (CfC) cells
- **mlx/**: Examples using Apple's MLX framework optimized for Apple Silicon
- **rbm/**: Examples of Restricted Boltzmann Machines (RBMs)

## Additional Files

- **otherneurons.py**: Examples of other neuron types not covered in the subdirectories
- **ncp_example.py**: Example of Neural Circuit Policy (NCP) implementation

## Usage

Each subdirectory contains its own README.md file with specific information about the examples in that directory. Generally, examples can be run directly:

```bash
# Run an example
python examples/ncp_example.py

# Run an example from a subdirectory
python examples/rbm/rbm_example.py
```

## Adding New Examples

When adding new examples to this directory, please follow these guidelines:

1. Place related examples in the appropriate subdirectory
2. Create a new subdirectory if the example doesn't fit into an existing category
3. Include a README.md file in any new subdirectory
4. Use clear, descriptive names for files
5. Include docstrings and comments to explain what the example does
6. Make sure the example runs without errors on all supported platforms