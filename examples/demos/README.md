# Demos

This directory contains demonstration scripts showcasing various features and capabilities of the ember_ml framework.

## Files

- `torch_mps_demo.py`: Demonstration of PyTorch MPS (Metal Performance Shaders) acceleration on Apple Silicon

## Usage

These demo scripts can be run directly to see the ember_ml framework in action:

```bash
# Run the PyTorch MPS demo
python examples/demos/torch_mps_demo.py
```

## Adding New Demos

When adding new demonstration scripts to this directory, please follow these guidelines:

1. Use clear, descriptive names for files
2. Include docstrings and comments to explain what the demo does
3. Add the new demo to this README.md file
4. Make sure the demo runs without errors on all supported platforms
5. Include example output or visualizations where appropriate

## Demo Requirements

Some demos may have additional requirements beyond the core ember_ml dependencies. These requirements should be documented in the demo script itself, as well as in this README.md file.

### PyTorch MPS Demo Requirements

- PyTorch 1.12 or later
- macOS 12.3 or later
- Apple Silicon (M1/M2/M3) processor