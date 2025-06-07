"""
Demo script for the auto-selection of backends in Ember ML.

This script demonstrates how Ember ML automatically selects the best backend
based on the available hardware.
"""

import platform
import sys
import os

# Add the parent directory to the path so we can import ember_ml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ember_ml.backend import auto_select_backend, get_backend, get_device, set_backend, set_device
from ember_ml.nn.tensor import zeros # Import zeros directly

def print_system_info():
    """Print information about the system."""
    print("System Information:")
    print(f"  Platform: {platform.system()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Python: {platform.python_version()}")
    print()

def check_torch_availability():
    """Check if PyTorch is available and what devices it can use."""
    print("PyTorch Availability:")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Check for MPS (Metal Performance Shaders) on macOS
        mps_available = False
        if platform.system() == 'Darwin':
            try:
                mps_available = torch.backends.mps.is_available()
                print(f"  MPS available: {mps_available}")
            except:
                print("  MPS not supported in this PyTorch version")
    except ImportError:
        print("  PyTorch not installed")
    print()

def check_mlx_availability():
    """Check if MLX is available."""
    print("MLX Availability:")
    try:
        import mlx.core
        print(f"  MLX installed")
        try:
            metal_available = mlx.core.metal.is_available()
            print(f"  Metal backend available: {metal_available}")
            if metal_available:
                device_info = mlx.core.metal.device_info()
                print(f"  Device info: {device_info}")
        except:
            print("  Could not check Metal backend availability")
    except ImportError:
        print("  MLX not installed")
    print()

def demonstrate_auto_selection():
    """Demonstrate the auto-selection of backends."""
    print("Auto-selecting Backend:")
    backend, device = auto_select_backend()
    print(f"  Selected backend: {backend}")
    print(f"  Selected device: {device}")
    print()
    
    print("Current Backend Configuration:")
    print(f"  Current backend: {get_backend()}")
    # Create a dummy tensor to get the device
    dummy = zeros((1, 1)) # Use imported zeros
    print(f"  Current device: {get_device(dummy)}")
    print()

def demonstrate_backend_switching():
    """Demonstrate switching between backends."""
    print("Demonstrating Backend Switching:")
    from ember_ml import ops
    # Save the original backend and device
    original_backend = get_backend()
    set_backend(original_backend) # Correct call to set_backend
    # Create a dummy tensor to get the device
    dummy = zeros((1, 1)) # Use imported zeros
    original_device = get_device(dummy)
    
    # Try each available backend
    backends = ['numpy', 'torch', 'mlx']
    for backend in backends:
        try:
            print(f"  Switching to {backend}...")
            set_backend(backend)
            print(f"  Current backend: {get_backend()}")
            
            # If torch, try different devices
            if backend == 'torch':
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(f"  Setting device to cuda...")
                        set_device('cuda')
                        # Create a dummy tensor to get the device
                        dummy = zeros((1, 1)) # Use imported zeros
                        print(f"  Current device: {get_device(dummy)}")
                    
                    # Check for MPS on macOS
                    if platform.system() == 'Darwin':
                        try:
                            if torch.backends.mps.is_available():
                                print(f"  Setting device to mps...")
                                set_device('mps')
                                # Create a dummy tensor to get the device
                                dummy = zeros((1, 1)) # Use imported zeros
                                print(f"  Current device: {get_device(dummy)}")
                        except:
                            pass
                    
                    print(f"  Setting device to cpu...")
                    set_device('cpu')
                    # Create a dummy tensor to get the device
                    dummy = zeros((1, 1)) # Use imported zeros
                    print(f"  Current device: {get_device(dummy)}")
                except ImportError:
                    print("  PyTorch not installed, skipping device tests")
        except Exception as e:
            print(f"  Error switching to {backend}: {e}")
    
    # Restore the original backend and device
    print(f"  Restoring original backend: {original_backend}")
    set_backend(original_backend)
    if original_device and original_backend != 'mlx':
        set_device(original_device)
    print(f"  Current backend: {get_backend()}")
    # Create a dummy tensor to get the device
    dummy = zeros((1, 1)) # Use imported zeros
    print(f"  Current device: {get_device(dummy)}")
    print()

def main():
    """Main function."""
    print("=" * 80)
    print("Ember ML Backend Auto-Selection Demo")
    print("=" * 80)
    print()
    
    print_system_info()
    check_torch_availability()
    check_mlx_availability()
    demonstrate_auto_selection()
    demonstrate_backend_switching()
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()