from ember_ml import ops
import matplotlib.pyplot as plt
import numpy as np

# Create an EmberTensor
tensor = ops.ones((10,))
print('Type:', type(tensor))

# Try to plot directly with matplotlib
try:
    plt.figure()
    plt.plot(tensor)
    plt.close()
    print('Can plot directly with matplotlib: True')
except Exception as e:
    print('Can plot directly with matplotlib: False')
    print('Error:', e)

# Try to convert to NumPy if needed
try:
    if hasattr(tensor, 'numpy'):
        numpy_tensor = tensor.numpy()
        print('Has numpy() method: True')
    else:
        print('Has numpy() method: False')
        # If it's already a NumPy array, we can use it directly
        if isinstance(tensor, np.ndarray):
            numpy_tensor = tensor
            print('Is already a NumPy array: True')
        else:
            numpy_tensor = np.array(tensor)
            print('Converted to NumPy array using np.array()')
    
    # Try plotting the NumPy array
    plt.figure()
    plt.plot(numpy_tensor)
    plt.close()
    print('Can plot NumPy array: True')
except Exception as e:
    print('Can plot NumPy array: False')
    print('Error:', e)