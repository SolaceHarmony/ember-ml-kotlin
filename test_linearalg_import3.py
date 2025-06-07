import ember_ml.ops.linearalg
from ember_ml.ops.linearalg import svd
import numpy as np

# Create a test matrix
matrix = np.array([[1, 2], [3, 4]])

# Perform SVD
u, s, vh = svd(matrix)

# Print the results
print("Original matrix:")
print(matrix)
print("\nU matrix:")
print(u)
print("\nSingular values:")
print(s)
print("\nVh matrix:")
print(vh)

# Verify the decomposition by reconstructing the original matrix
reconstructed = np.dot(u * s, vh)
print("\nReconstructed matrix:")
print(reconstructed)
print("\nIs the reconstruction close to the original?")
print(np.allclose(matrix, reconstructed))