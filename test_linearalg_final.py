# Test the first import pattern
from ember_ml.ops import linearalg
import numpy as np

# Create a test matrix
matrix = np.array([[1, 2], [3, 4]])

# Perform SVD
u1, s1, vh1 = linearalg.svd(matrix)

print("First import pattern:")
print("Original matrix:")
print(matrix)
print("\nU matrix:")
print(u1)
print("\nSingular values:")
print(s1)
print("\nVh matrix:")
print(vh1)

# Test the second import pattern
import ember_ml.ops.linearalg
from ember_ml.ops.linearalg import svd

# Perform SVD
u2, s2, vh2 = svd(matrix)

print("\nSecond import pattern:")
print("Original matrix:")
print(matrix)
print("\nU matrix:")
print(u2)
print("\nSingular values:")
print(s2)
print("\nVh matrix:")
print(vh2)

# Verify that both patterns give the same result
print("\nDo both patterns give the same result?")
print(np.allclose(u1, u2) and np.allclose(s1, s2) and np.allclose(vh1, vh2))