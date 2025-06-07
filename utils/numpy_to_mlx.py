#!/usr/bin/env python3
"""
NumPy to MLX Function Converter

This script provides a command-line tool for looking up MLX equivalents of NumPy functions.
It loads data from a CSV file containing mappings between NumPy and MLX functions,
and allows users to search for functions by name or pattern.

The tool also provides code examples and detailed explanations to help developers
correctly implement MLX backend operations in the Ember ML framework.

Usage:
    python numpy_to_mlx.py [search_term]
    
Examples:
    python numpy_to_mlx.py                  # List all available functions
    python numpy_to_mlx.py array            # Search for functions containing "array"
    python numpy_to_mlx.py "np.sin"         # Look up the exact function
    python numpy_to_mlx.py --category math  # List functions in the math category
    python numpy_to_mlx.py --example slice  # Show detailed example for slice
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


# Function categories for organization
CATEGORIES = {
    "creation": ["array", "zeros", "ones", "eye", "arange", "linspace", "random"],
    "indexing": ["take", "slice", "slice_update", "[", "gather"],
    "math": ["add", "subtract", "multiply", "divide", "matmul", "dot", "exp", "log", 
             "sin", "cos", "tan", "tanh", "abs", "sqrt", "power", "square", "sign", 
             "round", "ceil", "floor"],
    "reduction": ["sum", "max", "min", "mean", "var", "std", "argmax", "argmin"],
    "shape": ["reshape", "transpose", "concatenate", "stack", "split", "expand_dims", 
              "squeeze", "flatten"],
    "logic": ["where", "logical_and", "logical_or", "logical_not", "equal", "not_equal",
              "greater", "greater_equal", "less", "less_equal", "isnan", "isinf"],
    "manipulation": ["pad", "tile", "repeat", "diag", "triu", "tril"],
    "linalg": ["norm", "inv", "solve", "svd", "eig", "eigvals", "qr", "cholesky"],
    "fft": ["fft", "ifft", "fft2", "ifft2"],
    "conversion": ["astype", "float", "int", "cast"],
    "properties": ["T", "copy", "flatten"],
    "io": ["save", "load"]
}


# Code examples for specific functions
CODE_EXAMPLES = {
    "mx.array": """
# Creating an MLX array
import mlx.core as mx

# From a Python list
a = mx.array([1, 2, 3])
print(a)  # array([1, 2, 3])

# With specific dtype
a = mx.array([1, 2, 3], dtype=mx.float32)
print(a)  # array([1.0, 2.0, 3.0])

# From a nested list (2D array)
b = mx.array([[1, 2], [3, 4]])
print(b)  # array([[1, 2], [3, 4]])

# IMPORTANT: In Ember ML backend implementation
from ember_ml import ops

def array_impl(self, data, dtype=None):
    # Convert input to MLX array
    result = mx.array(data)
    
    # Apply dtype if specified
    if dtype is not None:
        result = mx.astype(result, dtype)
        
    return result
""",

    "mx.slice": """
# Using MLX's slice function (MLX-specific, no direct NumPy equivalent)
import mlx.core as mx

# Create a test array
a = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Extract a slice
start_indices = mx.array([0, 1])  # Start at row 0, column 1 - MUST be an MLX array
axes = [0, 1]  # Python list, NOT an MLX array!
slice_size = [2, 2]  # Python list, NOT an MLX array!
b = mx.slice(a, start_indices, axes, slice_size)
print(b)  # array([[2, 3], [5, 6]])

# Equivalent to this basic slicing in NumPy/Python:
# b = a[0:2, 1:3]  # But mx.slice is more flexible for dynamic indices

# IMPORTANT: In Ember ML backend implementation
def slice_impl(self, tensor, starts, axes, sizes):
    # Ensure starts is an MLX array
    starts_mx = mx.array(starts, dtype=mx.int32)
    
    # Ensure axes and sizes are Python lists
    axes_list = list(axes)
    sizes_list = list(sizes)
    
    return mx.slice(tensor, starts_mx, axes_list, sizes_list)
""",

    "mx.slice_update": """
# Using MLX's slice_update function (MLX-specific, no direct NumPy equivalent)
import mlx.core as mx

# Create a test array
a = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Update a slice
start_indices = mx.array([0, 1])  # Start at row 0, column 1 - MUST be an MLX array
axes = [0, 1]  # Python list, NOT an MLX array!
update = mx.array([[10, 11], [12, 13]])  # New values to insert
c = mx.slice_update(a, update, start_indices, axes)
print(c)  # array([[1, 10, 11], [4, 12, 13], [7, 8, 9]])

# Equivalent to this in NumPy/Python:
# c = a.copy()
# c[0:2, 1:3] = update  # But mx.slice_update is more flexible for dynamic indices

# IMPORTANT: In Ember ML backend implementation
def slice_update_impl(self, tensor, update, starts, axes):
    # Ensure starts is an MLX array
    starts_mx = mx.array(starts, dtype=mx.int32)
    
    # Ensure axes is a Python list
    axes_list = list(axes)
    
    # Ensure update is an MLX array
    update_mx = mx.array(update)
    
    return mx.slice_update(tensor, update_mx, starts_mx, axes_list)
""",

    "mx.add": """
# Using MLX's add function
import mlx.core as mx

# Create test arrays
a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])

# Add arrays
c = mx.add(a, b)
print(c)  # array([5, 7, 9])

# Broadcasting
d = mx.add(a, mx.array(1))
print(d)  # array([2, 3, 4])

# IMPORTANT: In Ember ML backend implementation
def add_impl(self, x, y):
    # Ensure inputs are MLX arrays
    x_mx = mx.array(x)
    y_mx = mx.array(y)
    
    return mx.add(x_mx, y_mx)
""",

    "mx.matmul": """
# Using MLX's matmul function
import mlx.core as mx

# Create test matrices
a = mx.array([[1, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])

# Matrix multiplication
c = mx.matmul(a, b)
print(c)  # array([[19, 22], [43, 50]])

# IMPORTANT: In Ember ML backend implementation
def matmul_impl(self, x, y):
    # Ensure inputs are MLX arrays
    x_mx = mx.array(x)
    y_mx = mx.array(y)
    
    return mx.matmul(x_mx, y_mx)
""",

    "mx.reshape": """
# Using MLX's reshape function
import mlx.core as mx

# Create a test array
a = mx.array([1, 2, 3, 4, 5, 6])

# Reshape to 2x3
b = mx.reshape(a, (2, 3))
print(b)  # array([[1, 2, 3], [4, 5, 6]])

# Flatten with -1
c = mx.reshape(b, (-1,))
print(c)  # array([1, 2, 3, 4, 5, 6])

# IMPORTANT: In Ember ML backend implementation
def reshape_impl(self, x, shape):
    # Ensure input is an MLX array
    x_mx = mx.array(x)
    
    # Convert shape to tuple if it's a list
    shape_tuple = tuple(shape)
    
    return mx.reshape(x_mx, shape_tuple)
""",

    "mx.cast": """
# Using MLX's cast function
import mlx.core as mx

# Create a test array
a = mx.array([1, 2, 3])

# Cast to float32
b = mx.cast(a, mx.float32)
print(b)  # array([1.0, 2.0, 3.0])

# IMPORTANT: In Ember ML backend implementation
def cast_impl(self, x, dtype):
    # Ensure input is an MLX array
    x_mx = mx.array(x)
    
    # Map Ember ML dtype to MLX dtype
    mlx_dtype = self._convert_dtype(dtype)
    
    return mx.cast(x_mx, mlx_dtype)
""",

    "mx.where": """
# Using MLX's where function
import mlx.core as mx

# Create test arrays
condition = mx.array([True, False, True])
x = mx.array([1, 2, 3])
y = mx.array([4, 5, 6])

# Apply where
result = mx.where(condition, x, y)
print(result)  # array([1, 5, 3])

# IMPORTANT: In Ember ML backend implementation
def where_impl(self, condition, x, y):
    # Ensure inputs are MLX arrays
    condition_mx = mx.array(condition, dtype=mx.bool_)
    x_mx = mx.array(x)
    y_mx = mx.array(y)
    
    return mx.where(condition_mx, x_mx, y_mx)
"""
}


# Common pitfalls and solutions
COMMON_PITFALLS = {
    "slice": """
COMMON PITFALLS WITH mx.slice:

1. Using MLX arrays for axes or slice_size parameters:
   ❌ WRONG: mx.slice(tensor, starts, mx.array([0, 1]), mx.array([2, 2]))
   ✅ CORRECT: mx.slice(tensor, starts, [0, 1], [2, 2])

2. Not converting starts to an MLX array:
   ❌ WRONG: mx.slice(tensor, [0, 1], [0, 1], [2, 2])
   ✅ CORRECT: mx.slice(tensor, mx.array([0, 1]), [0, 1], [2, 2])

3. Using Python type conversions on MLX arrays:
   ❌ WRONG: int(mx.array(5))
   ✅ CORRECT: mx.array(5).item() (if absolutely necessary)

4. Confusing with NumPy-style slicing:
   ❌ WRONG: tensor[start_indices:start_indices+slice_size]
   ✅ CORRECT: mx.slice(tensor, mx.array(start_indices), axes, slice_size)

5. Forgetting that slice_size is the size of the slice, not the end indices:
   ❌ WRONG: mx.slice(tensor, mx.array([0, 1]), [0, 1], [3, 4])  # Thinking these are end indices
   ✅ CORRECT: mx.slice(tensor, mx.array([0, 1]), [0, 1], [3, 3])  # These are sizes: 3 rows, 3 columns
""",

    "slice_update": """
COMMON PITFALLS WITH mx.slice_update:

1. Using MLX arrays for axes parameter:
   ❌ WRONG: mx.slice_update(tensor, update, starts, mx.array([0, 1]))
   ✅ CORRECT: mx.slice_update(tensor, update, starts, [0, 1])

2. Not converting starts to an MLX array:
   ❌ WRONG: mx.slice_update(tensor, update, [0, 1], [0, 1])
   ✅ CORRECT: mx.slice_update(tensor, update, mx.array([0, 1]), [0, 1])

3. Providing update tensor with incorrect shape:
   ❌ WRONG: mx.slice_update(tensor, mx.array([10, 11, 12]), mx.array([0, 1]), [0, 1])
   ✅ CORRECT: mx.slice_update(tensor, mx.array([[10, 11]]), mx.array([0, 1]), [0, 1])

4. Forgetting that slice_update modifies a copy, not in-place:
   ❌ WRONG: mx.slice_update(a, update, starts, axes)  # Expecting 'a' to be modified
   ✅ CORRECT: a = mx.slice_update(a, update, starts, axes)  # Assign the result back
""",

    "operators": """
COMMON PITFALLS WITH OPERATORS:

1. Using Python operators directly on MLX arrays:
   ❌ WRONG: a + b
   ✅ CORRECT: mx.add(a, b)

2. Using Python comparison operators:
   ❌ WRONG: a > b
   ✅ CORRECT: mx.greater(a, b)

3. Using Python boolean operators:
   ❌ WRONG: a and b
   ✅ CORRECT: mx.logical_and(a, b)

4. Mixing Python operators and MLX functions:
   ❌ WRONG: mx.add(a, b) * c
   ✅ CORRECT: mx.multiply(mx.add(a, b), c)

5. Assuming operator precedence works the same as in Python:
   ❌ WRONG: mx.add(a, b * c)  # Expecting a + (b * c)
   ✅ CORRECT: mx.add(a, mx.multiply(b, c))

6. Using in-place operators:
   ❌ WRONG: a += b  # MLX arrays are immutable
   ✅ CORRECT: a = mx.add(a, b)

7. Forgetting that MLX operations create new arrays:
   ❌ WRONG: mx.add(a, b)  # Result not stored
   ✅ CORRECT: c = mx.add(a, b)  # Store the result
""",

    "conversion": """
COMMON PITFALLS WITH TYPE CONVERSION:

1. Using Python type conversions:
   ❌ WRONG: float(mx.array(5))
   ✅ CORRECT: mx.array(5).item() (if absolutely necessary)
   ✅ BETTER: mx.cast(mx.array(5), mx.float32)

2. Using astype method:
   ❌ WRONG: a.astype(tensor.float32)
   ✅ CORRECT: mx.cast(a, mx.float32)

3. Using NumPy dtype constants:
   ❌ WRONG: mx.cast(a, tensor.float32)
   ✅ CORRECT: mx.cast(a, mx.float32)

4. Forgetting to convert scalar values to tensors before operations:
   ❌ WRONG: mx.add(a, 5)  # Implicit conversion may not work as expected
   ✅ CORRECT: mx.add(a, mx.array(5))

5. Assuming dtype compatibility across backends:
   ❌ WRONG: mx.cast(a, torch.float32)  # Mixing backend dtypes
   ✅ CORRECT: mx.cast(a, mx.float32)  # Use MLX's own dtype constants

6. Using item() on non-scalar tensors:
   ❌ WRONG: mx.array([1, 2, 3]).item()  # Only works on scalar tensors
   ✅ CORRECT: mx.array(5).item()  # Works on scalar tensors
""",

    "shape_operations": """
COMMON PITFALLS WITH SHAPE OPERATIONS:

1. Using NumPy-style shape access:
   ❌ WRONG: a.shape[0]  # MLX arrays don't have a shape attribute like NumPy
   ✅ CORRECT: mx.shape(a)[0]  # Use mx.shape() function

2. Using reshape with incompatible dimensions:
   ❌ WRONG: mx.reshape(a, (3, 4))  # When a.size != 12
   ✅ CORRECT: mx.reshape(a, (3, 4))  # Only when a.size == 12

3. Using -1 in multiple dimensions:
   ❌ WRONG: mx.reshape(a, (-1, -1))  # Can only use -1 once
   ✅ CORRECT: mx.reshape(a, (-1, 4))  # -1 in one dimension only

4. Forgetting that transpose changes the order of all dimensions:
   ❌ WRONG: mx.transpose(a)  # Expecting to swap only first two dimensions
   ✅ CORRECT: mx.transpose(a, (1, 0, 2))  # For a 3D tensor, specify the exact permutation

5. Using NumPy-style axis specification in reduction operations:
   ❌ WRONG: mx.sum(a, axis=(0, 1))  # MLX expects a single axis or a list of axes
   ✅ CORRECT: mx.sum(a, axis=0)  # For a single axis
   ✅ CORRECT: mx.sum(a, axis=[0, 1])  # For multiple axes

6. Forgetting that expand_dims adds a dimension at the specified position:
   ❌ WRONG: mx.expand_dims(a, 0)  # Expecting it to add at the end
   ✅ CORRECT: mx.expand_dims(a, -1)  # To add at the end
   ✅ CORRECT: mx.expand_dims(a, 0)  # To add at the beginning

7. Using squeeze without specifying an axis on tensors with multiple dimensions of size 1:
   ❌ WRONG: mx.squeeze(a)  # When a has shape (1, 3, 1), removes all size-1 dimensions
   ✅ CORRECT: mx.squeeze(a, axis=0)  # To remove only the first dimension
""",

    "reduction_operations": """
COMMON PITFALLS WITH REDUCTION OPERATIONS:

1. Forgetting that keepdims defaults to False:
   ❌ WRONG: mx.sum(a, axis=0)  # When you need to preserve dimensions
   ✅ CORRECT: mx.sum(a, axis=0, keepdims=True)  # Preserves dimensions

2. Using tuple for axis parameter:
   ❌ WRONG: mx.mean(a, axis=(0, 1))  # MLX doesn't accept tuples for axis
   ✅ CORRECT: mx.mean(a, axis=[0, 1])  # Use a list instead

3. Using negative axis values incorrectly:
   ❌ WRONG: mx.max(a, axis=-3)  # When a has fewer than 3 dimensions
   ✅ CORRECT: mx.max(a, axis=-1)  # Use negative indices that are valid for the tensor's rank

4. Assuming reduction operations preserve dtype:
   ❌ WRONG: mx.sum(mx.array([1, 2, 3], dtype=mx.int32))  # May change dtype
   ✅ CORRECT: mx.cast(mx.sum(mx.array([1, 2, 3], dtype=mx.int32)), mx.int32)  # Explicitly cast

5. Forgetting that argmax/argmin return indices, not values:
   ❌ WRONG: mx.argmax(a, axis=0)  # When you need the maximum values
   ✅ CORRECT: mx.max(a, axis=0)  # For maximum values
   ✅ CORRECT: mx.argmax(a, axis=0)  # For indices of maximum values

6. Using reduction operations on empty tensors:
   ❌ WRONG: mx.sum(mx.array([]))  # May produce unexpected results
   ✅ CORRECT: Check if tensor is empty before reduction

7. Assuming reduction operations always reduce to scalars:
   ❌ WRONG: mx.sum(a)  # Expecting a scalar when a is multi-dimensional and axis is not specified
   ✅ CORRECT: mx.sum(a, axis=None)  # Explicitly specify axis=None to reduce to a scalar
""",

    "logic_operations": """
COMMON PITFALLS WITH LOGIC OPERATIONS:

1. Using Python boolean operators instead of MLX functions:
   ❌ WRONG: a and b  # Python 'and' operator doesn't work element-wise on tensors
   ✅ CORRECT: mx.logical_and(a, b)  # Element-wise AND operation

2. Forgetting to cast boolean conditions to MLX boolean type:
   ❌ WRONG: mx.where(a > b, x, y)  # Comparison result might not be properly typed
   ✅ CORRECT: mx.where(mx.greater(a, b), x, y)  # Use MLX comparison functions

3. Using Python comparison operators:
   ❌ WRONG: a == b  # Python '==' operator doesn't work element-wise on tensors
   ✅ CORRECT: mx.equal(a, b)  # Element-wise equality check

4. Assuming logical operations preserve shape:
   ❌ WRONG: mx.logical_and(a, mx.array(True))  # Broadcasting may change shape
   ✅ CORRECT: Check shapes before and after logical operations

5. Using non-boolean tensors in logical operations:
   ❌ WRONG: mx.logical_and(a, b)  # When a and b are not boolean tensors
   ✅ CORRECT: mx.logical_and(mx.cast(a, mx.bool_), mx.cast(b, mx.bool_))

6. Forgetting that where requires all arguments to be tensors:
   ❌ WRONG: mx.where(condition, 1, 0)  # Scalar values need to be converted to tensors
   ✅ CORRECT: mx.where(condition, mx.array(1), mx.array(0))

7. Using complex conditions without breaking them down:
   ❌ WRONG: mx.where((a > b) & (c < d), x, y)  # Python operators don't work this way
   ✅ CORRECT: mx.where(mx.logical_and(mx.greater(a, b), mx.less(c, d)), x, y)
"""
}


# Ember ML integration examples
EMBER_ML_EXAMPLES = {
    "backend_implementation": """
# Example of implementing a function in the MLX backend

# In ember_ml/backend/mlx/math_ops.py
def sin(self, x):
    \"\"\"MLX implementation of sine function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sine of x
    \"\"\"
    # Convert input to MLX array if needed
    x_mx = mx.array(x)
    
    # Apply the operation
    return mx.sin(x_mx)
""",

    "frontend_abstraction": """
# Example of exposing a function in the frontend

# In ember_ml/ops/__init__.py
sin = lambda x: math_ops().sin(x)

# Usage in application code
from ember_ml import ops

def process_data(data):
    # Convert input to tensor
    tensor = tensor.convert_to_tensor(data)
    
    # Apply operation (will use the appropriate backend)
    return ops.sin(tensor)
"""
}


class NumpyToMLXConverter:
    """A tool for converting NumPy functions to their MLX equivalents."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the converter with data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing NumPy to MLX mappings
        """
        self.csv_path = csv_path
        self.mappings = []
        self.load_data()
        
    def load_data(self) -> None:
        """Load function mappings from the CSV file."""
        if not os.path.exists(self.csv_path):
            print(f"Error: CSV file not found at {self.csv_path}")
            sys.exit(1)
            
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                self.mappings = list(reader)
                
            # Clean up the data (remove quotes, handle None values)
            for mapping in self.mappings:
                for key in mapping:
                    if mapping[key] == "None":
                        mapping[key] = None
                    elif mapping[key] and mapping[key].startswith('"') and mapping[key].endswith('"'):
                        mapping[key] = mapping[key][1:-1]
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            sys.exit(1)
    
    def get_category(self, function_name: str) -> str:
        """
        Determine the category of a function based on its name.
        
        Args:
            function_name: The name of the function
            
        Returns:
            The category name
        """
        if not function_name:
            return "other"
            
        # Remove np. or mx. prefix for matching
        clean_name = function_name.replace("np.", "").replace("mx.", "")
        
        # Check each category's keywords
        for category, keywords in CATEGORIES.items():
            for keyword in keywords:
                if keyword in clean_name:
                    return category
        
        return "other"
    
    def search(self, term: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """
        Search for functions matching the given term and/or category.
        
        Args:
            term: Search term to match against function names
            category: Category to filter by
            
        Returns:
            List of matching function mappings
        """
        results = []
        
        for mapping in self.mappings:
            numpy_cmd = mapping["numpy_command"]
            mlx_cmd = mapping["mlx_equivalent"]
            
            # Skip entries where both are None
            if not numpy_cmd and not mlx_cmd:
                continue
                
            # Filter by search term if provided
            if term:
                term_matches = False
                if numpy_cmd and term.lower() in numpy_cmd.lower():
                    term_matches = True
                elif mlx_cmd and term.lower() in mlx_cmd.lower():
                    term_matches = True
                    
                if not term_matches:
                    continue
            
            # Filter by category if provided
            if category:
                func_category = self.get_category(numpy_cmd or mlx_cmd)
                if func_category != category.lower():
                    continue
            
            results.append(mapping)
            
        return results
    
    def get_example(self, function_name: str) -> str:
        """
        Get a code example for a specific function.
        
        Args:
            function_name: The name of the function
            
        Returns:
            Code example as a string
        """
        # Clean up function name
        clean_name = function_name.strip()
        if clean_name.startswith("np."):
            # Convert np.function to mx.function
            clean_name = "mx." + clean_name[3:]
            
        # Look for exact match
        if clean_name in CODE_EXAMPLES:
            return CODE_EXAMPLES[clean_name]
            
        # Look for partial match
        for key in CODE_EXAMPLES:
            if clean_name in key:
                return CODE_EXAMPLES[key]
                
        # Check for common pitfalls
        for key, pitfall in COMMON_PITFALLS.items():
            if key in clean_name.lower():
                return pitfall
                
        return "No specific example available for this function."
    
    def format_results(self, results: List[Dict]) -> str:
        """
        Format search results for display.
        
        Args:
            results: List of function mappings
            
        Returns:
            Formatted string for display
        """
        if not results:
            return "No matching functions found."
            
        # Group results by category
        by_category = {}
        for result in results:
            numpy_cmd = result["numpy_command"] or ""
            mlx_cmd = result["mlx_equivalent"] or ""
            category = self.get_category(numpy_cmd or mlx_cmd)
            
            if category not in by_category:
                by_category[category] = []
                
            by_category[category].append(result)
        
        # Format the output
        output = []
        
        for category, items in sorted(by_category.items()):
            output.append(f"\n{category.upper()} FUNCTIONS:")
            output.append("-" * 80)
            
            for item in items:
                numpy_cmd = item["numpy_command"] or "N/A"
                mlx_cmd = item["mlx_equivalent"] or "N/A"
                param_diff = item["parameter_differences"] or ""
                notes = item["notes"] or ""
                
                output.append(f"NumPy: {numpy_cmd}")
                output.append(f"MLX:   {mlx_cmd}")
                
                if param_diff:
                    output.append(f"Parameter Differences: {param_diff}")
                    
                if notes:
                    output.append(f"Notes: {notes}")
                    
                # Add example hint
                if mlx_cmd != "N/A":
                    func_name = mlx_cmd.split("(")[0].strip()
                    output.append(f"Example: Use --example {func_name} for code examples")
                    
                output.append("-" * 80)
        
        return "\n".join(output)
    
    def format_example(self, function_name: str) -> str:
        """
        Format a detailed example for a specific function.
        
        Args:
            function_name: The name of the function
            
        Returns:
            Formatted example as a string
        """
        # Find the function in the mappings
        function_info = None
        for mapping in self.mappings:
            numpy_cmd = mapping["numpy_command"] or ""
            mlx_cmd = mapping["mlx_equivalent"] or ""
            
            if function_name in numpy_cmd or function_name in mlx_cmd:
                function_info = mapping
                break
                
        output = []
        
        if function_info:
            numpy_cmd = function_info["numpy_command"] or "N/A"
            mlx_cmd = function_info["mlx_equivalent"] or "N/A"
            param_diff = function_info["parameter_differences"] or ""
            notes = function_info["notes"] or ""
            
            output.append(f"\nFUNCTION DETAILS: {function_name}")
            output.append("=" * 80)
            output.append(f"NumPy: {numpy_cmd}")
            output.append(f"MLX:   {mlx_cmd}")
            
            if param_diff:
                output.append(f"\nParameter Differences: {param_diff}")
                
            if notes:
                output.append(f"\nNotes: {notes}")
        
        # Add code example
        output.append("\nCODE EXAMPLE:")
        output.append("=" * 80)
        output.append(self.get_example(function_name))
        
        # Add Ember ML integration example if relevant
        if "backend_implementation" in EMBER_ML_EXAMPLES:
            output.append("\nEMBER ML INTEGRATION:")
            output.append("=" * 80)
            output.append(EMBER_ML_EXAMPLES["backend_implementation"])
            output.append("\n" + EMBER_ML_EXAMPLES["frontend_abstraction"])
        
        return "\n".join(output)
    
    def print_help(self) -> None:
        """Print help information about the tool."""
        help_text = """
NumPy to MLX Function Converter

This tool helps you find MLX equivalents for NumPy functions and provides code examples
for implementing them correctly in the Ember ML framework.

Usage:
    python numpy_to_mlx.py [search_term]
    python numpy_to_mlx.py --category CATEGORY
    python numpy_to_mlx.py --example FUNCTION
    
Examples:
    python numpy_to_mlx.py                  # List all available functions
    python numpy_to_mlx.py array            # Search for functions containing "array"
    python numpy_to_mlx.py "np.sin"         # Look up the exact function
    python numpy_to_mlx.py --category math  # List functions in the math category
    python numpy_to_mlx.py --example slice  # Show detailed example for slice
    
Available categories:
    - creation    (array, zeros, ones, etc.)
    - indexing    (take, slice, etc.)
    - math        (add, subtract, sin, cos, etc.)
    - reduction   (sum, max, min, mean, etc.)
    - shape       (reshape, transpose, etc.)
    - logic       (where, equal, greater, etc.)
    - manipulation (pad, tile, repeat, etc.)
    - linalg      (norm, inv, solve, etc.)
    - fft         (fft, ifft, etc.)
    - conversion  (astype, cast, etc.)
    - properties  (T, copy, etc.)
    - io          (save, load, etc.)
        """
        print(help_text)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Convert NumPy functions to MLX equivalents")
    parser.add_argument("search_term", nargs="?", help="Function name or pattern to search for")
    parser.add_argument("--category", help="Filter by function category")
    parser.add_argument("--example", help="Show detailed example for a specific function")
    parser.add_argument("--help-categories", action="store_true", help="Show available categories")
    
    args = parser.parse_args()
    
    # Determine the path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "docs", "tutorials", "numpyconversiondata.csv")
    
    converter = NumpyToMLXConverter(csv_path)
    
    if args.help_categories:
        print("Available categories:")
        for category, keywords in sorted(CATEGORIES.items()):
            print(f"  - {category}: {', '.join(keywords[:5])}...")
        return
        
    if args.example:
        print(converter.format_example(args.example))
        return
        
    results = converter.search(args.search_term, args.category)
    
    if not args.search_term and not args.category:
        converter.print_help()
        print("\nShowing all available functions:")
    
    print(converter.format_results(results))


if __name__ == "__main__":
    main()