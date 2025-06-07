#!/bin/bash
# This script applies the fix to the notebook

# Check if a notebook file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <notebook_file>"
    exit 1
fi

NOTEBOOK_FILE=$1

# Check if the notebook file exists
if [ ! -f "$NOTEBOOK_FILE" ]; then
    echo "Error: Notebook file $NOTEBOOK_FILE not found."
    exit 1
fi

# Apply the fix
echo "Applying fix to $NOTEBOOK_FILE..."
python update_notebook.py "$NOTEBOOK_FILE"

# Check if the fixed notebook was created
FIXED_NOTEBOOK="${NOTEBOOK_FILE%.*}_fixed.${NOTEBOOK_FILE##*.}"
if [ -f "$FIXED_NOTEBOOK" ]; then
    echo "Fix applied successfully. Fixed notebook saved to $FIXED_NOTEBOOK"
    echo "You can now open the fixed notebook and run it."
else
    echo "Error: Failed to create fixed notebook."
    exit 1
fi

echo "Done!"