#!/bin/bash

# Ensure a .py file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_file.py>"
    exit 1
fi

# Get the full path and base name of the input Python file
PYTHON_FILE=$(realpath "$1")
BASENAME=$(basename "$PYTHON_FILE" .py)
DIRNAME=$(dirname "$PYTHON_FILE")

# Define the full path for the output notebook and PDF files
NOTEBOOK_FILE="${DIRNAME}/${BASENAME}.ipynb"
HTML_FILE="${DIRNAME}/${BASENAME}.html"

# Check if jupytext and jupyter are installed
command -v jupytext >/dev/null 2>&1 || { echo >&2 "Jupytext is required but it's not installed. Aborting."; exit 1; }
command -v jupyter >/dev/null 2>&1 || { echo >&2 "Jupyter is required but it's not installed. Aborting."; exit 1; }

# Step 1: Convert the .py file to a Jupyter notebook (.ipynb)
echo "Converting $PYTHON_FILE to $NOTEBOOK_FILE using jupytext..."
jupytext --to ipynb "$PYTHON_FILE"

# Step 2: Execute the notebook and convert it to a PDF
echo "Executing $NOTEBOOK_FILE and converting to $HTML_FILE using nbconvert..."
jupyter nbconvert --to html --execute "$NOTEBOOK_FILE"

# Step 3: Delete the intermediate notebook file
echo "Deleting the intermediate notebook file $NOTEBOOK_FILE..."
rm "$NOTEBOOK_FILE"

echo "Conversion complete. The HTML is saved as $HTML_FILE"
