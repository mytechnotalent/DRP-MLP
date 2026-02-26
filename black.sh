#!/bin/bash

# Format Python files with black
echo "Running black on Python files..."
black --quiet *.py
# Format Jupyter notebooks with black via nbqa
echo "Running black on Jupyter notebooks..."
nbqa black --quiet *.ipynb
echo "Done! All Python files and notebooks formatted."
