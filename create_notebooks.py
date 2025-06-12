import json
import os

# Create empty notebook template
empty_notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Create 79 notebooks
for i in range(1, 80):
    filename = f"processing/eeg{i}.ipynb"
    with open(filename, 'w') as f:
        json.dump(empty_notebook, f, indent=1) 