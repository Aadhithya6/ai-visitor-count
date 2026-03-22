#!/bin/bash

echo "=========================================="
echo "Initializing Python Environment (Linux/Mac)"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment and install dependencies
echo "Activating .venv and installing dependencies..."
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Dependency installation failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup successful!"
echo "To activate the environment manually, run:"
echo "source .venv/bin/activate"
echo "=========================================="
