#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Instructions to install LLaMA (Ollama)
echo "Please install LLaMA (Ollama) manually from the official source and ensure it's accessible in your PATH."

# Run database migrations
flask db upgrade

echo "Installation complete. You can now run DocuOracle."
