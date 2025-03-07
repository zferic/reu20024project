#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p papers
mkdir -p vectorstore

# Generate SSL certificates if they don't exist
if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
fi

# Start the backend server
python main.py 