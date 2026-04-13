#!/bin/bash

echo "🚀 Starting Jarvis..."

# Start Ollama (if not already running)
pgrep ollama > /dev/null
if [ $? -ne 0 ]; then
    echo "Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
else
    echo "Ollama already running."
fi

# Go to project
cd ~/jarvis

# Activate environment
source .venv/bin/activate

# Start Streamlit
echo "Launching Web UI..."
streamlit run app.py --server.address 0.0.0.0
