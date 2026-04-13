#!/bin/bash

echo "Starting Jarvis..."

cd ~/jarvis
source .venv/bin/activate

echo "Starting Streamlit..."
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &

sleep 3

echo "Starting Cloudflare tunnel..."
nohup cloudflared tunnel run jarvis > tunnel.log 2>&1 &

echo "Jarvis is up 🚀"
