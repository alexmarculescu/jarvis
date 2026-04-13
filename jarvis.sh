#!/bin/bash

case "$1" in

up)
    echo "🚀 Starting Jarvis..."

    # Start Ollama if not running
    if ! pgrep -x "ollama" > /dev/null
    then
        echo "🧠 Starting Ollama..."
        ollama serve > /dev/null 2>&1 &
    else
        echo "🧠 Ollama already running"
    fi

    # Activate environment
    source ~/jarvis/.venv/bin/activate

    # Kill old Streamlit if exists
    fuser -k 8501/tcp > /dev/null 2>&1

    # Start Streamlit in background
    echo "🌐 Starting Jarvis UI..."
    nohup streamlit run ~/jarvis/app.py --server.address 0.0.0.0 --server.port 8501 > ~/jarvis/jarvis.log 2>&1 &

    echo "✅ Jarvis is running"
    echo "🌍 Open: http://100.77.229.32:8501"
    ;;

down)
    echo "🛑 Stopping Jarvis..."

    # Kill Streamlit
    fuser -k 8501/tcp > /dev/null 2>&1

    # Kill Ollama
    pkill ollama > /dev/null 2>&1

    echo "✅ Jarvis stopped"
    ;;

*)
    echo "Usage: jarvis up | jarvis down"
    ;;

esac
