import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"

system_prompt = """
You are Jarvis, Alex's local AI assistant.
You are running locally through Ollama on this machine and are being accessed from WSL on Windows.
Do not deny or question this runtime context unless the user explicitly asks for troubleshooting.
Be concise, practical, and direct.
Prefer short answers by default.
When the user asks about their setup, rely on the context they already gave in the chat.
Do not ask for logs, screenshots, or command output unless it is truly needed to fix a problem.
"""

messages = [{"role": "system", "content": system_prompt}]

print("Jarvis is online. Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Jarvis: Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=300
        )
        response.raise_for_status()
        data = response.json()

        reply = data["message"]["content"].strip()
        print(f"\nJarvis: {reply}\n")

        messages.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        print("\nJarvis: Stopped.")
        break
    except Exception as e:
        print(f"\nJarvis: Error -> {e}\n")
