import os, sys
BASE = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE, "src"))
from skillpilot.ui.app import ui

def main():
    app = ui()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    # Gradio v4.x: без аргументов (или используйте default_concurrency_limit, если нужен лимит)
    app.queue().launch(server_name=host, server_port=port)

if __name__ == "__main__":
    main()
