import os
import sys

BASE = os.path.dirname(__file__)

# Убираем несуществующий "src" и гарантируем доступ к пакету skillpilot
if BASE not in sys.path:
    sys.path.append(BASE)

from skillpilot.ui.app import ui

def main():
    app = ui()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    # Gradio v4.x: очередь включена для стабильности под нагрузкой
    app.queue().launch(server_name=host, server_port=port)

if __name__ == "__main__":
    main()
