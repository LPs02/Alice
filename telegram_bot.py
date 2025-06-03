import requests
import os

TOKEN = os.getenv("TELEGRAM_TOKEN")
URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

def send_telegram_message(text, chat_id=None):
    if chat_id is None:
        print("Chat ID não fornecido.")
        return

    payload = {
        'chat_id': chat_id,
        'text': text
    }
    requests.post(URL, json=payload)
