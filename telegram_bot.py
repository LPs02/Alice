import requests

TELEGRAM_BOT_TOKEN = "8129764087:AAFXieX5qd1-pnsafwKcFuFxR08OGh_vLB8"
TELEGRAM_CHAT_ID = "5322238901"

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    requests.post(url, json=payload)
