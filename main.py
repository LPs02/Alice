# main.py
from fastapi import FastAPI, Request
from Neural import ALMA
from telegram_bot import send_telegram_message
from MINE import send_minecraft_command
import requests
import os

app = FastAPI()
alice = ALMA()
alice.load("alice_model.pt")

TELEGRAM_TOKEN = "8129764087:AAFXieX5qd1-pnsafwKcFuFxR08OGh_vLB8"
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'

def process_message(text):
    response, action, value = alice(text)
    return response

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    if "message" in data:
        chat_id = data['message']['chat']['id']
        message_text = data['message'].get('text', '')

        response_text = process_message(message_text)

        payload = {
            'chat_id': chat_id,
            'text': response_text
        }
        requests.post(TELEGRAM_API_URL, json=payload)

    return {"status": "ok"}

@app.post("/interact")
async def interact(request: Request):
    data = await request.json()
    source = data.get("source")  # "telegram" ou "minecraft"
    message = data.get("message")

    response, action, value = alice(message)

    if source == "telegram":
        send_telegram_message(response)
    elif source == "minecraft":
        send_minecraft_command(response)

    return {"response": response, "action": action, "value": value.item()}

@app.post("/reward")
async def reward(request: Request):
    data = await request.json()
    message = data.get("message")
    action_taken = data.get("action")
    reward = data.get("reward")
    old_value = data.get("old_value")

    alice.train_step(message, action_taken, reward, old_value)
    alice.save("alice_model.pt")
    return {"status": "trained"}
