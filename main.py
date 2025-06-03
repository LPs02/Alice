# main.py
from fastapi import FastAPI, Request
from Neural import ALMA
from telegram_bot import send_telegram_message
from MINE import send_minecraft_command

app = FastAPI()
alice = ALMA()
alice.load("alice_model.pt")

@app.post("/interact")
async def interact(request: Request):
    data = await request.json()
    source = data.get("source")  # "telegram" ou "minecraft"
    message = data.get("message")

    response, action, value = alice(message)

    # Pode reagir conforme a origem da mensagem
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
