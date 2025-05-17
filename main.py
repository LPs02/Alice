from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

async def query_huggingface(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
        data = response.json()
        if isinstance(data, list):
            return data[0]["generated_text"]
        else:
            return "Desculpe, não consegui gerar uma resposta."

async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": text}
        )

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data:
        return {"ok": True}

    message = data["message"]
    user_id = message["from"]["id"]
    user_msg = message.get("text", "")

    # Envia o prompt para o Hugging Face
    response_text = await query_huggingface(user_msg)

    # Envia a resposta para o Telegram
    await send_message(user_id, response_text)

    return {"ok": True}
