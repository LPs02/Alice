from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

HF_API_URL = "https://api-inference.huggingface.co/models/Lps02/Alice_finetuned"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

async def query_huggingface(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
        data = response.json()
        # A resposta pode ser uma lista com o texto gerado
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        # Se der erro, pode vir com um campo 'error'
        if isinstance(data, dict) and "error" in data:
            return "Desculpe, estou temporariamente indisponível."
        return "Não consegui gerar uma resposta."

async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": text}
        )

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"error": "No message found"}
    
    user_from = message.get("from")
    if not user_from:
        return {"error": "No 'from' field in message"}
    
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    if not text:
        return {"status": "ok"}  # mensagem sem texto
    
    resposta = await query_huggingface(text)
    await send_message(chat_id, resposta)

    return {"status": "ok"}
