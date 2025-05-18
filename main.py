from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

HF_API_URL = "https://huggingface.co/Lps02/Alice_finetuned"
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
    
    # Exemplo básico para extrair a mensagem e o user_id de forma segura
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"error": "No message found"}
    
    user_from = message.get("from")
    if not user_from:
        return {"error": "No 'from' field in message"}
    
    user_id = user_from.get("id")
    if not user_id:
        return {"error": "No user ID found"}
    
    # Aqui você pode seguir com a lógica para responder ou processar a mensagem
    text = message.get("text", "")
    # ... seu processamento e resposta

    return {"status": "ok"}
