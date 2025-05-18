from fastapi import FastAPI, Request
import httpx
import os
import logging

app = FastAPI()

# Configura o nível de log
logging.basicConfig(level=logging.INFO)

# Tokens e URLs
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Verificação básica
if not TELEGRAM_BOT_TOKEN or not HF_API_TOKEN:
    raise RuntimeError("Os tokens do Telegram ou Hugging Face não foram configurados.")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
HF_API_URL = "https://api-inference.huggingface.co/models/Lps02/Alice-model"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Consulta ao modelo Hugging Face
async def query_huggingface(prompt: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={"inputs": prompt},
                timeout=30.0
            )
            data = response.json()
        except Exception as e:
            logging.error(f"Erro ao consultar a Hugging Face: {e}")
            return "Desculpe, houve um erro ao me conectar com o servidor de IA."

        # Trata diferentes formatos de resposta
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "error" in data:
            logging.warning(f"Erro da Hugging Face: {data['error']}")
            return "Desculpe, estou temporariamente indisponível."
        
        return "Não consegui entender a resposta da IA."

# Envia mensagem ao Telegram
async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": text}
        )

# Webhook do Telegram
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    logging.info(f"Mensagem recebida: {data}")
    
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"error": "No message found"}
    
    user_from = message.get("from")
    if not user_from:
        return {"error": "No 'from' field in message"}
    
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    
    if not text:
        return {"status": "ok"}  # ignora mensagens sem texto

    # Ignora comandos como /start
    if text.startswith("/"):
        return {"status": "ok"}

    # Consulta IA e envia resposta
    resposta = await query_huggingface(text)
    await send_message(chat_id, resposta)

    return {"status": "ok"}
