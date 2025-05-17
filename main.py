from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# CONFIGURAÇÕES - coloque suas variáveis de ambiente
TELEGRAM_BOT_TOKEN = os.getenv("8129764087:AAFXieX5qd1-pnsafwKcFuFxR08OGh_vLB8")  # seu token do bot Telegram
AUTHORIZED_USER_ID = int(os.getenv("TELEGRAM_USER_ID"))
""
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Carregar modelo e tokenizer (no início do main.py ou outro módulo)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_response(prompt: str, max_length: int = 100) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Opcional: limpar resposta (retirar o prompt da saída, se necessário)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

# Memória curta: dicionário user_id -> lista de mensagens (string)
# Guarda só as últimas 10 mensagens (5 trocas)
memory_short = {}

# Para salvar as conversas num arquivo (append)
CONVERSATION_LOG_FILE = "conversation_log.txt"

class TelegramUpdate(BaseModel):
    update_id: int
    message: dict = None

def update_memory(user_id: int, user_msg: str, bot_resp: str, max_len=10):
    history = memory_short.get(user_id, [])
    history.append(f"User: {user_msg}")
    history.append(f"Alice: {bot_resp}")
    # mantém só as últimas max_len mensagens
    if len(history) > max_len * 2:
        history = history[-max_len*2 :]
    memory_short[user_id] = history

def get_prompt(user_id: int):
    history = memory_short.get(user_id, [])
    return "\n".join(history) + "\nAlice: "

def save_conversation(user_id: int, user_msg: str, bot_resp: str):
    with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User({user_id}): {user_msg}\n")
        f.write(f"Alice: {bot_resp}\n\n")

async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": text}
        )
        return resp

# Rota para webhook do Telegram
@app.post("/webhook")
async def webhook(update: dict):
    message = update['message']['text']
    chat_id = update['message']['chat']['id']

    if str(chat_id) != str(AUTHORIZED_USER_ID):
        return {"status": "unauthorized"}

    response = generate_response(message)
    send_message(chat_id, response)
    return {"status": "ok"}

    # Aqui você colocaria a chamada ao modelo de IA (GPT2, etc)
    # Por enquanto, responde ecoando a mensagem + memória curta
    prompt = get_prompt(user_id) + user_msg + "\nAlice: "
    # Exemplo simples: responde com "Você disse: {mensagem}"
    bot_resp = f"Você disse: {user_msg}"

    # Atualiza a memória curta
    update_memory(user_id, user_msg, bot_resp)
    # Salva no log
    save_conversation(user_id, user_msg, bot_resp)
    # Envia a resposta para o Telegram
    await send_message(user_id, bot_resp)

    return {"ok": True}

# Rota teste simples
@app.get("/")
async def root():
    return {"message": "Alice bot está no ar!"}
