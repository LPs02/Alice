from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# CONFIGURAÇÕES - coloque suas variáveis de ambiente
TELEGRAM_BOT_TOKEN = os.getenv("8129764087:AAFXieX5qd1-pnsafwKcFuFxR08OGh_vLB8")  # seu token do bot Telegram
AUTHORIZED_USER_ID = int(os.getenv("5322238901"))  # seu user id no Telegram (int)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

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
async def telegram_webhook(update: TelegramUpdate):
    if not update.message:
        return {"ok": True}

    user_id = update.message["from"]["id"]
    user_msg = update.message.get("text", "")

    if user_id != AUTHORIZED_USER_ID:
        # Ignora mensagens de usuários não autorizados
        return {"ok": True}

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
