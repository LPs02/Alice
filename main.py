from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_USER_ID = int(os.getenv("TELEGRAM_USER_ID"))
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# URL do Hugging Face Space que vai gerar a resposta
HUGGINGFACE_SPACE_URL = os.getenv("HUGGINGFACE_SPACE_URL")  # ex: "https://your-username-your-space-name.hf.space"

async def generate_response(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{HUGGINGFACE_SPACE_URL}/generate",
                json={"prompt": prompt}
            )
            if response.status_code == 200:
                return response.json().get("response", "Desculpe, não entendi.")
            else:
                return "Erro ao gerar resposta via IA."
        except Exception as e:
            return "Erro na comunicação com o servidor de IA."

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

    if user_id != AUTHORIZED_USER_ID:
        await send_message(user_id, "Desculpe, você não está autorizado a usar a Alice.")
        return {"ok": True}

    prompt = f"User: {user_msg}\nAlice:"
    bot_resp = await generate_response(prompt)

    await send_message(user_id, bot_resp)
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "Alice backend está rodando no Render!"}
