from fastapi import FastAPI, Request, Response
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import asyncio
from mensagem import process_input

import logging
logging.basicConfig(level=logging.INFO)

TOKEN = "8129764087:AAFXieX5qd1-pnsafwKcFuFxR08OGh_vLB8"

app = FastAPI()
bot = Bot(token=TOKEN)
application = ApplicationBuilder().token(TOKEN).build()

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)
    await application.process_update(update)
    return Response(status_code=200)

async def start(update, context):
    await update.message.reply_text("Olá! Eu sou a Alice, sua assistente virtual. Como posso ajudar?")

async def handle_message(update, context):
    user_id = str(update.message.from_user.id)
    user_text = update.message.text
    response = process_input(user_id, user_text)
    await update.message.reply_text(response)

application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
