from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelWithLMHead

import warnings

warnings.filterwarnings('ignore')

BOT_USERNAME: Final = "Insert your bot username"
TOKEN: Final = "Insert your token"


# Initializing the model

checkpoint_path = "Insert your checkpoint path"
base_model = "tinkoff-ai/ruDialoGPT-medium"

device = "cpu"
model = AutoModelWithLMHead.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model)


# Generation

def process_model_response(response: str) -> str:
    response = response.capitalize()
    response = response.replace("\n", " ")
    return response


def get_model_response(prompt: str) -> str:
    full_prompt = f"@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ {prompt} @@ВТОРОЙ@@"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=3,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]

    response = context_with_response[2][len(full_prompt)+1:]
    response = response[:response.find("<pad>")]
    response = process_model_response(response)
    return response


# Commands

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот, знающий все о жизни чата HSE ПМИ 2020")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Можешь мне уже что-нибудь написать или задать вопрос")


# Messages

def handle_response(text: str) -> str:
    return get_model_response(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    if message_type == "group":
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, "").strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)

    await update.message.reply_text(response)

# Errors

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


if __name__ == "__main__":
    START_MESSAGE = "Начинаю вспоминать русский язык..."
    print(START_MESSAGE)
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    # Polls the bot
    POLLING_MESSAGE = "Думаю..."
    print(POLLING_MESSAGE)
    app.run_polling(poll_interval=3)
