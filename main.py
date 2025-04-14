import pandas as pd
import requests
import joblib
import os
from dotenv import load_dotenv
from datetime import datetime
from telegram import Bot

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = Bot(token=TOKEN)

# Função de previsão usando modelo treinado
def prever_resultado(jogo):
    modelo = joblib.load("model.pkl")
    features = jogo  # Ex: [gols_mandante, gols_visitante]
    resultado = modelo.predict([features])[0]
    probas = modelo.predict_proba([features])[0]
    confianca = round(max(probas) * 100, 2)
    return resultado, confianca

# Exemplo de uso com jogo genérico (substituir por dados reais da API)
jogo_exemplo = [1, 0]
resultado, confianca = prever_resultado(jogo_exemplo)

mensagem = f"""
📊 *Previsão de Resultado com IA*

🏟️ *Jogo:* Time A vs Time B
📅 *Data:* {datetime.now().strftime('%d/%m/%Y')}
⏰ *Horário:* {datetime.now().strftime('%H:%M')}
🔍 *Previsão:* *{resultado.upper()}*
📈 *Confiança:* {confianca}%

🤖 _Bot de Apostas com IA_
"""

bot.send_message(chat_id=CHAT_ID, text=mensagem, parse_mode="Markdown")