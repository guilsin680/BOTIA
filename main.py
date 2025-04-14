import joblib
import pandas as pd
from telegram import Bot
import os

# Definindo variáveis de ambiente (substitua pelos valores corretos)
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Carregar o modelo treinado
model = joblib.load("model.pkl")

# Função de previsão de resultado
def prever_resultado(jogo):
    # O modelo espera as seguintes variáveis de entrada:
    # 'home_goals', 'away_goals', 'home_team_rank', 'away_team_rank'
    dados_jogo = pd.DataFrame([jogo])
    
    # Fazer a previsão
    resultado = model.predict(dados_jogo)
    confianca = model.predict_proba(dados_jogo).max()  # probabilidade máxima para confiança
    
    return resultado[0], confianca

# Função para enviar a previsão via Telegram
def enviar_previsao(resultado, confianca, jogo):
    bot = Bot(token=TOKEN)
    
    # Formatar a mensagem
    if resultado == 0:
        resultado_str = "Vitória do time da casa"
    elif resultado == 1:
        resultado_str = "Empate"
    else:
        resultado_str = "Vitória do time visitante"
    
    mensagem = (
        f"Previsão do Jogo:\n"
        f"Data e Hora: {jogo['date']}\n"
        f"Times: {jogo['home_team']} vs {jogo['away_team']}\n"
        f"Competição: {jogo['league']}\n"
        f"Resultado esperado: {resultado_str}\n"
        f"Confiança na previsão: {confianca:.2f}\n"
    )
    
    # Enviar a mensagem
    bot.send_message(chat_id=CHAT_ID, text=mensagem)

# Exemplo de jogo (substitua com dados reais)
jogo_exemplo = {
    "home_goals": 2,
    "away_goals": 1,
    "home_team_rank": 5,
    "away_team_rank": 8,
    "date": "2025-04-14 16:00",
    "home_team": "Time A",
    "away_team": "Time B",
    "league": "Campeonato Nacional"
}

# Fazer a previsão
resultado, confianca = prever_resultado(jogo_exemplo)

# Enviar a previsão para o Telegram
enviar_previsao(resultado, confianca, jogo_exemplo)
