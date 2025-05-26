# utils_api.py

from datetime import datetime
import requests

def get_current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_current_date():
    today = datetime.now()
    return today.strftime("%Y-%m-%d")

def get_weather(city="São Paulo", api_key=None):
    if not api_key:
        return "API de clima não configurada."

    url = f"http://api.weatherapi.com/v1/current.json?key=950fcafd74784ea6b0731049252605&q=São João de meriti&lang=pt"
    try:
        response = requests.get(url)
        data = response.json()
        condition = data['current']['condition']['text']
        temp_c = data['current']['temp_c']
        return f"O clima em {city} está {condition.lower()} com {temp_c}°C."
    except Exception as e:
        return f"Erro ao obter clima: {str(e)}"

def get_custom_response(text: str):
    if "horas" in text or "hora" in text:
        return f"Agora são {get_current_time()}."
    elif "data" in text:
        return f"Hoje é {get_current_date()}."
    elif "clima" in text or "tempo" in text:
        return get_weather()  # você pode passar a chave se tiver
    else:
        return None
