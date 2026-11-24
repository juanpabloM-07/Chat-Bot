# MotoInfoBot

Bot de Telegram desarrollado en Python para consultar fichas técnicas de motocicletas mediante un catálogo interactivo paginado, búsqueda por texto y reconocimiento de imágenes con Gemini.

## Ficha del taller

* **Nombre del bot:** MotoInfoBot
* **Lenguaje:** Python
* **Librerías principales:** `python-telegram-bot`, `google-genai`, `Pillow`
* **Fuente de datos local:** `chatbot.json`
* **Imágenes del catálogo:** Carpeta `imagenes/` (archivos `.jpg` / `.png` configurados en `chatbot.json`)
* **Comandos disponibles:** `/start`, `/catalogo`, `/ayuda`, `/salir`

## Como Ejecutar

* **Activa el entorno virtual en PowerShell** 
.venv\Scripts\Activate.ps1

* **Configura las variables de entorno**
$env:TELEGRAM_TOKEN="tu_token_de_telegram"
$env:GEMINI_API_KEY="tu_api_key_de_gemini"

* **Inicia el bot**
python bot.py


## Estructura del proyecto

```text
ChatBot/
├── .venv/
├── bot.py
├── chatbot.json
├── README.md
└── imagenes/
