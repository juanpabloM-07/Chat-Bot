import io
import json
import logging
import math
import os
import re
import socket
import time
import unicodedata
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
log = logging.getLogger("MotoInfoBot")

# Forzar resolución IPv4 para evitar ConnectTimeout en redes con IPv6 problemático
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(*args, **kwargs):
    resultados = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in resultados if r[0] == socket.AF_INET]
    return ipv4 or resultados


socket.getaddrinfo = _ipv4_only_getaddrinfo

RUTA_BASE = Path(__file__).parent
RUTA_JSON = Path(__file__).with_name("chatbot.json")
MOTOS_POR_PAGINA = 8
CODIGOS_REINTENTO = {429, 500, 502, 503, 504}
MAX_CAPTION_TELEGRAM = 1024


def cargar_datos(ruta: Path) -> dict:
    with ruta.open(encoding="utf-8") as f:
        return json.load(f)


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin caracteres especiales para comparar nombres."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", texto)


DATOS = cargar_datos(RUTA_JSON)

# Caché en memoria: clave normalizada -> ficha técnica
CACHE_FICHAS: dict[str, str] = {}
for moto in DATOS.get("motos_catalogo", []):
    CACHE_FICHAS[normalizar(moto["titulo"])] = moto["ficha"]
    for alias in moto.get("alias", []):
        CACHE_FICHAS[normalizar(alias)] = moto["ficha"]

cliente_gemini = genai.Client(
    http_options=types.HttpOptions(timeout=30000)
)
MODELOS_GEMINI: list[str] = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]


class ServicioSaturadoError(Exception):
    """Se lanza cuando todos los modelos y reintentos fallan por alta demanda."""


# ---------- Búsqueda local antes de llamar a la API ----------
def buscar_en_catalogo(consulta: str) -> str | None:
    limpia = " ".join(normalizar(consulta).split())
    if not limpia:
        return None

    if limpia in CACHE_FICHAS:
        return CACHE_FICHAS[limpia]

    for clave, ficha in CACHE_FICHAS.items():
        if len(clave) >= 4 and clave in limpia:
            return ficha
    return None


def extraer_encabezado_moto(ficha: str) -> str | None:
    """Extrae 'Marca y Modelo' de la primera línea si sigue el formato esperado."""
    primera_linea = ficha.strip().splitlines()[0] if ficha.strip() else ""
    match = re.match(r"^La\s+(.+?)\s+es\s+", primera_linea)
    return match.group(1).strip() if match else None


# ---------- Teclado de catálogo paginado (2 columnas) ----------
def teclado_catalogo(pagina: int = 0) -> InlineKeyboardMarkup:
    motos = DATOS.get("motos_catalogo", [])
    total_paginas = max(1, math.ceil(len(motos) / MOTOS_POR_PAGINA))
    pagina = max(0, min(pagina, total_paginas - 1))

    inicio = pagina * MOTOS_POR_PAGINA
    sublista = motos[inicio : inicio + MOTOS_POR_PAGINA]

    filas: list[list[InlineKeyboardButton]] = []
    fila_actual: list[InlineKeyboardButton] = []

    for moto in sublista:
        fila_actual.append(
            InlineKeyboardButton(moto["titulo"], callback_data=f"moto:{moto['id']}")
        )
        if len(fila_actual) == 2:
            filas.append(fila_actual)
            fila_actual = []

    if fila_actual:
        filas.append(fila_actual)

    # Botones de paginación si hay más de una página
    if total_paginas > 1:
        navegacion: list[InlineKeyboardButton] = []
        if pagina > 0:
            navegacion.append(
                InlineKeyboardButton("⬅️ Anterior", callback_data=f"pag:{pagina - 1}")
            )
        navegacion.append(
            InlineKeyboardButton(
                f"📄 {pagina + 1}/{total_paginas}", callback_data="noop"
            )
        )
        if pagina < total_paginas - 1:
            navegacion.append(
                InlineKeyboardButton("Siguiente ➡️", callback_data=f"pag:{pagina + 1}")
            )
        filas.append(navegacion)

    return InlineKeyboardMarkup(filas)


# ---------- Llamada resiliente a Gemini (reintentos + fallback) ----------
def generar_contenido_con_reintentos(
    contents, config: types.GenerateContentConfig, reintentos_por_modelo: int = 3
):
    ultimo_error: Exception | None = None
    hubo_saturacion = False

    for indice_modelo, modelo in enumerate(MODELOS_GEMINI):
        for intento in range(1, reintentos_por_modelo + 1):
            try:
                return cliente_gemini.models.generate_content(
                    model=modelo,
                    contents=contents,
                    config=config,
                )
            except APIError as e:
                ultimo_error = e
                codigo = getattr(e, "code", None)
                if codigo in CODIGOS_REINTENTO:
                    hubo_saturacion = True
                    espera = 2 ** (intento - 1)  # 1s, 2s, 4s
                    log.warning(
                        "Modelo %s devolvió %s (intento %d/%d). Reintentando en %ds...",
                        modelo,
                        codigo,
                        intento,
                        reintentos_por_modelo,
                        espera,
                    )
                    time.sleep(espera)
                else:
                    raise
            except Exception as e:
                ultimo_error = e
                log.warning(
                    "Error inesperado en modelo %s (intento %d/%d): %s",
                    modelo,
                    intento,
                    reintentos_por_modelo,
                    e,
                )
                time.sleep(1)

        if indice_modelo + 1 < len(MODELOS_GEMINI):
            log.info(
                "Cambiando al modelo de respaldo: %s", MODELOS_GEMINI[indice_modelo + 1]
            )

    if hubo_saturacion:
        raise ServicioSaturadoError(str(ultimo_error))
    raise ultimo_error or RuntimeError("No se obtuvo respuesta de Gemini.")


# ---------- Consultas a Gemini ----------
def consultar_moto_texto(nombre_moto: str) -> str:
    ficha_local = buscar_en_catalogo(nombre_moto)
    if ficha_local:
        return ficha_local

    config = types.GenerateContentConfig(
        system_instruction=DATOS["prompt_sistema"],
        temperature=0.3,
    )
    respuesta = generar_contenido_con_reintentos(
        contents=f"Dame la ficha técnica de la siguiente motocicleta: {nombre_moto}",
        config=config,
    )
    texto_generado = (respuesta.text or "").strip()
    if texto_generado:
        CACHE_FICHAS[normalizar(nombre_moto)] = texto_generado
    return texto_generado


def consultar_moto_imagen(imagen_bytes: bytes, comentario: str | None = None) -> str:
    prompt_usuario = "Identifica la motocicleta en esta imagen y dame su ficha técnica."
    if comentario:
        prompt_usuario += f" Nota adicional del usuario: {comentario}"

    config = types.GenerateContentConfig(
        system_instruction=DATOS["prompt_sistema"],
        temperature=0.3,
    )

    with Image.open(io.BytesIO(imagen_bytes)) as imagen:
        respuesta = generar_contenido_con_reintentos(
            contents=[imagen, prompt_usuario],
            config=config,
        )

    texto_generado = (respuesta.text or "").strip()

    if "NO_ES_MOTO" not in texto_generado:
        modelo_detectado = extraer_encabezado_moto(texto_generado)
        if modelo_detectado:
            CACHE_FICHAS[normalizar(modelo_detectado)] = texto_generado

    return texto_generado


# ---------- Envío seguro a Telegram ----------
async def enviar_respuesta_formateada(mensaje_objetivo, texto: str) -> None:
    try:
        await mensaje_objetivo.edit_text(texto, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await mensaje_objetivo.edit_text(texto)


async def enviar_ficha_con_foto(
    mensaje_destino, ficha: str, imagen_ref: str | None
) -> None:
    """Envía la foto de la moto junto con su ficha técnica y vuelve a mostrar el menú."""
    foto_cargada = None
    archivo_abierto = None
    teclado = teclado_catalogo(0)

    if imagen_ref:
        if imagen_ref.startswith(("http://", "https://")):
            foto_cargada = imagen_ref
        else:
            ruta_foto = (RUTA_BASE / imagen_ref).resolve()
            if ruta_foto.is_file():
                archivo_abierto = ruta_foto.open("rb")
                foto_cargada = archivo_abierto
            else:
                log.warning("Imagen local no encontrada para la moto: %s", ruta_foto)

    try:
        if foto_cargada is not None:
            if len(ficha) <= MAX_CAPTION_TELEGRAM:
                try:
                    await mensaje_destino.reply_photo(
                        photo=foto_cargada,
                        caption=ficha,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=teclado,
                    )
                    return
                except Exception:
                    if archivo_abierto:
                        archivo_abierto.seek(0)
                    try:
                        await mensaje_destino.reply_photo(
                            photo=foto_cargada,
                            caption=ficha,
                            reply_markup=teclado,
                        )
                        return
                    except Exception as e:
                        log.warning("No se pudo enviar la foto con caption: %s", e)
            else:
                if archivo_abierto:
                    archivo_abierto.seek(0)
                try:
                    await mensaje_destino.reply_photo(photo=foto_cargada)
                except Exception as e:
                    log.warning("No se pudo enviar la foto de la moto: %s", e)

        try:
            await mensaje_destino.reply_text(
                ficha, parse_mode=ParseMode.MARKDOWN, reply_markup=teclado
            )
        except Exception:
            await mensaje_destino.reply_text(ficha, reply_markup=teclado)
    finally:
        if archivo_abierto:
            archivo_abierto.close()


def mensaje_error_usuario(exc: Exception) -> str:
    if isinstance(exc, ServicioSaturadoError):
        return DATOS["bot"].get("servicio_saturado", DATOS["bot"]["error_api"])
    return DATOS["bot"]["error_api"]


# ---------- Handlers de Telegram ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nombre = update.effective_user.first_name or "piloto"
    total = len(DATOS.get("motos_catalogo", []))
    texto = DATOS["bot"]["bienvenida"].format(nombre=nombre, total=total)
    await update.message.reply_text(
        texto, parse_mode=ParseMode.MARKDOWN, reply_markup=teclado_catalogo(0)
    )


async def cmd_catalogo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    total = len(DATOS.get("motos_catalogo", []))
    await update.message.reply_text(
        f"📋 *Catálogo rápido ({total} motos pre-cargadas):*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=teclado_catalogo(0),
    )


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🏍️ *¿Cómo usar el bot?*\n\n"
        "• Escribe el nombre o modelo de cualquier moto.\n"
        "• Envía una foto de una moto.\n"
        "• Usa /catalogo para explorar los modelos pre-cargados.\n"
        "• Usa /start para reiniciar el menú.\n"
        "• Usa /salir para despedirte del chat.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_salir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        DATOS["bot"]["despedida"], parse_mode=ParseMode.MARKDOWN
    )


async def al_pulsar_boton(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "noop":
        return

    if query.data.startswith("pag:"):
        pagina = int(query.data.split(":", 1)[1])
        await query.edit_message_reply_markup(reply_markup=teclado_catalogo(pagina))
        return

    if not query.data.startswith("moto:"):
        return

    moto_id = query.data.split(":", 1)[1]
    moto = next(
        (m for m in DATOS.get("motos_catalogo", []) if m["id"] == moto_id), None
    )
    if not moto:
        return

    await enviar_ficha_con_foto(
        query.message, moto["ficha"], moto.get("imagen")
    )


async def al_escribir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    consulta = update.message.text.strip()
    ficha_local = buscar_en_catalogo(consulta)

    if ficha_local:
        await update.message.reply_text(ficha_local, parse_mode=ParseMode.MARKDOWN)
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    msg_espera = await update.message.reply_text(DATOS["bot"]["procesando_texto"])

    try:
        resultado = consultar_moto_texto(consulta)
        await enviar_respuesta_formateada(msg_espera, resultado)
    except Exception as e:
        log.error("Error procesando texto: %s", e)
        await msg_espera.edit_text(mensaje_error_usuario(e))


async def al_enviar_foto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    msg_espera = await update.message.reply_text(DATOS["bot"]["procesando_foto"])

    try:
        foto = update.message.photo[-1]
        archivo_tg = await context.bot.get_file(foto.file_id)
        buffer = io.BytesIO()
        await archivo_tg.download_to_memory(buffer)

        comentario = update.message.caption
        resultado = consultar_moto_imagen(buffer.getvalue(), comentario)

        if "NO_ES_MOTO" in resultado:
            await msg_espera.edit_text(DATOS["bot"]["no_es_moto"])
            return

        await enviar_respuesta_formateada(msg_espera, resultado)
    except Exception as e:
        log.error("Error procesando imagen: %s", e)
        await msg_espera.edit_text(mensaje_error_usuario(e))


async def al_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Error en Telegram", exc_info=context.error)


def main() -> None:
    if not os.environ.get("TELEGRAM_TOKEN"):
        raise SystemExit("Falta la variable TELEGRAM_TOKEN (créala con @BotFather).")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "Falta la variable GEMINI_API_KEY (obtén una gratis en Google AI Studio)."
        )

    app = (
        Application.builder()
        .token(os.environ["TELEGRAM_TOKEN"])
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("catalogo", cmd_catalogo))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    app.add_handler(CommandHandler("salir", cmd_salir))
    app.add_handler(CallbackQueryHandler(al_pulsar_boton))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, al_escribir))
    app.add_handler(MessageHandler(filters.PHOTO, al_enviar_foto))
    app.add_error_handler(al_error)

    log.info("%s en marcha. Ctrl+C para detener.", DATOS["bot"]["nombre"])
    app.run_polling(bootstrap_retries=3)


if __name__ == "__main__":
    main()