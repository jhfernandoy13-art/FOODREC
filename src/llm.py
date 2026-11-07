"""Cliente mínimo para LLMs vía HTTP (Gemini, Groq o Claude).

- La clave nunca se escribe en el código: llega desde st.secrets (app) o desde
  los Secretos de Colab (notebook).
- Cada llamada devuelve tokens de entrada/salida y latencia, para medir costo.
- El proveedor "simulado" no llama a ninguna API: sirve solo para probar la app
  sin clave. Nunca debe usarse para reportar resultados.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass

import requests

MODELOS_POR_DEFECTO = {
    # Alias que Google mantiene apuntando al Flash más reciente. Si no responde,
    # el cliente busca solo un modelo Flash disponible (ver _modelos_gemini).
    "gemini": "gemini-flash-lite-latest",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-haiku-4-5",
    "simulado": "simulado",
}

# USD por millón de tokens (entrada, salida). Referenciales: actualizar con la
# página de precios del proveedor. En el free tier de Gemini/Groq el costo es 0.
# Un modelo que no está aquí se reporta con costo 0 y precio_registrado = False.
PRECIOS_USD_POR_MILLON = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "claude-haiku-4-5": (1.00, 5.00),
    "simulado": (0.0, 0.0),
}


@dataclass
class RespuestaLLM:
    texto: str
    tokens_entrada: int
    tokens_salida: int
    latencia_s: float
    proveedor: str
    modelo: str

    def costo_usd(self) -> float:
        entrada, salida = PRECIOS_USD_POR_MILLON.get(self.modelo, (0.0, 0.0))
        return (self.tokens_entrada * entrada + self.tokens_salida * salida) / 1e6

    def como_dict(self) -> dict:
        datos = asdict(self)
        datos["costo_usd_referencial"] = round(self.costo_usd(), 6)
        return datos


class ErrorLLM(RuntimeError):
    pass


def _post(url, headers, cuerpo, timeout, reintentos=3):
    for intento in range(reintentos + 1):
        try:
            respuesta = requests.post(url, headers=headers, json=cuerpo, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as error:
            # Servidor lento o saturado: un reintento y luego se informa como ErrorLLM
            if intento < 1:
                time.sleep(5)
                continue
            raise ErrorLLM(f"timeout: el proveedor no respondió a tiempo ({error.__class__.__name__})")
        if respuesta.status_code in (429, 500, 503) and intento < reintentos:
            time.sleep(10 * (intento + 1))  # límite por minuto: esperar y reintentar
            continue
        if respuesta.status_code != 200:
            raise ErrorLLM(f"HTTP {respuesta.status_code}: {respuesta.text[:400]}")
        return respuesta.json()
    raise ErrorLLM("Sin respuesta del proveedor.")


def llamar_llm(
    proveedor: str,
    modelo: str | None,
    api_key: str | None,
    sistema: str,
    mensajes: list[dict],
    json_mode: bool = False,
    max_tokens: int = 1200,
    temperatura: float = 0.2,
    timeout: int = 60,
) -> RespuestaLLM:
    """mensajes: [{"role": "user" | "assistant", "content": str}, ...]"""
    proveedor = proveedor.lower()
    modelo = modelo or MODELOS_POR_DEFECTO[proveedor]
    if proveedor != "simulado" and not api_key:
        raise ErrorLLM("Falta la clave de la API (st.secrets o Secretos de Colab).")

    inicio = time.perf_counter()

    if proveedor == "gemini":
        texto, t_in, t_out, modelo = _llamar_gemini(
            modelo, api_key, sistema, mensajes, json_mode, max_tokens, timeout)

    elif proveedor == "groq":
        cuerpo = {
            "model": modelo,
            "messages": [{"role": "system", "content": sistema}] + mensajes,
            "temperature": temperatura,
            "max_tokens": max_tokens,
        }
        if json_mode:
            cuerpo["response_format"] = {"type": "json_object"}
        datos = _post("https://api.groq.com/openai/v1/chat/completions",
                      {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      cuerpo, timeout)
        texto = datos["choices"][0]["message"]["content"]
        t_in = datos.get("usage", {}).get("prompt_tokens", 0)
        t_out = datos.get("usage", {}).get("completion_tokens", 0)

    elif proveedor == "anthropic":
        cuerpo = {
            "model": modelo,
            "system": sistema,
            "messages": mensajes,
            "max_tokens": max_tokens,
            "temperature": temperatura,
        }
        datos = _post("https://api.anthropic.com/v1/messages",
                      {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"},
                      cuerpo, timeout)
        texto = "".join(b.get("text", "") for b in datos.get("content", []))
        t_in = datos.get("usage", {}).get("input_tokens", 0)
        t_out = datos.get("usage", {}).get("output_tokens", 0)

    elif proveedor == "simulado":
        texto = _respuesta_simulada(mensajes, json_mode)
        t_in = sum(len(m["content"]) for m in mensajes) // 4 + len(sistema) // 4
        t_out = len(texto) // 4

    else:
        raise ErrorLLM(f"Proveedor desconocido: {proveedor}")

    return RespuestaLLM(
        texto=texto.strip(),
        tokens_entrada=int(t_in),
        tokens_salida=int(t_out),
        latencia_s=round(time.perf_counter() - inicio, 3),
        proveedor=proveedor,
        modelo=modelo,
    )


URL_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
_modelo_gemini_resuelto: dict = {}


def _modelos_gemini(api_key: str) -> list[str]:
    """Modelos Flash disponibles para esta clave, del más nuevo al más antiguo."""
    try:
        respuesta = requests.get(URL_GEMINI, headers={"x-goog-api-key": api_key},
                                 params={"pageSize": 200}, timeout=30)
        modelos = respuesta.json().get("models", []) if respuesta.status_code == 200 else []
    except requests.RequestException:
        modelos = []
    excluir = ("lite", "image", "tts", "live", "audio", "embedding", "exp", "thinking", "robotics")
    candidatos = []
    for m in modelos:
        nombre = m.get("name", "").replace("models/", "")
        if ("flash" in nombre and not any(x in nombre for x in excluir)
                and "generateContent" in m.get("supportedGenerationMethods", [])):
            version = re.findall(r"gemini-(\d+(?:\.\d+)?)", nombre)
            clave = (float(version[0]) if version else 0.0, "preview" not in nombre)
            candidatos.append((clave, nombre))
    listados = [n for _, n in sorted(candidatos, reverse=True)]
    # Respaldo si la lista no está disponible (Flash-Lite al final: más rápido y liviano)
    return listados + ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash",
                       "gemini-flash-lite-latest", "gemini-2.5-flash-lite"]


def _cuerpo_gemini(modelo, sistema, mensajes, json_mode, max_tokens):
    config = {"maxOutputTokens": max_tokens}
    if modelo.startswith("gemini-2.5"):
        # Modelos 2.5: temperatura baja y sin "pensamiento" (menos latencia y tokens)
        config["temperature"] = 0.2
        config["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        # Gemini 3.x en adelante: temperature/top_p/top_k están obsoletos, y el
        # razonamiento interno consume tokens de salida → más margen.
        config["maxOutputTokens"] = max(max_tokens, 8192)
    if json_mode:
        config["responseMimeType"] = "application/json"
    return {
        "system_instruction": {"parts": [{"text": sistema}]},
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in mensajes
        ],
        "generationConfig": config,
    }


def _llamar_gemini(modelo, api_key, sistema, mensajes, json_mode, max_tokens, timeout):
    """Llama a Gemini; si el modelo no existe o fue retirado, prueba otro disponible."""
    cabeceras = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    pedido = modelo
    intentos = [_modelo_gemini_resuelto.get(pedido, pedido)]
    ultimo_error = None
    for _ in range(6):
        actual = intentos.pop(0)
        try:
            datos = _post(f"{URL_GEMINI}/{actual}:generateContent", cabeceras,
                          _cuerpo_gemini(actual, sistema, mensajes, json_mode, max_tokens), timeout)
        except ErrorLLM as error:
            texto_error = str(error)
            retirado = any(x in texto_error for x in (
                "HTTP 404", "no longer available", "not found", "ACCESS_TOKEN_TYPE_UNSUPPORTED",
                "is not supported", "INVALID_ARGUMENT", "timeout", "HTTP 503", "HTTP 500"))
            if not retirado:
                raise
            ultimo_error = error
            if not intentos:
                intentos = [m for m in _modelos_gemini(api_key) if m != actual]
                if "timeout" in texto_error:
                    # Si el Flash está saturado, probar primero la variante liviana
                    intentos = ["gemini-flash-lite-latest"] + intentos
            intentos = [m for m in intentos if m != actual]
            if not intentos:
                break
            continue
        _modelo_gemini_resuelto[pedido] = actual
        try:
            partes = datos["candidates"][0]["content"]["parts"]
            texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
        except (KeyError, IndexError) as error:
            raise ErrorLLM(f"Respuesta inesperada de Gemini: {str(datos)[:400]}") from error
        uso = datos.get("usageMetadata", {})
        t_in = uso.get("promptTokenCount", 0)
        t_out = uso.get("candidatesTokenCount", 0) + uso.get("thoughtsTokenCount", 0)
        return texto, t_in, t_out, datos.get("modelVersion", actual)
    raise ErrorLLM(f"Ningún modelo Gemini respondió. Último error: {ultimo_error}")


def precio_registrado(modelo: str) -> bool:
    return modelo in PRECIOS_USD_POR_MILLON


def extraer_json(texto: str) -> dict:
    """Extrae el primer objeto JSON del texto (tolerante a ```json ... ```)."""
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio, fin = texto.find("{"), texto.rfind("}")
        if inicio >= 0 and fin > inicio:
            return json.loads(texto[inicio:fin + 1])
        raise


def _respuesta_simulada(mensajes, json_mode):
    """Respuesta fija y honesta para probar la interfaz sin API (no es un LLM)."""
    contexto = mensajes[0]["content"]
    niveles = re.findall(r'"nivel_sesgo_popularidad":\s*"(\w+)"', contexto)
    analisis = (
        "**[Modo simulado: respuesta fija, no generada por un LLM]**\n\n"
        "La lista del Transformer comienza con [S1] y [S2].\n\n"
        f"AUDITORIA: nivel de sesgo de popularidad = {niveles[0] if niveles else '?'}\n"
    )
    if len(niveles) > 1:
        analisis += f"AUDITORIA_BASE: nivel de sesgo de popularidad = {niveles[1]}\n"
    analisis += "\nRecuerda: asociación no es causalidad."
    if json_mode:
        return json.dumps({"respuesta": analisis, "reconsulta": None}, ensure_ascii=False)
    return analisis
