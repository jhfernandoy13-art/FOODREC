"""Analista con IA: contexto estructurado real → LLM → validación de honestidad.

Regla de oro: el Transformer recomienda; el LLM solo analiza. La lista que ve la
persona se dibuja desde el modelo (nunca desde el texto del LLM) y además cada
respuesta pasa por `validar_respuesta`, que detecta códigos inventados,
listas reordenadas y un nivel de auditoría distinto al calculado.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .llm import ErrorLLM, extraer_json, llamar_llm
from .recomendador import CATEGORIAS_FILTRO, Recomendador

NOMBRE_LINEA_BASE = {
    "Item-Item": "Ítem–ítem con coseno (últimas 20 recetas, decaimiento 0,9)",
    "Popularidad": "Popularidad en entrenamiento",
    "Reglas + popularidad": "Reglas de asociación por lift + respaldo de popularidad",
}
MAX_HISTORIAL_EN_CONTEXTO = 20
PATRON_CODIGO = re.compile(r"\b([HSBP])(\d{1,3})\b")
PATRON_NUMERADA = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*)$", re.MULTILINE)
PATRON_AUDITORIA = re.compile(r"AUDITOR[IÍ]A:[^\n=]*=\s*\**\s*(alto|medio|bajo)", re.IGNORECASE)
PATRON_AUDITORIA_BASE = re.compile(r"AUDITOR[IÍ]A_BASE:[^\n=]*=\s*\**\s*(alto|medio|bajo)", re.IGNORECASE)


def cargar_prompts(ruta: Path) -> dict:
    texto = Path(ruta).read_text(encoding="utf-8")
    secciones = dict(re.findall(r"<!-- INICIO:(\w+) -->\n(.*?)<!-- FIN:\1 -->", texto, re.DOTALL))
    version = re.search(r"\*\*Versión:\*\*\s*([^\n]+)", texto)
    secciones["version"] = version.group(1).strip() if version else "sin versión"
    assert {"analista", "chat"} <= set(secciones), "Faltan secciones en analista_prompt.md"
    return secciones


# ---------------------------------------------------------------- contexto
def _receta(rec: Recomendador, item: int, extra: dict | None = None) -> dict:
    fila = rec.catalogo.loc[int(item)]
    datos = {
        "receta": fila["nombre"],
        "categoria": fila["categoria"],
        "minutos": int(fila["minutos"]),
        "ingredientes_principales": list(fila["ingredientes"])[:6],
        "usuarios_que_la_resenaron_en_train": int(fila["usuarios_train"]),
        "es_popular_top10": bool(fila["es_popular"]),
    }
    if extra:
        datos.update(extra)
    return datos


def construir_contexto(
    rec: Recomendador,
    historial: list[int],
    top_n: int = 10,
    filtros: dict | None = None,
    linea_base: str = "Item-Item",
    metricas: pd.DataFrame | None = None,
    nota: str | None = None,
) -> tuple[dict, dict]:
    """Devuelve (contexto para el LLM, tablas/mapeos para la app)."""
    historial = [int(i) for i in historial]
    filtros = {k: v for k, v in (filtros or {}).items() if v}
    tablas: dict = {"codigos": {}}

    contexto: dict = {
        "rubro": "Recetas de cocina (Food.com). Interacciones = reseñas con fecha.",
        "modo": "personalizado" if historial else "arranque_en_frio",
        "filtros_aplicados": filtros or "ninguno",
    }
    if nota:
        contexto["nota_de_la_aplicacion"] = nota

    # Historial (H1 = más antigua). Se envían solo las últimas 20 para limitar tokens.
    inicio_visible = max(0, len(historial) - MAX_HISTORIAL_EN_CONTEXTO)
    historial_ctx = []
    for posicion, item in enumerate(historial, start=1):
        codigo = f"H{posicion}"
        tablas["codigos"][codigo] = item
        if posicion - 1 >= inicio_visible:
            historial_ctx.append({"codigo": codigo, **_receta(rec, item)})
    contexto["historial"] = {
        "largo_total": len(historial),
        "omitidas_por_espacio": inicio_visible,
        "recetas_en_orden": historial_ctx,
    }

    if historial:
        sasrec, atencion, recientes = rec.recomendar_sasrec(historial, top_n, filtros)
        tablas["sasrec"] = sasrec
        tablas["atencion"] = pd.DataFrame({
            "codigo": [f"H{len(historial) - len(recientes) + i + 1}" for i in range(len(recientes))],
            "item_idx": recientes,
            "receta": [rec.nombre(i) for i in recientes],
            "atencion": atencion,
        })
        peso_por_codigo = dict(zip(tablas["atencion"]["codigo"], tablas["atencion"]["atencion"]))
        for h in historial_ctx:
            if h["codigo"] in peso_por_codigo:
                h["atencion_transformer"] = round(float(peso_por_codigo[h["codigo"]]), 4)
        top_atencion = tablas["atencion"].nlargest(3, "atencion")
        contexto["atencion_transformer"] = {
            "explicacion": "Peso de cada receta del historial al calcular la última posición "
                           "(promedio de cabezas del último bloque). Suman 1. No es causalidad.",
            "recetas_visibles": len(recientes),
            "mas_atendidas": [
                {"codigo": c, "peso": round(float(p), 4)}
                for c, p in zip(top_atencion["codigo"], top_atencion["atencion"])
            ],
        }

        lista = []
        codigos_hist = {v: k for k, v in tablas["codigos"].items()}
        visibles = [tablas["codigos"][h["codigo"]] for h in historial_ctx]
        for fila in sasrec.itertuples(index=False):
            codigo = f"S{fila.rank}"
            tablas["codigos"][codigo] = int(fila.item_idx)
            jaccards = [(rec.jaccard(fila.item_idx, h), h) for h in visibles]
            mejor_j, mejor_h = max(jaccards) if jaccards else (0.0, None)
            lista.append({
                "codigo": codigo,
                "posicion": int(fila.rank),
                "puntaje_transformer": round(float(fila.puntaje), 3),
                **_receta(rec, fila.item_idx),
                "receta_del_historial_mas_parecida": {
                    "codigo": codigos_hist.get(mejor_h),
                    "jaccard_ingredientes": round(mejor_j, 3),
                } if mejor_h is not None else None,
            })
        contexto["recomendacion_transformer"] = {
            "modelo": f"SASRec ({rec.config.get('nombre', '')}, {rec.config.get('bloques')} bloque(s), "
                      f"dim {rec.config.get('dimension')})",
            "nota": "Lista y orden definidos por el modelo. Puntaje = producto punto (mayor = mejor).",
            "lista": lista,
        }

        base = rec.recomendar(linea_base, historial, top_n, filtros)
        tablas["base"] = base
        lista_base = []
        for fila in base.itertuples(index=False):
            codigo = f"B{fila.rank}"
            tablas["codigos"][codigo] = int(fila.item_idx)
            extra = {"codigo": codigo, "posicion": int(fila.rank)}
            if linea_base == "Item-Item":
                extra["mas_parecida_a"] = codigos_hist.get(int(fila.mas_parecida_a))
                extra["coseno"] = round(float(fila.coseno_max), 3)
            if linea_base == "Reglas + popularidad":
                extra["origen"] = fila.regla
                if fila.regla == "reglas":
                    extra["regla"] = f"{codigos_hist.get(int(fila.antecedente))} → {codigo}"
                    extra["lift"] = round(float(fila.puntaje), 2)
            lista_base.append({**extra, **_receta(rec, fila.item_idx)})
        contexto["linea_base"] = {"modelo": NOMBRE_LINEA_BASE[linea_base], "lista": lista_base}

        s_items = sasrec["item_idx"].tolist()
        b_items = base["item_idx"].tolist()
        contexto["coincidencias_transformer_vs_base"] = [
            f"S{s_items.index(i) + 1} = B{b_items.index(i) + 1}" for i in s_items if i in b_items
        ]
        contexto["auditoria_sesgo"] = {
            "definiciones": {
                "cuota_popular": "fracción de la lista en el 10 % de recetas más reseñadas en train",
                "novedad_bits": "promedio de -log2(popularidad suavizada); más alto = menos obvia",
                "diversidad_ingredientes": "1 - Jaccard medio entre pares de la lista (0 a 1)",
                "afinidad_historial": "Jaccard medio con la receta más parecida del historial",
                "nivel": "alto si cuota ≥ 0,60; bajo si ≤ 0,30; medio en otro caso",
            },
            "transformer": rec.auditar(s_items, historial),
            "linea_base": rec.auditar(b_items, historial),
        }
        contexto["patrones_calculados"] = rec.patrones_historial(historial, s_items)
    else:
        pop = rec.recomendar_popularidad([], top_n, filtros)
        tablas["popularidad_frio"] = pop
        lista = []
        for fila in pop.itertuples(index=False):
            codigo = f"P{fila.rank}"
            tablas["codigos"][codigo] = int(fila.item_idx)
            lista.append({"codigo": codigo, "posicion": int(fila.rank), **_receta(rec, fila.item_idx)})
        contexto["lista_mostrada"] = {
            "modelo": "Popularidad (respaldo explícito: sin historial no se puede usar el Transformer)",
            "lista": lista,
        }
        contexto["auditoria_sesgo"] = {"lista_mostrada": rec.auditar(pop["item_idx"].tolist())}

    if metricas is not None:
        k10 = metricas[metricas["K"] == 10].set_index("Modelo")
        contexto["metricas_offline_prueba_K10"] = {
            modelo: {
                "HR@10": round(float(f["HR"]), 4),
                "NDCG@10": round(float(f["NDCG"]), 4),
                "Cobertura@10": round(float(f["Cobertura"]), 4),
            }
            for modelo, f in k10.iterrows()
        }
    contexto["limites_conocidos"] = [
        "Las interacciones son reseñas: no prueban que la persona cocinó la receta.",
        "Las fechas son diarias; el orden dentro de un mismo día es un desempate reproducible.",
        "En la prueba offline, SASRec y popularidad tienen precisión muy parecida (ver métricas).",
        "Asociación no es causalidad; la atención no explica por qué alguien elige una receta.",
    ]
    return contexto, tablas


# ---------------------------------------------------------------- validación
def validar_respuesta(texto: str, contexto: dict, tablas: dict) -> dict:
    """Pruebas automáticas de honestidad sobre una respuesta del LLM."""
    validos = set(tablas["codigos"])
    citados = {f"{l}{n}" for l, n in PATRON_CODIGO.findall(texto)}
    invalidos = sorted(citados - validos)

    # Una lista numerada cuyo número no coincide con el código S/P sugiere reordenamiento
    reordenados = []
    for numero, linea in PATRON_NUMERADA.findall(texto):
        codigos = re.findall(r"\b([SP])(\d{1,3})\b", linea)
        if codigos and int(codigos[0][1]) != int(numero):
            reordenados.append(f"{numero}. → {codigos[0][0]}{codigos[0][1]}")

    auditoria = contexto.get("auditoria_sesgo", {})
    nivel_real = (auditoria.get("transformer") or auditoria.get("lista_mostrada") or {}).get(
        "nivel_sesgo_popularidad")
    declarado = PATRON_AUDITORIA.search(texto)
    nivel_declarado = declarado.group(1).lower() if declarado else None
    base_real = (auditoria.get("linea_base") or {}).get("nivel_sesgo_popularidad")
    base = PATRON_AUDITORIA_BASE.search(texto)
    base_declarado = base.group(1).lower() if base else None

    return {
        "codigos_citados": sorted(citados),
        "codigos_inventados": invalidos,
        "posible_reordenamiento": reordenados,
        "nivel_auditoria_real": nivel_real,
        "nivel_auditoria_declarado": nivel_declarado,
        "auditoria_coincide": None if nivel_declarado is None else nivel_declarado == nivel_real,
        "nivel_base_real": base_real,
        "nivel_base_declarado": base_declarado,
        "auditoria_base_coincide": None if base_declarado is None else base_declarado == base_real,
        "auditoria_base_inventada": base_real is None and base_declarado is not None,
        "ok": not invalidos and not reordenados and not (base_real is None and base_declarado is not None),
    }


# ---------------------------------------------------------------- llamadas
def _mensaje_contexto(contexto: dict) -> str:
    return "CONTEXTO (datos reales calculados por la aplicación, en JSON):\n" + json.dumps(
        contexto, ensure_ascii=False, separators=(",", ":"))


def analizar(contexto: dict, tablas: dict, prompts: dict, cfg: dict, instruccion: str | None = None):
    """Análisis inicial automático. cfg = {proveedor, modelo, api_key}."""
    pedido = instruccion or "Genera el análisis inicial siguiendo exactamente tus instrucciones."
    mensajes = [{"role": "user", "content": _mensaje_contexto(contexto) + "\n\n" + pedido}]
    respuesta = llamar_llm(cfg["proveedor"], cfg.get("modelo"), cfg.get("api_key"),
                           prompts["analista"], mensajes, max_tokens=1400)
    return respuesta.texto, respuesta, validar_respuesta(respuesta.texto, contexto, tablas)


def chatear(contexto: dict, tablas: dict, conversacion: list[dict], pregunta: str,
            prompts: dict, cfg: dict):
    """Una vuelta de chat. Devuelve (dict con respuesta y reconsulta, RespuestaLLM, validación)."""
    mensajes = [
        {"role": "user", "content": _mensaje_contexto(contexto)},
        {"role": "assistant", "content": '{"respuesta": "Contexto recibido.", "reconsulta": null}'},
    ]
    for turno in conversacion[-6:]:  # memoria corta para limitar tokens
        mensajes.append({"role": turno["role"], "content": turno["content"]})
    mensajes.append({"role": "user", "content": pregunta})

    respuesta = llamar_llm(cfg["proveedor"], cfg.get("modelo"), cfg.get("api_key"),
                           prompts["chat"], mensajes, json_mode=True, max_tokens=900)
    try:
        datos = extraer_json(respuesta.texto)
    except (json.JSONDecodeError, ValueError):
        datos = {"respuesta": respuesta.texto, "reconsulta": None}
    datos.setdefault("respuesta", "")
    datos["reconsulta"] = normalizar_reconsulta(datos.get("reconsulta"), tablas)
    return datos, respuesta, validar_respuesta(str(datos["respuesta"]), contexto, tablas)


def normalizar_reconsulta(reconsulta, tablas) -> dict | None:
    """Acepta solo reconsultas válidas; cualquier otra cosa se descarta."""
    if not isinstance(reconsulta, dict):
        return None
    tipo = reconsulta.get("tipo")
    if tipo == "filtrar_categoria" and reconsulta.get("categoria") in CATEGORIAS_FILTRO:
        return {"tipo": tipo, "categoria": reconsulta["categoria"]}
    if tipo == "tiempo_maximo":
        try:
            minutos = int(reconsulta.get("minutos"))
        except (TypeError, ValueError):
            return None
        return {"tipo": tipo, "minutos": max(5, min(minutos, 24 * 60))}
    if tipo == "quitar_del_historial" and str(reconsulta.get("codigo", "")).startswith("H") \
            and reconsulta["codigo"] in tablas["codigos"]:
        return {"tipo": tipo, "codigo": reconsulta["codigo"]}
    if tipo == "agregar_al_historial" and str(reconsulta.get("busqueda", "")).strip():
        return {"tipo": tipo, "busqueda": str(reconsulta["busqueda"]).strip()[:60]}
    if tipo == "sin_filtros":
        return {"tipo": tipo}
    return None


def aplicar_reconsulta(rec: Recomendador, historial: list[int], filtros: dict,
                       reconsulta: dict, tablas: dict):
    """Ejecuta la reconsulta en Python. Devuelve (historial, filtros, descripción) o error."""
    historial = list(historial)
    filtros = dict(filtros or {})
    tipo = reconsulta["tipo"]
    if tipo == "filtrar_categoria":
        filtros["categoria"] = reconsulta["categoria"]
        descripcion = f"solo recetas de la categoría {reconsulta['categoria']}"
    elif tipo == "tiempo_maximo":
        filtros["max_minutos"] = reconsulta["minutos"]
        descripcion = f"solo recetas de hasta {reconsulta['minutos']} minutos"
    elif tipo == "quitar_del_historial":
        item = tablas["codigos"][reconsulta["codigo"]]
        historial.remove(item)
        descripcion = f"historial sin {reconsulta['codigo']} ({rec.nombre(item)})"
    elif tipo == "agregar_al_historial":
        encontrados = rec.buscar(reconsulta["busqueda"], n=1)
        if encontrados.empty:
            raise ValueError(f"No encontré '{reconsulta['busqueda']}' en el catálogo.")
        item = int(encontrados["item_idx"].iloc[0])
        if item in historial:
            raise ValueError(f"'{rec.nombre(item)}' ya está en el historial.")
        historial.append(item)
        descripcion = f"historial con '{rec.nombre(item)}' agregada al final"
    else:
        filtros = {}
        descripcion = "sin filtros"
    return historial, filtros, descripcion


def costo_resumen(registros: list[dict]) -> pd.DataFrame:
    """Tabla de tokens, latencia y costo referencial por tipo de llamada."""
    if not registros:
        return pd.DataFrame()
    df = pd.DataFrame(registros)
    return df.groupby("tipo").agg(
        llamadas=("latencia_s", "size"),
        tokens_entrada_media=("tokens_entrada", "mean"),
        tokens_salida_media=("tokens_salida", "mean"),
        latencia_media_s=("latencia_s", "mean"),
        latencia_p95_s=("latencia_s", lambda x: float(np.quantile(x, 0.95))),
        costo_usd_total=("costo_usd_referencial", "sum"),
    ).round(4)


__all__ = [
    "ErrorLLM", "aplicar_reconsulta", "analizar", "cargar_prompts", "chatear",
    "construir_contexto", "costo_resumen", "validar_respuesta",
]
