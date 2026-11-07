"""FoodRec · Dashboard Streamlit (Entrega 2).

El Transformer (SASRec) recomienda: la lista y su orden salen del modelo.
La IA generativa analiza: explica, contrasta, advierte límites y responde preguntas.
Nada se entrena aquí: se cargan los artefactos de la Entrega 1 en CPU con caché.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.analista import (
    ErrorLLM, analizar, aplicar_reconsulta, cargar_prompts, chatear,
    construir_contexto, costo_resumen,
)
from src.llm import MODELOS_POR_DEFECTO
from src.recomendador import CATEGORIAS_FILTRO, Recomendador

RAIZ = Path(__file__).parent
MAX_LLAMADAS_SESION = 30  # protege la cuota gratuita durante la exposición
MODELOS_BASE = ["Popularidad", "Item-Item", "Reglas + popularidad"]

st.set_page_config(page_title="FoodRec · recetas que entienden el orden",
                   page_icon="🍲", layout="wide")


# ---------------------------------------------------------------- carga con caché
@st.cache_resource(show_spinner="Cargando SASRec en CPU…")
def cargar_recomendador() -> Recomendador:
    return Recomendador(RAIZ)


@st.cache_resource
def cargar_prompts_cache() -> dict:
    return cargar_prompts(RAIZ / "src" / "analista_prompt.md")


@st.cache_data
def leer_csv(nombre: str) -> pd.DataFrame | None:
    ruta = RAIZ / "data" / nombre
    return pd.read_csv(ruta) if ruta.is_file() else None


@st.cache_data(show_spinner=False, max_entries=64)
def contexto_cache(_rec, historial: tuple, top_n: int, filtros: tuple, linea_base: str):
    return construir_contexto(_rec, list(historial), top_n, dict(filtros), linea_base,
                              leer_csv("metricas.csv"))


def configuracion_llm() -> dict:
    try:
        secretos = dict(st.secrets)
    except Exception:  # sin secrets.toml
        secretos = {}
    proveedor = str(secretos.get("LLM_PROVEEDOR", "gemini")).lower()
    clave = secretos.get("LLM_API_KEY") or secretos.get(f"{proveedor.upper()}_API_KEY")
    modelo = secretos.get("LLM_MODELO") or MODELOS_POR_DEFECTO.get(proveedor)
    if not clave:
        return {"proveedor": "simulado", "modelo": "simulado", "api_key": None, "sin_clave": True}
    return {"proveedor": proveedor, "modelo": modelo, "api_key": clave, "sin_clave": False}


rec = cargar_recomendador()
prompts = cargar_prompts_cache()
cfg = configuracion_llm()
metricas = leer_csv("metricas.csv")

estado = st.session_state
estado.setdefault("historial", [])
estado.setdefault("chat", [])
estado.setdefault("registros_llm", [])
estado.setdefault("analisis", {})
estado.setdefault("llamadas", 0)


def registrar(tipo: str, uso) -> None:
    datos = uso.como_dict()
    datos.pop("texto", None)
    datos["tipo"] = tipo
    estado.registros_llm.append(datos)
    estado.llamadas += 1


def puede_llamar() -> bool:
    if estado.llamadas >= MAX_LLAMADAS_SESION:
        st.warning(f"Se alcanzó el límite de {MAX_LLAMADAS_SESION} llamadas al LLM en esta sesión "
                   "(protección de cuota). Recarga la página para reiniciar.")
        return False
    return True


def reiniciar_chat():
    estado.chat = []


# ---------------------------------------------------------------- barra lateral: historial
with st.sidebar:
    st.header("🧺 Tu historial (en orden)")
    st.caption("El Transformer lee el ORDEN: la última receta es la más reciente.")

    personas = rec.personas()
    opciones = ["—"] + list(personas) + ["Arranque en frío (sin historial)"]
    persona = st.selectbox("Historial de ejemplo", opciones)
    if st.button("Cargar ejemplo", width="stretch", disabled=persona == "—"):
        estado.historial = list(personas.get(persona, []))
        reiniciar_chat()
        st.rerun()

    busqueda = st.text_input("Buscar receta (nombres en inglés)",
                             placeholder="brownies, chili, lasagna, salad…")
    if busqueda:
        encontrados = rec.buscar(busqueda, n=25)
        encontrados = encontrados[~encontrados["item_idx"].isin(estado.historial)]
        if encontrados.empty:
            st.caption("Sin resultados en el catálogo.")
        else:
            etiquetas = {
                int(f.item_idx): f"{f.nombre} · {f.categoria} · {f.minutos} min"
                for f in encontrados.itertuples()
            }
            elegido = st.selectbox("Resultados", list(etiquetas), format_func=etiquetas.get)
            if st.button("➕ Agregar al final", width="stretch"):
                estado.historial.append(int(elegido))
                reiniciar_chat()
                st.rerun()

    if not estado.historial:
        st.info("Historial vacío → arranque en frío (se muestra popularidad).")
    for i, item in enumerate(estado.historial):
        c1, c2, c3, c4 = st.columns([7, 1, 1, 1])
        c1.markdown(f"**H{i + 1}.** {rec.nombre(item)}")
        if c2.button("↑", key=f"sube{i}", disabled=i == 0):
            h = estado.historial
            h[i - 1], h[i] = h[i], h[i - 1]
            reiniciar_chat()
            st.rerun()
        if c3.button("↓", key=f"baja{i}", disabled=i == len(estado.historial) - 1):
            h = estado.historial
            h[i + 1], h[i] = h[i], h[i + 1]
            reiniciar_chat()
            st.rerun()
        if c4.button("✕", key=f"quita{i}"):
            estado.historial.pop(i)
            reiniciar_chat()
            st.rerun()
    if estado.historial and st.button("🗑️ Vaciar historial", width="stretch"):
        estado.historial = []
        reiniciar_chat()
        st.rerun()

    st.divider()
    st.header("⚙️ Opciones")
    top_n = st.slider("Tamaño de la lista (top-N)", 5, 20, 10)
    categoria = st.selectbox("Categoría", ["Todas"] + CATEGORIAS_FILTRO)
    max_minutos = st.selectbox("Tiempo máximo", [None, 15, 30, 60, 120],
                               format_func=lambda m: "Sin límite" if m is None else f"{m} min")
    linea_base = st.selectbox("Línea base que ve el analista", MODELOS_BASE, index=1)
    st.caption(f"Analista: **{cfg['proveedor']}** · `{cfg['modelo']}` · prompt v{prompts['version']}")

historial = list(estado.historial)
filtros = {"categoria": None if categoria == "Todas" else categoria, "max_minutos": max_minutos}
filtros_tupla = tuple(sorted((k, v) for k, v in filtros.items() if v))
contexto, tablas = contexto_cache(rec, tuple(historial), top_n, filtros_tupla, linea_base)

# ---------------------------------------------------------------- encabezado
st.title("🍲 FoodRec — recetas que entienden el orden")
st.markdown(
    "Un **Transformer secuencial (SASRec)** predice la siguiente receta a partir del *orden* de tu "
    "historial en Food.com. Una **IA generativa no recomienda**: analiza lo que el modelo recomendó, "
    "lo contrasta con las líneas base y advierte sus límites."
)
if cfg["sin_clave"]:
    st.warning("No hay clave de LLM en `st.secrets`: el analista funciona en **modo simulado** "
               "(respuesta fija, solo para probar la interfaz).")

tab_rec, tab_tr, tab_ia, tab_cat, tab_met = st.tabs(
    ["🍽️ Recomendaciones", "🔍 Transparencia", "🤖 Analista IA", "📚 Catálogo", "📊 Métricas y costo"]
)

COLUMNAS = {"rank": "#", "receta": "Receta", "categoria": "Categoría", "minutos": "Min",
            "es_popular": "Top 10 %", "puntaje": "Puntaje"}


def mostrar_lista(tabla: pd.DataFrame, extra: dict | None = None):
    columnas = {**COLUMNAS, **(extra or {})}
    vista = tabla[[c for c in columnas if c in tabla.columns]].rename(columns=columnas)
    st.dataframe(vista, hide_index=True, width="stretch",
                 column_config={"Puntaje": st.column_config.NumberColumn(format="%.3f")})


# ---------------------------------------------------------------- 1. recomendaciones
with tab_rec:
    if not historial:
        st.subheader("Arranque en frío")
        st.info("Sin historial no se puede usar el Transformer. Se muestra **popularidad** como "
                "respaldo explícito: esta lista **no está personalizada**. Agrega recetas en la "
                "barra lateral para activar SASRec.")
        mostrar_lista(tablas["popularidad_frio"], {"usuarios_train": "Reseñas (train)"})
    else:
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.subheader("🧠 Transformer (SASRec)")
            st.caption("Lista y orden definidos por el modelo.")
            mostrar_lista(tablas["sasrec"])
        with derecha:
            st.subheader("📏 Líneas base del curso")
            listas_base = {m: rec.recomendar(m, historial, top_n, filtros) for m in MODELOS_BASE}
            for pestaña, modelo in zip(st.tabs(MODELOS_BASE), MODELOS_BASE):
                with pestaña:
                    mostrar_lista(listas_base[modelo])

        st.subheader("🧪 Auditor de sesgo (innovación del grupo)")
        st.caption("Calculado en Python con datos de entrenamiento; el analista solo lo interpreta.")
        filas = []
        for nombre, items in [("SASRec", tablas["sasrec"]["item_idx"])] + [
                (m, t["item_idx"]) for m, t in listas_base.items()]:
            a = rec.auditar(list(items), historial)
            filas.append({"Modelo": nombre, "Cuota popular": a["cuota_popular"],
                          "Novedad (bits)": a["novedad_bits"],
                          "Diversidad ingredientes": a["diversidad_ingredientes"],
                          "Afinidad con historial": a["afinidad_historial"],
                          "Sesgo de popularidad": a["nivel_sesgo_popularidad"]})
        auditoria = pd.DataFrame(filas)
        st.dataframe(auditoria, hide_index=True, width="stretch")
        nivel = auditoria.iloc[0]["Sesgo de popularidad"]
        if nivel == "alto":
            st.warning("⚠️ La lista del Transformer está dominada por recetas muy populares "
                       "(sesgo de popularidad alto).")
        elif nivel == "bajo":
            st.success("La lista del Transformer explora más allá de lo popular (sesgo bajo).")

# ---------------------------------------------------------------- 2. transparencia
with tab_tr:
    if not historial:
        st.info("Agrega recetas al historial para ver la atención del Transformer.")
    else:
        st.subheader("Atención del Transformer sobre tu historial")
        st.caption("Peso de cada receta del historial al calcular la última posición "
                   "(promedio de cabezas del último bloque, suman 1). Describe el modelo; "
                   "no es una explicación causal.")
        atencion = tablas["atencion"].copy()
        atencion["etiqueta"] = atencion["codigo"] + " · " + atencion["receta"].str[:40]
        grafico = alt.Chart(atencion).mark_bar(color="#C8553D").encode(
            x=alt.X("atencion:Q", title="Peso de atención"),
            y=alt.Y("etiqueta:N", sort=list(atencion["etiqueta"]), title=None),
            tooltip=["codigo", "receta", alt.Tooltip("atencion:Q", format=".3f")],
        ).properties(height=max(160, 26 * len(atencion)))
        st.altair_chart(grafico, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Reglas que se activaron")
            reglas = rec.recomendar_reglas(historial, top_n, filtros)
            activas = reglas[reglas["regla"] == "reglas"]
            if activas.empty:
                st.info("Ninguna regla A → B se activó con este historial; la lista de reglas "
                        "se completó con popularidad.")
            else:
                vista = pd.DataFrame({
                    "Si reseñaste (A)": [rec.nombre(a) for a in activas["antecedente"]],
                    "→ recomienda (B)": activas["receta"],
                    "Lift": activas["puntaje"].round(2),
                    "Confianza": activas["confianza"].round(3),
                })
                st.dataframe(vista, hide_index=True, width="stretch")
        with c2:
            st.subheader("Ítem–ítem: receta más parecida")
            ii = rec.recomendar_item_item(historial, top_n, filtros)
            if "mas_parecida_a" in ii:
                vista = pd.DataFrame({
                    "Recomendada": ii["receta"],
                    "Más parecida en tu historial": [rec.nombre(i) for i in ii["mas_parecida_a"]],
                    "Coseno": ii["coseno_max"].round(3),
                })
                st.dataframe(vista, hide_index=True, width="stretch")

        with st.expander("¿Por qué la máscara de atención es causal?"):
            st.markdown(
                "Durante el entrenamiento, cada posición predice la **siguiente** receta. Si la "
                "posición *t* pudiera mirar las recetas *t+1, t+2…*, vería la respuesta y el modelo "
                "aprendería a copiar en vez de predecir. La máscara triangular superior pone en "
                "−∞ esas posiciones antes del softmax(QKᵀ/√d)·V, así cada receta solo atiende a sí "
                "misma y a las anteriores — igual que en un modelo tipo GPT."
            )

# ---------------------------------------------------------------- 3. analista IA
with tab_ia:
    st.subheader("🤖 Analista con IA")
    st.caption("Recibe contexto estructurado real (historial con atributos, top-N con puntajes, "
               "atención, una línea base, popularidad y auditoría). Solo habla de los ítems que "
               "recibe, citándolos con códigos [H#] [S#] [B#] [P#]. No agrega, quita ni reordena.")

    with st.expander("Ver el contexto exacto que recibe el LLM"):
        st.json(contexto, expanded=False)

    def mostrar_validacion(validacion: dict, uso: dict | None):
        partes = []
        partes.append("✅ sin ítems inventados" if not validacion["codigos_inventados"]
                      else f"❌ códigos fuera del contexto: {validacion['codigos_inventados']}")
        partes.append("✅ sin reordenar" if not validacion["posible_reordenamiento"]
                      else f"❌ posible reordenamiento: {validacion['posible_reordenamiento']}")
        for clave_v, etiqueta in [("auditoria_coincide", "auditoría"),
                                  ("auditoria_base_coincide", "auditoría base")]:
            if validacion.get(clave_v) is not None:
                partes.append(f"✅ {etiqueta} coincide" if validacion[clave_v]
                              else f"⚠️ {etiqueta}: nivel distinto al calculado")
        if uso:
            partes.append(f"🧮 {uso['tokens_entrada']}+{uso['tokens_salida']} tokens · "
                          f"⏱️ {uso['latencia_s']:.1f} s")
        st.caption(" · ".join(partes))

    automatico = st.toggle("Análisis inicial automático", value=True)
    clave = hashlib.sha256(json.dumps([contexto, cfg["proveedor"], cfg["modelo"],
                                       prompts["version"]], sort_keys=True,
                                      ensure_ascii=False, default=str).encode()).hexdigest()
    if clave not in estado.analisis and (automatico or st.button("Generar análisis")):
        if puede_llamar():
            with st.spinner("El analista está leyendo la recomendación…"):
                try:
                    texto, uso, validacion = analizar(contexto, tablas, prompts, cfg)
                    registrar("analisis_inicial", uso)
                    estado.analisis[clave] = (texto, uso.como_dict(), validacion)
                except ErrorLLM as error:
                    st.error(f"No se pudo consultar al LLM: {error}")
    if clave in estado.analisis:
        texto, uso, validacion = estado.analisis[clave]
        with st.container(border=True):
            st.markdown(texto)
            mostrar_validacion(validacion, uso)

    st.divider()
    st.markdown("**Chat de seguimiento** — si la pregunta exige otra lista, la app vuelve a "
                "consultar al Transformer y el analista analiza el nuevo resultado.")
    for mensaje in estado.chat:
        with st.chat_message(mensaje["role"], avatar="🧑‍🍳" if mensaje["role"] == "user" else "🤖"):
            st.markdown(mensaje["content"])
            if mensaje.get("tabla") is not None:
                mostrar_lista(mensaje["tabla"])
            if mensaje.get("validacion"):
                mostrar_validacion(mensaje["validacion"], mensaje.get("uso"))

    pregunta = st.chat_input("Ej.: ¿por qué aparece S1? · ¿y si solo quiero postres? · "
                             "¿hay sesgo a lo popular?")
    if pregunta and puede_llamar():
        conversacion = [{"role": m["role"], "content": m["content"]} for m in estado.chat]
        estado.chat.append({"role": "user", "content": pregunta})
        try:
            datos, uso, validacion = chatear(contexto, tablas, conversacion, pregunta, prompts, cfg)
            registrar("chat", uso)
            estado.chat.append({"role": "assistant", "content": datos["respuesta"],
                                "validacion": validacion, "uso": uso.como_dict()})
            if datos["reconsulta"]:
                try:
                    nuevo_hist, nuevos_filtros, descripcion = aplicar_reconsulta(
                        rec, historial, filtros, datos["reconsulta"], tablas)
                    ctx2, tablas2 = construir_contexto(
                        rec, nuevo_hist, top_n, nuevos_filtros, linea_base, metricas,
                        nota=f"Reconsulta pedida en el chat: {descripcion}.")
                    tabla_nueva = tablas2.get("sasrec", tablas2.get("popularidad_frio"))
                    if puede_llamar():
                        texto2, uso2, val2 = analizar(
                            ctx2, tablas2, prompts, cfg,
                            instruccion=(f"La persona preguntó: «{pregunta}». Se reconsultó al "
                                         f"Transformer con: {descripcion}. Analiza el NUEVO resultado "
                                         "en máximo 150 palabras y mantén la línea AUDITORIA."))
                        registrar("analisis_reconsulta", uso2)
                        estado.chat.append({
                            "role": "assistant",
                            "content": f"🔁 **Reconsulta al modelo:** {descripcion}\n\n{texto2}",
                            "tabla": tabla_nueva, "validacion": val2, "uso": uso2.como_dict()})
                except ValueError as error:
                    estado.chat.append({"role": "assistant",
                                        "content": f"No pude reconsultar al modelo: {error}"})
        except ErrorLLM as error:
            estado.chat.append({"role": "assistant", "content": f"Error del LLM: {error}"})
        st.rerun()

    if estado.registros_llm:
        st.caption(f"Llamadas al LLM en esta sesión: {estado.llamadas}/{MAX_LLAMADAS_SESION}")
        st.dataframe(costo_resumen(estado.registros_llm), width="stretch")

# ---------------------------------------------------------------- 4. catálogo
with tab_cat:
    c1, c2, c3 = st.columns(3)
    c1.metric("Recetas en el catálogo", f"{rec.n_items:,}".replace(",", "."))
    c2.metric("Usuarios de entrenamiento", f"{rec.n_usuarios_train:,}".replace(",", "."))
    c3.metric("Recetas 'populares' (top 10 %)", f"{len(rec.populares):,}".replace(",", "."))

    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        st.subheader("⭐ Las 20 recetas más reseñadas (entrenamiento)")
        populares = rec.catalogo.sort_values("rank_popularidad").head(20)
        st.dataframe(populares[["rank_popularidad", "nombre", "categoria", "minutos", "usuarios_train"]]
                     .rename(columns={"rank_popularidad": "#", "nombre": "Receta",
                                      "categoria": "Categoría", "minutos": "Min",
                                      "usuarios_train": "Reseñas"}),
                     hide_index=True, width="stretch")
    with derecha:
        st.subheader("Recetas por categoría")
        conteo = rec.catalogo["categoria"].value_counts().rename_axis("Categoría").reset_index(name="Recetas")
        st.altair_chart(alt.Chart(conteo).mark_bar(color="#E9A23B").encode(
            x="Recetas:Q", y=alt.Y("Categoría:N", sort="-x")), width="stretch")

    st.subheader("🔎 Explorar el catálogo")
    texto = st.text_input("Buscar por nombre (en inglés)", key="buscar_catalogo")
    if texto:
        res = rec.buscar(texto, n=50)
        st.dataframe(res[["nombre", "categoria", "minutos", "n_ingredientes", "usuarios_train"]]
                     .rename(columns={"nombre": "Receta", "categoria": "Categoría", "minutos": "Min",
                                      "n_ingredientes": "Ingredientes", "usuarios_train": "Reseñas"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5. métricas
with tab_met:
    st.subheader("Evaluación offline (prueba, leave-last-out, catálogo completo)")
    if metricas is not None:
        tabla = metricas.pivot_table(index="Modelo", columns="K",
                                     values=["HR", "NDCG", "MRR", "Cobertura"])
        tabla.columns = [f"{m}@{k}" for m, k in tabla.columns]
        st.dataframe(tabla.round(4), width="stretch")
        st.caption("Con 17.813 usuarios y 41.240 recetas, SASRec y popularidad tienen precisión muy "
                   "parecida; SASRec recomienda un catálogo ~10 veces más amplio (cobertura).")
    for archivo, titulo in [("costos_modelos.csv", "Costo de los modelos"),
                            ("auditoria_modelos.csv", "Auditor de sesgo sobre todos los usuarios de prueba"),
                            ("metricas_segmentos.csv", "Segmentos (K = 10)"),
                            ("pruebas_honestidad.csv", "Pruebas de honestidad del analista"),
                            ("costo_llm.csv", "Costo del analista (notebook 2)")]:
        df = leer_csv(archivo)
        if df is not None:
            st.subheader(titulo)
            if archivo == "metricas_segmentos.csv":
                df = df[df["K"] == 10]
            st.dataframe(df, hide_index=True, width="stretch")

st.divider()
st.caption("Datos: Food.com (Kaggle). Sin identificadores ni datos personales: los historiales de "
           "ejemplo se construyen con recetas del catálogo. Asociación no es causalidad.")
