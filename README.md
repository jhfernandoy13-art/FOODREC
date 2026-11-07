# 🍲 FoodRec — recetas que entienden el orden

Recomendador secuencial de recetas (Food.com) basado en un **Transformer (SASRec)**, comparado con
las líneas base del curso y publicado como **dashboard Streamlit** con un **analista de IA generativa**
que *no recomienda*: analiza lo que el modelo recomendó.

> **Regla de oro.** El Transformer recomienda: la lista y su orden salen del modelo.
> La IA no agrega, quita ni reordena recetas; explica, contrasta con las líneas base,
> advierte el sesgo de popularidad y los límites, y responde preguntas.

**App publicada:** `https://<completar>.streamlit.app` · **Presentación:** `<link de Canva>`

---

## Rubro, datos y decisión que apoya

- **Base:** [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) (reseñas con fecha).
- **Tras limpieza y 5-core:** 555.618 interacciones · 17.813 usuarios · 41.240 recetas.
- **Atributos de ítem:** categoría (tags), minutos, ingredientes.
- **Usuario concreto y decisión:** una persona que cocina en casa y decide *qué receta preparar después*,
  según lo que ha venido cocinando (y en qué orden).

## Estructura del repositorio

```
app.py                         Dashboard Streamlit (carga sasrec.pt en CPU con caché; no entrena)
src/modelo.py                  Arquitectura SASRec (misma que el notebook 1)
src/recomendador.py            SASRec + líneas base + auditor de sesgo (innovación)
src/analista.py                Contexto estructurado, llamadas al LLM, validador de honestidad, reconsulta
src/llm.py                     Cliente HTTP para Gemini / Groq / Claude con tokens y latencia
src/analista_prompt.md         Instrucciones de sistema VERSIONADAS del analista
notebooks/FoodRec_colab.ipynb Notebook único: escribe estos archivos, Entrega 1 y Entrega 2
data/secuencias.parquet        Secuencias por usuario (índices internos, sin IDs originales)
data/mapa_items.csv            item_idx → receta
data/metricas.csv              Métricas de prueba de los 4 modelos (K = 5, 10, 20)
data/catalogo_app.parquet      Catálogo liviano para la app (lo genera el notebook 2)
data/*.csv                     Costos, auditoría, segmentos, pruebas de honestidad, costo del LLM
modelos/sasrec.pt              Checkpoint seleccionado con validación (< 100 MB)
modelos/popularidad.csv · items_normalizados.npz · reglas_asociacion.csv   Líneas base
Resultados/                    Curvas, figuras, rankings de prueba y huellas del experimento
```

## Cómo reproducir

### 1. Notebooks en Google Colab
1. Clona este repositorio en `MyDrive/FoodRec` (o sube la carpeta a Drive).
2. Pon `RAW_interactions.csv` y `RAW_recipes.csv` en `MyDrive/FoodRec/Data_raw/`
   (o deja que el notebook los descargue con `kagglehub`).
3. Abre `notebooks/FoodRec_colab.ipynb` en Colab → *Entorno de ejecución → GPU (T4)* → *Ejecutar todas*.
   - **Camino A (recomendado):** con los `data/`, `modelos/` y `Resultados/` de la Entrega 1 ya en la
     carpeta, rehace datos y líneas base, pero reutiliza `sasrec.pt` y los rankings de prueba
     (verificados con huellas SHA-256): las cifras quedan iguales a las del informe de la Entrega 1.
   - **Camino B:** con esas carpetas vacías, entrena las 3 configuraciones en GPU y evalúa la prueba una vez.
   - Semilla fija (42) en partición, negativos y PyTorch. Si vuelves a ejecutar, **no reentrena**
     (`REENTRENAR = False`) y **no reevalúa la prueba**.
4. Antes de ejecutar, agrega el secreto `LLM_API_KEY` (ícono 🔑): la parte de Entrega 2 lo usa.

### 2. App local
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # y pega tu clave
streamlit run app.py
```
Sin clave, el analista funciona en **modo simulado** (respuesta fija, solo para probar la interfaz).

### 3. Publicación (Streamlit Community Cloud)
*New app* → este repositorio → `app.py` → Python 3.12. En *Settings → Secrets* pega:
```toml
LLM_PROVEEDOR = "gemini"
LLM_API_KEY = "tu-clave"
```
**La clave nunca va en GitHub** (`.streamlit/secrets.toml` está en `.gitignore`).

## El dashboard

| Pestaña | Qué muestra |
|---|---|
| 🍽️ Recomendaciones | Top-N de SASRec junto a popularidad, ítem–ítem y reglas; auditor de sesgo de cada lista |
| 🔍 Transparencia | Pesos de atención sobre el historial, reglas A → B que se activaron, receta más parecida (ítem–ítem) |
| 🤖 Analista IA | Análisis inicial automático + chat; si la pregunta exige otra lista, se reconsulta al modelo |
| 📚 Catálogo | Recetas más reseñadas, categorías y buscador |
| 📊 Métricas y costo | HR/NDCG/MRR/cobertura de prueba, costos, segmentos, auditoría, pruebas de honestidad, tokens |

El **simulador de historial** (barra lateral) permite agregar recetas, moverlas ↑↓ y quitarlas: el orden importa.
Incluye historiales de ejemplo armados con recetas del catálogo (sin usuarios reales) y el caso de
**arranque en frío** (sin historial → popularidad, declarada explícitamente como no personalizada).

## El analista con IA

- **Contexto estructurado real** (`construir_contexto`): historial con categoría, minutos, ingredientes y
  atención; top-N con puntajes; una línea base con su explicación; coincidencias; auditoría de sesgo;
  patrones calculados (cambio de gusto, repetición de categoría); métricas offline y límites conocidos.
- **Códigos obligatorios** `[H#] [S#] [B#] [P#]`: el analista solo puede citar recetas que recibió.
- **Validador automático** de cada respuesta: códigos inventados, listas numeradas que no respetan el
  orden del modelo, y nivel de sesgo declarado vs. calculado.
- **Reconsulta:** el chat responde en JSON; si la pregunta pide otra lista, el LLM solo puede solicitar
  5 tipos de reconsulta (categoría, tiempo máximo, quitar/agregar receta al historial, sin filtros),
  validados en Python. La app vuelve a ejecutar SASRec y el analista analiza el nuevo resultado.
- **Cuota:** caché por contexto, historial recortado a 20 recetas y máximo 30 llamadas por sesión.
- **Pruebas de honestidad:** 5 pruebas (+ reconsulta) documentadas en el notebook 2 y en `data/pruebas_honestidad.csv`.

## Innovación: auditor de sesgo de popularidad, novedad y diversidad

**Problema.** Popularidad y SASRec tienen precisión casi igual, pero popularidad recomienda casi siempre
las mismas ~85 recetas. El HR@K no muestra *qué tan obvias* son las recomendaciones.

**Qué tiene de distinto.** Para cada lista se calcula en Python (solo con entrenamiento) la cuota de
recetas populares, la novedad (bits), la diversidad de ingredientes y la afinidad con el historial, con
un nivel de sesgo alto/medio/bajo. La app lo muestra para los cuatro modelos y el analista debe
interpretarlo y advertir el sesgo, sin poder cambiarlo.

**Cómo lo demostramos.** (1) Offline, sobre los 17.813 usuarios de prueba (`data/auditoria_modelos.csv`);
(2) en 30 casos con el LLM medimos si el analista reporta el mismo nivel de sesgo que el auditor,
sin inventar ni reordenar (`data/calidad_analista_30.csv`).

## Ética y límites

- Sin identificadores ni datos personales en la app, en el contexto del LLM ni en el repositorio
  (índices internos; historiales de ejemplo construidos con recetas del catálogo).
- Las interacciones son reseñas, no prueba de que alguien cocinó la receta.
- Asociación no es causalidad; predecir la siguiente receta no explica por qué la persona la eligió;
  la atención describe el modelo, no las razones de la persona.

## Uso de IA (declaración)

- **Para programar:** `<completar: herramientas usadas por el grupo>`.
- **Dentro del producto:** `<proveedor y modelo del analista, p. ej. Gemini Flash (el modelo exacto aparece en el notebook, sección E2.6)>`.

## Equipo

`<integrantes>` — todos los integrantes deben tener commits en este repositorio.
