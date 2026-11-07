"""Recomendador de la app: SASRec + líneas base + auditor (innovación).

Todo se calcula con artefactos de la Entrega 1 (solo entrenamiento):
- modelos/sasrec.pt               checkpoint seleccionado con validación
- modelos/popularidad.csv         usuarios por receta en entrenamiento
- modelos/items_normalizados.npz  matriz receta-usuario normalizada (coseno)
- modelos/reglas_asociacion.csv   reglas A → B con lift > 1
- data/mapa_items.csv             item_idx → receta (nombre, minutos, tags, ingredientes)

Nada se entrena aquí. La lista y su orden salen siempre de los modelos.
"""
from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, load_npz

from .modelo import cargar_sasrec

# Orden fijo de categorías visibles (igual que la PCA de la Entrega 1)
CATEGORIAS = [
    ("desserts", "Postres"),
    ("breads", "Panes"),
    ("soups-stews", "Sopas y guisos"),
    ("salads", "Ensaladas"),
    ("main-dish", "Platos principales"),
]
CATEGORIAS_FILTRO = [nombre for _, nombre in CATEGORIAS]
UMBRAL_SESGO_ALTO = 0.60   # ≥ 60 % de la lista en el top 10 % más popular
UMBRAL_SESGO_BAJO = 0.30   # ≤ 30 %
MINUTOS_RAPIDA = 30

# Historiales de ejemplo para la demo: se arman con recetas del catálogo,
# no con usuarios reales (sin identificadores ni datos personales).
PERSONAS = {
    "Repostería de fin de semana": [
        "brownies", "banana bread", "chocolate chip cookies",
        "cheesecake", "cinnamon rolls", "apple pie",
    ],
    "Olla lenta y comida casera": [
        "crock pot", "slow cooker", "pot roast", "chili", "meatloaf", "beef stew",
    ],
    "Cambio de gusto: de dulces a saludable": [
        "brownies", "cupcakes", "fudge", "cheesecake",
        "salad", "quinoa", "grilled chicken", "vegetable soup",
    ],
}


def nombre_legible(nombre: str) -> str:
    """'kittencal s italian  meatballs' → "Kittencal's italian meatballs"."""
    texto = re.sub(r"\s+", " ", str(nombre)).strip()
    texto = re.sub(r"\b(\w+) s\b", r"\1's", texto)
    return texto[:1].upper() + texto[1:] if texto else texto


def categoria_principal(tags) -> str:
    for etiqueta, nombre in CATEGORIAS:
        if etiqueta in tags:
            return nombre
    return "Otras"


def construir_catalogo(raiz: Path) -> pd.DataFrame:
    """Catálogo liviano para la app (se guarda en data/catalogo_app.parquet)."""
    raiz = Path(raiz)
    mapa = pd.read_csv(raiz / "data" / "mapa_items.csv")
    pop = pd.read_csv(raiz / "modelos" / "popularidad.csv")

    tags = mapa["tags"].apply(lambda t: set(ast.literal_eval(t)))
    ingredientes = mapa["ingredients"].apply(
        lambda t: [str(i).strip().lower() for i in ast.literal_eval(t)]
    )

    catalogo = pd.DataFrame({
        "item_idx": mapa["item_idx"].astype(int),
        "nombre": mapa["name"].apply(nombre_legible),
        # Algunas recetas tienen minutos absurdos; se acotan solo para mostrar
        "minutos": mapa["minutes"].clip(lower=0, upper=24 * 60).astype(int),
        "categoria": tags.apply(categoria_principal),
        "ingredientes": ingredientes,
        "n_ingredientes": ingredientes.apply(len),
    })
    for etiqueta, nombre in CATEGORIAS:
        catalogo[f"es_{etiqueta}"] = tags.apply(lambda t, e=etiqueta: e in t)

    catalogo = catalogo.merge(pop, on="item_idx", how="left")
    catalogo["usuarios_train"] = catalogo["usuarios_train"].fillna(0).astype(int)

    orden = pop["item_idx"].tolist()
    rango = {item: i + 1 for i, item in enumerate(orden)}
    catalogo["rank_popularidad"] = catalogo["item_idx"].map(rango).astype(int)
    n_populares = max(1, math.ceil(len(catalogo) * 0.10))
    catalogo["es_popular"] = catalogo["rank_popularidad"] <= n_populares
    return catalogo.sort_values("item_idx").reset_index(drop=True)


class Recomendador:
    """Carga todos los artefactos una vez (en la app, con st.cache_resource)."""

    def __init__(self, raiz):
        self.raiz = Path(raiz)

        self.modelo, self.checkpoint = cargar_sasrec(self.raiz / "modelos" / "sasrec.pt")
        self.max_len = int(self.checkpoint["max_len"])
        self.n_items = int(self.checkpoint["n_items"])
        self.config = dict(self.checkpoint["config"])

        ruta_catalogo = self.raiz / "data" / "catalogo_app.parquet"
        if ruta_catalogo.is_file():
            catalogo = pd.read_parquet(ruta_catalogo)
            catalogo["ingredientes"] = catalogo["ingredientes"].apply(list)
        else:
            catalogo = construir_catalogo(self.raiz)
        self.catalogo = catalogo.set_index("item_idx", drop=False)
        self.catalogo.index.name = None  # evita ambigüedad índice/columna
        assert len(self.catalogo) == self.n_items, "El catálogo no coincide con el modelo."

        pop = pd.read_csv(self.raiz / "modelos" / "popularidad.csv")
        self.orden_popularidad = pop["item_idx"].to_numpy()
        self.conteos = np.zeros(self.n_items + 1, dtype=np.int64)
        self.conteos[pop["item_idx"].to_numpy()] = pop["usuarios_train"].to_numpy()

        meta = self.raiz / "data" / "meta_app.json"
        if meta.is_file():
            self.n_usuarios_train = int(json.loads(meta.read_text())["n_usuarios_train"])
        else:
            usuarios = pd.read_parquet(
                self.raiz / "data" / "secuencias.parquet", columns=["user_idx"]
            )
            self.n_usuarios_train = int(usuarios["user_idx"].nunique())

        n_populares = max(1, math.ceil(self.n_items * 0.10))
        self.populares = set(self.orden_popularidad[:n_populares].tolist())

        self.items_norm = load_npz(self.raiz / "modelos" / "items_normalizados.npz").tocsr()

        reglas = pd.read_csv(self.raiz / "modelos" / "reglas_asociacion.csv")
        self.reglas = reglas
        self.reglas_por_antecedente: dict[int, list] = {}
        for r in reglas.itertuples(index=False):
            self.reglas_por_antecedente.setdefault(int(r.antecedente), []).append(
                (int(r.consecuente), float(r.lift), float(r.confidence), float(r.support))
            )

        self.ingredientes = {
            int(i): frozenset(ings)
            for i, ings in zip(self.catalogo["item_idx"], self.catalogo["ingredientes"])
        }
        self.minutos = self.catalogo["minutos"].to_numpy()
        self._personas = None

    # ------------------------------------------------------------ utilidades
    def nombre(self, item: int) -> str:
        return self.catalogo.at[int(item), "nombre"]

    def mascara_filtros(self, filtros: dict | None) -> np.ndarray:
        """Vector booleano (n_items + 1) de recetas que cumplen los filtros."""
        mascara = np.ones(self.n_items + 1, dtype=bool)
        mascara[0] = False
        if not filtros:
            return mascara
        categoria = filtros.get("categoria")
        if categoria:
            etiqueta = dict((n, e) for e, n in CATEGORIAS)[categoria]
            permitidas = self.catalogo.loc[self.catalogo[f"es_{etiqueta}"], "item_idx"].to_numpy()
            nueva = np.zeros_like(mascara)
            nueva[permitidas] = True
            mascara &= nueva
        max_minutos = filtros.get("max_minutos")
        if max_minutos:
            tiempos = np.full(self.n_items + 1, np.inf)
            tiempos[self.catalogo["item_idx"].to_numpy()] = self.catalogo["minutos"].to_numpy()
            mascara &= tiempos <= float(max_minutos)
        return mascara

    def buscar(self, texto: str, n: int = 20) -> pd.DataFrame:
        texto = str(texto).strip().lower()
        if not texto:
            return self.catalogo.iloc[0:0]
        coincide = self.catalogo["nombre"].str.lower().str.contains(texto, regex=False)
        return (
            self.catalogo[coincide]
            .sort_values(["usuarios_train", "item_idx"], ascending=[False, True])
            .head(n)
        )

    def personas(self) -> dict[str, list[int]]:
        if self._personas is None:
            self._personas = {}
            for persona, palabras in PERSONAS.items():
                elegidos = []
                for palabra in palabras:
                    candidatos = self.buscar(palabra, n=50)
                    for item in candidatos["item_idx"]:
                        if int(item) not in elegidos:
                            elegidos.append(int(item))
                            break
                self._personas[persona] = elegidos
        return self._personas

    # ------------------------------------------------------------ SASRec
    @torch.no_grad()
    def puntajes_sasrec(self, historial: list[int]):
        """Puntajes del Transformer para todo el catálogo + atención de la última posición."""
        assert historial, "SASRec necesita al menos una receta en el historial."
        recientes = list(historial)[-self.max_len:]
        largo = len(recientes)
        entrada = torch.zeros((1, self.max_len), dtype=torch.long)
        entrada[0, :largo] = torch.tensor(recientes, dtype=torch.long)

        representaciones, atenciones = self.modelo(entrada, devolver_atencion=True)
        perfil = representaciones[0, largo - 1]
        puntajes = (perfil @ self.modelo.item_embedding.weight.T).numpy().astype(np.float64)
        puntajes[0] = -np.inf
        puntajes[list(historial)] = -np.inf

        # [lote, cabezas, consulta, clave] → promedio de cabezas del último bloque
        atencion = atenciones[-1][0, :, largo - 1, :largo].mean(dim=0).numpy()
        return puntajes, atencion, recientes

    def recomendar_sasrec(self, historial, top_n=10, filtros=None):
        puntajes, atencion, recientes = self.puntajes_sasrec(historial)
        puntajes = puntajes.copy()
        puntajes[~self.mascara_filtros(filtros)] = -np.inf
        validos = int(np.isfinite(puntajes).sum())
        n = min(top_n, validos)
        orden = np.argsort(-puntajes, kind="stable")[:n]
        tabla = pd.DataFrame({
            "rank": np.arange(1, n + 1),
            "item_idx": orden.astype(int),
            "puntaje": puntajes[orden],
        })
        return self._enriquecer(tabla), atencion, recientes

    # ------------------------------------------------------------ líneas base
    def recomendar_popularidad(self, historial, top_n=10, filtros=None):
        vistos = set(historial)
        mascara = self.mascara_filtros(filtros)
        elegidos = []
        for item in self.orden_popularidad:
            if item not in vistos and mascara[item]:
                elegidos.append(int(item))
            if len(elegidos) == top_n:
                break
        tabla = pd.DataFrame({
            "rank": np.arange(1, len(elegidos) + 1),
            "item_idx": elegidos,
            "puntaje": self.conteos[elegidos].astype(float),
        })
        return self._enriquecer(tabla)

    def recomendar_item_item(self, historial, top_n=10, filtros=None, ultimos=20, decaimiento=0.9):
        """Ítem–ítem con coseno ponderando las últimas recetas (igual que la Entrega 1)."""
        if not historial:
            return self.recomendar_popularidad(historial, top_n, filtros)
        recientes = list(historial)[-ultimos:]
        pesos = decaimiento ** np.arange(len(recientes) - 1, -1, -1, dtype=np.float64)
        pesos /= pesos.sum()
        vector = csr_matrix(
            (pesos, (np.zeros(len(recientes), dtype=int), recientes)),
            shape=(1, self.n_items + 1),
        )
        perfil = vector @ self.items_norm
        puntajes = (perfil @ self.items_norm.T).toarray().ravel()
        puntajes[0] = -np.inf
        puntajes[list(historial)] = -np.inf
        puntajes[~self.mascara_filtros(filtros)] = -np.inf

        candidatos = self.orden_popularidad
        orden = np.argsort(-puntajes[candidatos], kind="stable")
        elegidos = [int(candidatos[i]) for i in orden if np.isfinite(puntajes[candidatos[i]])][:top_n]

        # Transparencia: receta reciente del historial más parecida a cada recomendada
        similitud = (self.items_norm[elegidos] @ self.items_norm[recientes].T).toarray()
        mas_parecida = [recientes[int(np.argmax(fila))] for fila in similitud]
        tabla = pd.DataFrame({
            "rank": np.arange(1, len(elegidos) + 1),
            "item_idx": elegidos,
            "puntaje": puntajes[elegidos],
            "mas_parecida_a": mas_parecida,
            "coseno_max": similitud.max(axis=1) if len(elegidos) else [],
        })
        return self._enriquecer(tabla)

    def recomendar_reglas(self, historial, top_n=10, filtros=None):
        """Reglas A → B ordenadas por lift; se completa con popularidad (explícito)."""
        vistos = set(historial)
        mascara = self.mascara_filtros(filtros)
        candidatos = {}
        for antecedente in vistos:
            for item, lift, confianza, soporte in self.reglas_por_antecedente.get(antecedente, []):
                if item in vistos or not mascara[item]:
                    continue
                calidad = (lift, confianza, soporte, antecedente)
                if item not in candidatos or calidad[:3] > candidatos[item][:3]:
                    candidatos[item] = calidad
        ordenados = sorted(
            candidatos,
            key=lambda i: (-candidatos[i][0], -candidatos[i][1], -candidatos[i][2], i),
        )[:top_n]
        filas = [
            {"item_idx": i, "puntaje": candidatos[i][0], "regla": "reglas",
             "antecedente": candidatos[i][3], "confianza": candidatos[i][1]}
            for i in ordenados
        ]
        if len(filas) < top_n:
            ya = vistos | set(ordenados)
            for item in self.orden_popularidad:
                if item not in ya and mascara[item]:
                    filas.append({"item_idx": int(item), "puntaje": np.nan,
                                  "regla": "respaldo popularidad",
                                  "antecedente": None, "confianza": np.nan})
                if len(filas) == top_n:
                    break
        tabla = pd.DataFrame(filas)
        tabla.insert(0, "rank", np.arange(1, len(tabla) + 1))
        return self._enriquecer(tabla)

    def recomendar(self, modelo: str, historial, top_n=10, filtros=None):
        if modelo == "Popularidad":
            return self.recomendar_popularidad(historial, top_n, filtros)
        if modelo == "Item-Item":
            return self.recomendar_item_item(historial, top_n, filtros)
        if modelo == "Reglas + popularidad":
            return self.recomendar_reglas(historial, top_n, filtros)
        if modelo == "SASRec":
            return self.recomendar_sasrec(historial, top_n, filtros)[0]
        raise ValueError(modelo)

    def _enriquecer(self, tabla: pd.DataFrame) -> pd.DataFrame:
        cat = self.catalogo.loc[tabla["item_idx"].astype(int)]
        tabla = tabla.copy()
        tabla["receta"] = cat["nombre"].to_numpy()
        tabla["categoria"] = cat["categoria"].to_numpy()
        tabla["minutos"] = cat["minutos"].to_numpy()
        tabla["usuarios_train"] = cat["usuarios_train"].to_numpy()
        tabla["es_popular"] = cat["es_popular"].to_numpy()
        return tabla

    # ------------------------------------------------------------ innovación: auditor
    def jaccard(self, a, b) -> float:
        A, B = self.ingredientes[int(a)], self.ingredientes[int(b)]
        union = A | B
        return len(A & B) / len(union) if union else 0.0

    def auditar(self, items, historial=None) -> dict:
        """Auditor de sesgo: popularidad, novedad, diversidad y afinidad con el historial."""
        items = [int(i) for i in items]
        historial = [int(i) for i in (historial or [])]
        if not items:
            return {}
        cuota_popular = float(np.mean([i in self.populares for i in items]))
        novedad = float(np.mean([
            -math.log2((self.conteos[i] + 1) / (self.n_usuarios_train + 1)) for i in items
        ]))
        pares = list(combinations(items, 2))
        diversidad = float(np.mean([1 - self.jaccard(a, b) for a, b in pares])) if pares else 0.0
        afinidad = (
            float(np.mean([max(self.jaccard(i, h) for h in historial) for i in items]))
            if historial else None
        )
        if cuota_popular >= UMBRAL_SESGO_ALTO:
            nivel = "alto"
        elif cuota_popular <= UMBRAL_SESGO_BAJO:
            nivel = "bajo"
        else:
            nivel = "medio"
        categorias = Counter(self.catalogo.loc[items, "categoria"])
        return {
            "n": len(items),
            "cuota_popular": round(cuota_popular, 3),
            "recetas_populares": int(round(cuota_popular * len(items))),
            "novedad_bits": round(novedad, 2),
            "diversidad_ingredientes": round(diversidad, 3),
            "afinidad_historial": None if afinidad is None else round(afinidad, 3),
            "nivel_sesgo_popularidad": nivel,
            "categorias": dict(categorias.most_common()),
        }

    def patrones_historial(self, historial, items_recomendados=None) -> dict:
        """Patrones calculados en Python (el LLM solo los interpreta)."""
        historial = [int(i) for i in historial]
        if not historial:
            return {"largo_historial": 0}
        cats = list(self.catalogo.loc[historial, "categoria"])
        mitad = len(cats) // 2
        resultado = {
            "largo_historial": len(historial),
            "visibles_para_transformer": min(len(historial), self.max_len),
            "categorias_historial": dict(Counter(cats).most_common()),
            "cuota_populares_historial": round(float(np.mean([h in self.populares for h in historial])), 3),
            "mediana_minutos_historial": float(np.median(self.catalogo.loc[historial, "minutos"])),
        }
        if len(historial) >= 4:
            def dominante(lista):
                c = Counter(lista).most_common()
                if len(c) > 1 and c[0][1] == c[1][1]:
                    return ("sin categoría dominante (empate)", c[0][1])
                return c[0]
            inicio = dominante(cats[:mitad])
            final = dominante(cats[mitad:])
            resultado["cambio_de_gusto"] = {
                "detectado": inicio[0] != final[0],
                "categoria_dominante_primera_mitad": inicio[0],
                "categoria_dominante_segunda_mitad": final[0],
            }
        ultimas = cats[-5:]
        conteo = Counter(ultimas).most_common()
        empate = len(conteo) > 1 and conteo[0][1] == conteo[1][1]
        dominante_reciente = None if empate else conteo[0][0]
        resultado["categoria_dominante_ultimas_5"] = (
            dominante_reciente or "sin categoría dominante (empate)")
        if items_recomendados is not None and len(items_recomendados):
            rec_cats = list(self.catalogo.loc[[int(i) for i in items_recomendados], "categoria"])
            if dominante_reciente:
                resultado["cuota_recomendaciones_misma_categoria_reciente"] = round(
                    float(np.mean([c == dominante_reciente for c in rec_cats])), 3
                )
            resultado["mediana_minutos_recomendaciones"] = float(
                np.median(self.catalogo.loc[[int(i) for i in items_recomendados], "minutos"])
            )
        return resultado
