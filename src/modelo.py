"""Arquitectura SASRec (idéntica a la del notebook de la Entrega 1).

Se replica aquí para que la app cargue `modelos/sasrec.pt` sin depender del
notebook. Los nombres de los módulos deben coincidir con el `state_dict`.
"""
import math

import torch
from torch import nn


class BloqueSASRec(nn.Module):
    """Pre-norm: LayerNorm → atención causal → residual → LayerNorm → FFN → residual."""

    def __init__(self, dimension, cabezas, dropout):
        super().__init__()
        self.norm_atencion = nn.LayerNorm(dimension)
        self.atencion = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=cabezas,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mascara_causal, mascara_padding, devolver_atencion=False):
        normalizado = self.norm_atencion(x)
        salida, pesos = self.atencion(
            normalizado, normalizado, normalizado,
            attn_mask=mascara_causal,
            key_padding_mask=mascara_padding,
            need_weights=devolver_atencion,
            average_attn_weights=False,
        )
        x = x + self.dropout(salida)
        x = x + self.dropout(self.feed_forward(self.norm_ff(x)))
        x = x.masked_fill(mascara_padding.unsqueeze(-1), 0.0)
        return x, pesos


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, dimension=64, cabezas=2, bloques=1, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.dimension = dimension
        self.item_embedding = nn.Embedding(n_items + 1, dimension, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, dimension)
        self.dropout = nn.Dropout(dropout)
        self.bloques = nn.ModuleList(
            [BloqueSASRec(dimension, cabezas, dropout) for _ in range(bloques)]
        )
        self.norm_final = nn.LayerNorm(dimension)

    def forward(self, items, devolver_atencion=False):
        longitud = items.size(1)
        posiciones = torch.arange(longitud, device=items.device)
        mascara_padding = items.eq(0)

        x = (
            self.item_embedding(items) * math.sqrt(self.dimension)
            + self.pos_embedding(posiciones).unsqueeze(0)
        )
        x = self.dropout(x)
        x = x.masked_fill(mascara_padding.unsqueeze(-1), 0.0)

        # True sobre la diagonal: una posición no puede mirar el futuro
        mascara_causal = torch.triu(
            torch.ones(longitud, longitud, dtype=torch.bool, device=items.device),
            diagonal=1,
        )

        atenciones = []
        for bloque in self.bloques:
            x, pesos = bloque(x, mascara_causal, mascara_padding, devolver_atencion)
            if devolver_atencion:
                atenciones.append(pesos)

        x = self.norm_final(x)
        x = x.masked_fill(mascara_padding.unsqueeze(-1), 0.0)

        if devolver_atencion:
            return x, atenciones
        return x


def cargar_sasrec(ruta_checkpoint):
    """Carga el checkpoint en CPU y devuelve (modelo en eval, checkpoint)."""
    checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    modelo = SASRec(
        n_items=checkpoint["n_items"],
        max_len=checkpoint["max_len"],
        dimension=config["dimension"],
        cabezas=config["cabezas"],
        bloques=config["bloques"],
        dropout=config["dropout"],
    )
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    return modelo, checkpoint
