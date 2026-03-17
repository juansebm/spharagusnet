"""Vocabulario y tokenización para el clasificador de bloques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PAD_TOKEN = 0
UNK_TOKEN = 1
MAX_SEQ_LEN = 64


def tokenizar(texto: str, vocab: Dict[str, int],
              max_len: int = MAX_SEQ_LEN) -> List[int]:
    """Convierte texto a lista de índices con padding a *max_len*."""
    palabras = texto.lower().split()
    tokens = [vocab.get(p, UNK_TOKEN) for p in palabras[:max_len]]
    tokens += [PAD_TOKEN] * (max_len - len(tokens))
    return tokens


def cargar_vocab(vocab_path: Path) -> Dict[str, int]:
    """Lee un vocabulario desde un archivo JSON."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)
