"""Rutas para modelos y cache de SpharagusNet."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Directorio de cache del usuario ──────────────────────────────────────────
# Windows: %LOCALAPPDATA%/spharagusnet
# Linux/Mac: ~/.cache/spharagusnet

if sys.platform == "win32":
    _CACHE_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
else:
    _CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

CACHE_DIR = _CACHE_ROOT / "spharagusnet"
MODEL_CACHE_DIR = CACHE_DIR / "models" / "texto_extractor"

# ── Directorio local (para desarrollo) ───────────────────────────────────────
_DEV_DIR = Path(__file__).parent.parent / "models" / "texto_extractor"


def get_model_path() -> Path:
    """Retorna la ruta al modelo, priorizando el directorio de desarrollo."""
    # 1. Desarrollo local (cuando trabajas dentro del repo)
    dev = _DEV_DIR / "modelo_entrenado.pth"
    if dev.exists():
        return dev
    # 2. Cache del usuario (descargado con download_model)
    return MODEL_CACHE_DIR / "modelo_entrenado.pth"


def get_vocab_path() -> Path:
    """Retorna la ruta al vocabulario, priorizando el directorio de desarrollo."""
    dev = _DEV_DIR / "vocab.json"
    if dev.exists():
        return dev
    return MODEL_CACHE_DIR / "vocab.json"


def model_is_available() -> bool:
    """True si el modelo está disponible (local o descargado)."""
    return get_model_path().exists() and get_vocab_path().exists()
