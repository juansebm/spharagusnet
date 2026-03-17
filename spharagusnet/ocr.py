"""Wrapper de Tesseract OCR con extracción de bloques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def extraer_bloques(imagen_path: Path, conf_min: int = 30) -> List[Dict]:
    """
    Extrae bloques de texto de una imagen con coordenadas usando Tesseract.

    Cada bloque es un dict con: texto, conf_ocr, x, y, w, h.
    Solo devuelve bloques con confianza >= *conf_min*.
    """
    from PIL import Image

    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError("pytesseract no está instalado") from exc

    imagen = Image.open(imagen_path)
    ocr_data = pytesseract.image_to_data(
        imagen, lang="spa", output_type=Output.DICT,
    )

    bloques: List[Dict] = []
    for i in range(len(ocr_data["text"])):
        texto = ocr_data["text"][i].strip()
        if not texto:
            continue
        conf = int(ocr_data["conf"][i])
        if conf < conf_min:
            continue
        bloques.append({
            "texto": texto,
            "conf_ocr": conf,
            "x": ocr_data["left"][i],
            "y": ocr_data["top"][i],
            "w": ocr_data["width"][i],
            "h": ocr_data["height"][i],
        })

    return bloques


# ---------------------------------------------------------------------------
# Cache de bloques OCR (para no re-procesar imágenes)
# ---------------------------------------------------------------------------

def cache_path_for(imagen_path: Path) -> Path:
    """Devuelve la ruta del cache JSON para una imagen."""
    return imagen_path.with_suffix(".ocr.json")


def guardar_cache(imagen_path: Path, bloques: List[Dict]) -> Path:
    """Guarda bloques OCR a un archivo JSON cache."""
    p = cache_path_for(imagen_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(bloques, f, ensure_ascii=False)
    return p


def cargar_cache(imagen_path: Path) -> List[Dict] | None:
    """Carga bloques OCR desde cache. Retorna None si no existe."""
    p = cache_path_for(imagen_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extraer_bloques_cached(imagen_path: Path, conf_min: int = 30) -> List[Dict]:
    """
    Extrae bloques con cache: si ya existe un .ocr.json, lo carga;
    si no, corre Tesseract y guarda el resultado.
    """
    cached = cargar_cache(imagen_path)
    if cached is not None:
        return cached

    bloques = extraer_bloques(imagen_path, conf_min=conf_min)
    guardar_cache(imagen_path, bloques)
    return bloques
