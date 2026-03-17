"""Pipeline completo: PDF escaneado → texto relevante."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from spharagusnet.model import DocumentTextExtractor

# ---------------------------------------------------------------------------
# Carga del modelo (singleton)
# ---------------------------------------------------------------------------

_model_cache: dict = {}   # {'model': ..., 'vocab': ...}


def load_model(
    model_path: Path | str | None = None,
    vocab_path: Path | str | None = None,
) -> Tuple["DocumentTextExtractor", Dict[str, int]]:
    """
    Carga el modelo y vocabulario.  Cachea en memoria para no recargar.

    Returns:
        (modelo, vocabulario)
    """
    import torch

    from spharagusnet.model import DEVICE, DocumentTextExtractor
    from spharagusnet.paths import get_model_path, get_vocab_path
    from spharagusnet.tokenizer import cargar_vocab

    model_path = Path(model_path) if model_path else get_model_path()
    vocab_path = Path(vocab_path) if vocab_path else get_vocab_path()

    cache_key = str(model_path)
    if cache_key in _model_cache:
        return _model_cache[cache_key]["model"], _model_cache[cache_key]["vocab"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            "Ejecuta: spharagusnet download"
        )
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulario no encontrado: {vocab_path}")

    vocab = cargar_vocab(vocab_path)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    vocab_size = checkpoint.get("vocab_size", len(vocab))
    hidden_dim = checkpoint.get("hidden_dim", 512)

    model = DocumentTextExtractor(
        vocab_size=vocab_size, hidden_dim=hidden_dim, device=DEVICE,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model_cache[cache_key] = {"model": model, "vocab": vocab}
    return model, vocab


# ---------------------------------------------------------------------------
# Clasificación de bloques
# ---------------------------------------------------------------------------

def _clasificar_bloques_modelo(
    bloques: List[Dict],
    modelo: "DocumentTextExtractor",
    vocab: Dict[str, int],
    confidence: float,
) -> List[str]:
    """Filtra bloques relevantes usando el modelo."""
    import torch

    from spharagusnet.model import DEVICE
    from spharagusnet.tokenizer import MAX_SEQ_LEN, tokenizar

    tokens_list = [tokenizar(b["texto"], vocab) for b in bloques]
    tokens_t = torch.tensor(tokens_list, dtype=torch.long, device=DEVICE)
    positions = torch.zeros(len(bloques), MAX_SEQ_LEN, 2, device=DEVICE)

    with torch.no_grad():
        _, scores = modelo(tokens_t, positions)
        probs = scores.softmax(dim=1)
        preds = scores.argmax(dim=1)
        confs = probs[:, 1].cpu().numpy()

    return [
        b["texto"]
        for b, pred, conf in zip(bloques, preds, confs)
        if pred == 1 and conf >= confidence
    ]


def _clasificar_bloques_heuristica(
    bloques: List[Dict], img_height: int,
) -> List[str]:
    """Fallback: filtra bloques con reglas manuales."""
    relevantes: List[str] = []
    for bloque in bloques:
        texto = bloque["texto"]
        tl = texto.lower()

        if any(p in tl for p in (
            "diario oficial", "página", "pág.", "pag.",
            "biblioteca del congreso", "bcn", "ley chile",
            "140 años", "fecha publicación", "fecha promulgación",
            "url corta", "qr",
        )):
            continue
        if len(texto.strip()) <= 3 and texto.strip().isdigit():
            continue
        if ("http" in tl or "www" in tl) and len(texto) < 50:
            continue

        y = bloque["y"]
        if y < img_height * 0.05 or y > img_height * 0.95:
            if len(texto) < 50 or any(p in tl for p in ("diario", "oficial", "bcn")):
                continue

        relevantes.append(texto)
    return relevantes


def _limpiar(textos: List[str]) -> str:
    txt = " ".join(textos)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\n\s*\n", "\n\n", txt)
    return txt.strip()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract(
    pdf_path: str | Path,
    confidence: float = 0.5,
    model_path: str | Path | None = None,
    vocab_path: str | Path | None = None,
    dpi: int = 200,
) -> str:
    """
    Extrae texto relevante de un PDF escaneado.

    Args:
        pdf_path:    Ruta al PDF.
        confidence:  Umbral de confianza del modelo [0-1].
        model_path:  Ruta al checkpoint .pth (usa el default si None).
        vocab_path:  Ruta al vocab.json (usa el default si None).
        dpi:         DPI para la rasterización del PDF.

    Returns:
        Texto relevante concatenado.
    """
    from pdf2image import convert_from_path

    from spharagusnet.ocr import extraer_bloques

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    # Intentar cargar modelo; si falla, usar heurísticas
    try:
        modelo, vocab = load_model(model_path, vocab_path)
        usar_modelo = True
    except (FileNotFoundError, RuntimeError, ImportError):
        modelo, vocab = None, None
        usar_modelo = False

    textos_paginas: List[str] = []
    page_num = 1

    while True:
        try:
            pages = convert_from_path(
                str(pdf_path), dpi=dpi,
                first_page=page_num, last_page=page_num,
            )
        except Exception:
            break
        if not pages:
            break

        page = pages[0]

        # Guardar temporalmente para OCR
        tmp = pdf_path.parent / f"_spharagus_tmp_{pdf_path.stem}_{page_num}.png"
        page.save(tmp, "PNG")

        try:
            bloques = extraer_bloques(tmp)

            if bloques:
                if usar_modelo:
                    relevantes = _clasificar_bloques_modelo(
                        bloques, modelo, vocab, confidence,
                    )
                else:
                    relevantes = _clasificar_bloques_heuristica(
                        bloques, page.height,
                    )
                texto_pagina = _limpiar(relevantes)
                if texto_pagina:
                    textos_paginas.append(texto_pagina)
        finally:
            tmp.unlink(missing_ok=True)

        del page, pages
        gc.collect()
        page_num += 1

    return "\n\n".join(textos_paginas)
