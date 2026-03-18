"""Descarga del modelo pre-entrenado desde GitHub Releases."""

from __future__ import annotations

from pathlib import Path

from spharagusnet.paths import MODEL_CACHE_DIR, get_model_path, get_vocab_path

# ── Configuración de la release ──────────────────────────────────────────────
# Cambia GITHUB_REPO a tu usuario/repo real antes de publicar.
GITHUB_REPO = "juansebm/spharagusnet"
RELEASE_TAG = "v0.2.1"

_BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

MODEL_FILES = {
    "modelo_entrenado.pth": f"{_BASE_URL}/modelo_entrenado.pth",
    "vocab.json": f"{_BASE_URL}/vocab.json",
}


def download_model(force: bool = False) -> Path:
    """
    Descarga el modelo pre-entrenado desde GitHub Releases.

    El modelo se guarda en el cache del usuario:
    - Linux/Mac: ``~/.cache/spharagusnet/models/texto_extractor/``
    - Windows:   ``%LOCALAPPDATA%/spharagusnet/models/texto_extractor/``

    Args:
        force: Si True, re-descarga aunque ya exista.

    Returns:
        Path al directorio con el modelo descargado.
    """
    import requests
    from tqdm import tqdm

    # Si ya está descargado (y no force), salir rápido
    if not force and get_model_path().exists() and get_vocab_path().exists():
        print(f"✅ Modelo ya disponible en {MODEL_CACHE_DIR}")
        return MODEL_CACHE_DIR

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_FILES.items():
        dest = MODEL_CACHE_DIR / filename

        if dest.exists() and not force:
            print(f"  ✓ {filename} ya existe")
            continue

        print(f"  ⬇️  Descargando {filename}...")
        headers = {"User-Agent": f"spharagusnet/{RELEASE_TAG}"}
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            disable=total == 0,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"✅ Modelo descargado en {MODEL_CACHE_DIR}")
    return MODEL_CACHE_DIR
