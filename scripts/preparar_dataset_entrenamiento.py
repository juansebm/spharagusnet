#!/usr/bin/env python3
"""
Preparar dataset de entrenamiento para extracción de texto.

Cambios vs. versión anterior:
- **Incremental**: solo descarga/procesa pares nuevos (los ya existentes se saltan).
- **Cache OCR**: guarda bloques OCR como .ocr.json junto a cada imagen,
  así el entrenamiento posterior no re-corre Tesseract.
- **Mejor etiquetado**: usa Levenshtein sobre n-gramas para decidir
  relevancia de cada bloque, en vez de overlap de palabras sueltas.
- Acepta --max-pares N desde CLI.
"""

import argparse
import csv
import gc
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from PIL import Image

# Importaciones opcionales
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False
    print("⚠️  pdf2image no disponible. Instala con: pip install pdf2image")

try:
    import pdfplumber
    PDFPLUMBER_OK = True
except ImportError:
    PDFPLUMBER_OK = False

try:
    import PyPDF2
    PYPDF2_OK = True
except ImportError:
    PYPDF2_OK = False

try:
    from Levenshtein import ratio as lev_ratio
    LEV_OK = True
except ImportError:
    LEV_OK = False
    print("⚠️  python-Levenshtein no disponible; usando overlap de palabras")

# ── Importar OCR con cache del paquete ───────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from spharagusnet.ocr import extraer_bloques_cached

# ── Configuración ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CSV_BCN = BASE_DIR / "data" / "planes_origen_modificaciones_bcn.csv"
DATASET_DIR = BASE_DIR / "data" / "dataset_entrenamiento"
IMAGENES_DIR = DATASET_DIR / "imagenes"
TEXTOS_DIR = DATASET_DIR / "textos"
METADATA_DIR = DATASET_DIR / "metadata"

TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.0


# ── Utilidades ───────────────────────────────────────────────────────────────

def crear_directorios():
    for d in (DATASET_DIR, IMAGENES_DIR, TEXTOS_DIR, METADATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def generar_hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def descargar_archivo(url: str, output_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ❌ Error descargando {url}: {e}")
        return False


# ── Limpieza de texto BCN ───────────────────────────────────────────────────

def limpiar_texto_bcn(texto: str) -> str:
    """Remueve encabezados/pies estandarizados de documentos BCN/LeyChile."""
    if not texto:
        return ""

    lineas = texto.split("\n")
    lineas_limpias: List[str] = []

    patron_decreto = re.compile(r"^(Decreto|Resolucion|Resolución)\s+\d+", re.I)
    patron_decreto_num = re.compile(r"^(Decreto|Resolución)\s+\d+$", re.I)
    patron_minvu = re.compile(r"^MINISTERIO DE VIVIENDA Y URBANISMO$", re.I)
    patron_muni = re.compile(r"^MUNICIPALIDAD DE", re.I)
    patron_gob = re.compile(r"^GOBIERNO REGIONAL", re.I)
    patron_fp = re.compile(r"^(Fecha\s+)?Publicación:", re.I)
    patron_fprom = re.compile(r"Promulgación:", re.I)
    patron_tv = re.compile(r"^(Tipo\s+)?Versión:", re.I)
    patron_url = re.compile(r"^Url\s+Corta:", re.I)
    patron_bcn = re.compile(r"https://bcn\.cl/", re.I)

    patron_biblio = re.compile(r"Biblioteca del Congreso Nacional de Chile", re.I)
    patron_ley = re.compile(r"www\.leychile\.cl", re.I)
    patron_doc = re.compile(r"documento generado el", re.I)
    patron_pag = re.compile(r"página\s+\d+\s+de\s+\d+", re.I)
    patron_inicio = re.compile(r"(Núm\.|Num\.|Sección|Seccion|Vistos?:)", re.I)

    en_encabezado = True
    lineas_enc = 0
    max_enc = 12
    titulo_principal = None

    for linea in lineas:
        ls = linea.strip()
        if not ls:
            if not en_encabezado:
                lineas_limpias.append("")
            continue

        if en_encabezado and lineas_enc < max_enc:
            es_enc = False
            if (patron_decreto.match(ls) or patron_decreto_num.match(ls)
                    or patron_minvu.match(ls) or patron_muni.match(ls)
                    or patron_gob.match(ls) or patron_fp.search(ls)
                    or patron_fprom.search(ls) or patron_tv.search(ls)
                    or patron_url.search(ls) or patron_bcn.search(ls)):
                es_enc = True
                lineas_enc += 1

            if (not es_enc and 2 <= lineas_enc <= 4 and ls.isupper()
                    and 30 < len(ls) < 150
                    and not patron_inicio.search(ls)):
                if titulo_principal is None:
                    titulo_principal = ls
                es_enc = True
                lineas_enc += 1

            if (not es_enc and titulo_principal
                    and ls.upper() == titulo_principal.upper()
                    and lineas_enc >= 7):
                es_enc = True
                lineas_enc += 1

            if es_enc:
                continue

        if en_encabezado:
            if patron_inicio.search(ls):
                en_encabezado = False
            elif (len(ls) > 40 and lineas_enc >= 6
                  and not patron_fp.search(ls) and not patron_tv.search(ls)
                  and not patron_url.search(ls) and not patron_bcn.search(ls)
                  and not (titulo_principal and ls.upper() == titulo_principal.upper())):
                en_encabezado = False

        if (patron_biblio.search(ls)
                or (patron_ley.search(ls)
                    and (patron_doc.search(ls) or patron_pag.search(ls)))):
            continue

        if not en_encabezado:
            lineas_limpias.append(linea)

    resultado = "\n".join(lineas_limpias)
    resultado = re.sub(r"\n\s*\n\s*\n+", "\n\n", resultado)
    return resultado.strip()


# ── Extracción de texto de referencia ────────────────────────────────────────

def extraer_texto_leychile(pdf_path: Path) -> str:
    texto = ""
    if PDFPLUMBER_OK:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        texto += t + "\n\n"
            if texto.strip():
                return limpiar_texto_bcn(texto.strip())
        except Exception:
            pass

    if PYPDF2_OK:
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    t = p.extract_text()
                    if t:
                        texto += t + "\n\n"
            return limpiar_texto_bcn(texto.strip())
        except Exception:
            pass

    return ""


# ── Conversión PDF MINVU → imágenes ─────────────────────────────────────────

def convertir_pdf_minvu_a_imagenes(pdf_path: Path, output_dir: Path) -> List[Path]:
    if not PDF2IMAGE_OK:
        return []

    imagenes: List[Path] = []
    page_num = 1
    try:
        while True:
            pages = convert_from_path(
                str(pdf_path), dpi=200,
                first_page=page_num, last_page=page_num,
            )
            if not pages:
                break
            img_path = output_dir / f"{pdf_path.stem}_page_{page_num:03d}.png"
            pages[0].save(img_path, "PNG")
            imagenes.append(img_path)
            del pages
            gc.collect()
            page_num += 1
    except Exception as e:
        if not imagenes:
            print(f"  ❌ Error convirtiendo PDF: {e}")
    return imagenes


# ── Etiquetado de bloques (mejorado) ────────────────────────────────────────

def _generar_ngramas(texto: str, n: int = 3) -> List[str]:
    """Genera n-gramas de palabras a partir de un texto."""
    palabras = texto.lower().split()
    if len(palabras) < n:
        return [" ".join(palabras)] if palabras else []
    return [" ".join(palabras[i:i + n]) for i in range(len(palabras) - n + 1)]


def etiquetar_bloques(
    bloques_ocr: List[Dict],
    texto_referencia: str,
) -> List[Dict]:
    """
    Etiqueta cada bloque como relevante (True) o irrelevante (False).

    Mejora respecto a la versión anterior:
    - Usa similitud Levenshtein sobre trigramas del texto de referencia
      en vez de overlap de palabras sueltas (que matchea "de", "la", etc.).
    - Las heurísticas de filtrado siguen aplicándose primero.
    """
    texto_ref_lower = texto_referencia.lower()

    # Pre-computar trigramas de referencia para matching rápido
    if LEV_OK:
        ngramas_ref = set(_generar_ngramas(texto_ref_lower, 3))
    else:
        palabras_ref = set(texto_ref_lower.split())

    bloques_marcados: List[Dict] = []
    for bloque in bloques_ocr:
        texto_bloque = bloque["texto"].lower().strip()

        # Heurísticas de filtrado rápido (basura obvia)
        es_irrelevante = False
        if any(p in texto_bloque for p in (
            "diario oficial", "página", "pág.", "pag.",
            "biblioteca del congreso", "bcn", "ley chile",
            "140 años", "fecha publicación", "fecha promulgación",
        )):
            es_irrelevante = True

        if len(texto_bloque) <= 3 and texto_bloque.isdigit():
            es_irrelevante = True

        if ("http" in texto_bloque or "www" in texto_bloque) and len(texto_bloque) < 50:
            es_irrelevante = True

        # Determinar relevancia
        if es_irrelevante:
            es_relevante = False
        elif LEV_OK:
            # Generar trigramas del bloque y ver cuántos matchean la referencia
            ngramas_bloque = _generar_ngramas(texto_bloque, 3)
            if not ngramas_bloque:
                # Bloque muy corto: matching directo con Levenshtein
                es_relevante = lev_ratio(texto_bloque, texto_ref_lower) > 0.3
            else:
                # Contar trigramas que tienen un match cercano en la referencia
                matches = 0
                for ng in ngramas_bloque:
                    # Buscar si el trigrama aparece como substring
                    if ng in texto_ref_lower:
                        matches += 1
                    else:
                        # Fallback: Levenshtein contra los trigramas de referencia
                        best = max(
                            (lev_ratio(ng, ref_ng) for ref_ng in ngramas_ref),
                            default=0.0,
                        )
                        if best > 0.7:
                            matches += 1
                es_relevante = matches > 0 and (matches / len(ngramas_bloque)) > 0.3
        else:
            # Fallback sin Levenshtein: overlap de palabras (original)
            palabras_bloque = set(texto_bloque.split())
            # Filtrar stopwords para reducir ruido
            stopwords = {"de", "la", "el", "en", "y", "a", "los", "las", "del",
                         "que", "por", "con", "un", "una", "se", "al", "es", "lo",
                         "para", "su", "no", "como", "más", "o", "pero"}
            palabras_sig = palabras_bloque - stopwords
            if not palabras_sig:
                palabras_sig = palabras_bloque  # si solo quedan stopwords, usar todas
            overlap = len(palabras_sig & palabras_ref)
            es_relevante = overlap > 0

        bloque["relevante"] = es_relevante
        bloques_marcados.append(bloque)

    return bloques_marcados


# ── Procesamiento de un registro ─────────────────────────────────────────────

def procesar_registro(row: Dict, idx: int) -> Optional[Dict]:
    """
    Procesa un registro del CSV.

    Es **incremental**: si los PDFs, imágenes, texto y metadata ya existen,
    solo genera los caches OCR faltantes.
    """
    url_leychile = row.get("url", "").strip()
    url_minvu = row.get("url_minvu", "").strip()
    comuna = row.get("comuna", "").strip()
    titulo = row.get("titulo", "").strip()

    if not url_leychile or not url_minvu:
        return None
    if not url_leychile.startswith("http") or not url_minvu.startswith("http"):
        return None

    hash_lc = generar_hash_url(url_leychile)
    hash_mv = generar_hash_url(url_minvu)
    registro_id = f"{hash_lc}_{hash_mv}"

    print(f"\n  [{idx}] {comuna} — {titulo[:60]}")

    # ── PDF LeyChile ─────────────────────────────────────────────────────
    pdf_lc = DATASET_DIR / "pdfs_leychile" / f"{hash_lc}.pdf"
    pdf_lc.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_lc.exists():
        print(f"    ⬇️  Descargando PDF LeyChile...")
        if not descargar_archivo(url_leychile, pdf_lc):
            return None
        time.sleep(DELAY_BETWEEN_REQUESTS)
    else:
        print(f"    ✓ PDF LeyChile ya existe")

    # ── Texto de referencia ──────────────────────────────────────────────
    texto_path = TEXTOS_DIR / f"{hash_lc}.txt"
    if texto_path.exists():
        with open(texto_path, "r", encoding="utf-8") as f:
            texto_lc = f.read()
        print(f"    ✓ Texto referencia ya existe ({len(texto_lc)} chars)")
    else:
        print(f"    📝 Extrayendo texto de LeyChile...")
        texto_lc = extraer_texto_leychile(pdf_lc)
        if not texto_lc or len(texto_lc) < 100:
            print(f"    ⚠️  Texto extraído muy corto, omitiendo")
            return None
        with open(texto_path, "w", encoding="utf-8") as f:
            f.write(texto_lc)

    # ── PDF MINVU ────────────────────────────────────────────────────────
    pdf_mv = DATASET_DIR / "pdfs_minvu" / f"{hash_mv}.pdf"
    pdf_mv.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_mv.exists():
        print(f"    ⬇️  Descargando PDF MINVU...")
        if not descargar_archivo(url_minvu, pdf_mv):
            return None
        time.sleep(DELAY_BETWEEN_REQUESTS)
    else:
        print(f"    ✓ PDF MINVU ya existe")

    # ── Imágenes ─────────────────────────────────────────────────────────
    existing_imgs = sorted(IMAGENES_DIR.glob(f"{hash_mv}_page_*.png"))
    if existing_imgs:
        imagenes = existing_imgs
        print(f"    ✓ {len(imagenes)} imágenes ya existen")
    else:
        print(f"    🖼️  Convirtiendo PDF MINVU a imágenes...")
        imagenes = convertir_pdf_minvu_a_imagenes(pdf_mv, IMAGENES_DIR)
        if not imagenes:
            print(f"    ⚠️  No se generaron imágenes, omitiendo")
            return None

    # ── Cache OCR + etiquetado (lo nuevo) ────────────────────────────────
    imgs_con_cache = 0
    imgs_nuevas_ocr = 0
    for img_path in imagenes:
        ocr_cache = img_path.with_suffix(".ocr.json")
        label_cache = img_path.with_suffix(".labels.json")

        # OCR cache
        if not ocr_cache.exists():
            bloques = extraer_bloques_cached(img_path)
            imgs_nuevas_ocr += 1
        else:
            imgs_con_cache += 1

        # Labels cache (re-generar si no existe o si el etiquetado cambió)
        if not label_cache.exists():
            # Cargar bloques (del cache que acabamos de crear o ya existía)
            with open(ocr_cache, "r", encoding="utf-8") as f:
                bloques = json.load(f)
            bloques_etiquetados = etiquetar_bloques(bloques, texto_lc)
            with open(label_cache, "w", encoding="utf-8") as f:
                json.dump(bloques_etiquetados, f, ensure_ascii=False)

    if imgs_nuevas_ocr:
        print(f"    🔍 OCR ejecutado en {imgs_nuevas_ocr} imágenes nuevas")
    if imgs_con_cache:
        print(f"    ✓ {imgs_con_cache} imágenes ya tenían cache OCR")

    # ── Metadata ─────────────────────────────────────────────────────────
    metadata = {
        "id": registro_id,
        "comuna": comuna,
        "titulo": titulo,
        "url_leychile": url_leychile,
        "url_minvu": url_minvu,
        "hash_leychile": hash_lc,
        "hash_minvu": hash_mv,
        "texto_path": str(texto_path.relative_to(DATASET_DIR)),
        "imagenes": [str(img.relative_to(DATASET_DIR)) for img in imagenes],
        "num_paginas": len(imagenes),
        "longitud_texto": len(texto_lc),
    }

    metadata_path = METADATA_DIR / f"{registro_id}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"    ✅ Par completo: {len(imagenes)} páginas, {len(texto_lc)} chars")
    return metadata


# ── Limpieza de huérfanos ────────────────────────────────────────────────────

def limpiar_huerfanos(metadatas: List[Dict]) -> None:
    hashes_lc = set()
    hashes_mv = set()
    imgs_validas = set()

    for m in metadatas:
        hashes_lc.add(m["hash_leychile"])
        hashes_mv.add(m["hash_minvu"])
        for img in m["imagenes"]:
            imgs_validas.add(Path(img).name)

    eliminados = 0
    for d, validos, ext in [
        (DATASET_DIR / "pdfs_leychile", hashes_lc, "*.pdf"),
        (DATASET_DIR / "pdfs_minvu", hashes_mv, "*.pdf"),
        (TEXTOS_DIR, hashes_lc, "*.txt"),
    ]:
        if d.exists():
            for f in d.glob(ext):
                if f.stem not in validos:
                    f.unlink()
                    eliminados += 1

    if IMAGENES_DIR.exists():
        for f in IMAGENES_DIR.glob("*.png"):
            if f.name not in imgs_validas:
                f.unlink()
                eliminados += 1
                # Limpiar caches asociados
                for suffix in (".ocr.json", ".labels.json"):
                    cache = f.with_suffix(suffix)
                    if cache.exists():
                        cache.unlink()

    if METADATA_DIR.exists():
        ids_validos = {m["id"] for m in metadatas}
        for f in METADATA_DIR.glob("*.json"):
            if f.stem not in ids_validos:
                f.unlink()
                eliminados += 1

    print(f"{'🧹 ' + str(eliminados) + ' huérfanos eliminados' if eliminados else '✓ Sin huérfanos'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepara dataset de entrenamiento")
    parser.add_argument("--max-pares", type=int, default=None,
                        help="Máximo de pares a procesar (default: todos)")
    args = parser.parse_args()

    print("🚀 Preparación de Dataset (incremental)")
    print("=" * 80)

    crear_directorios()

    if not CSV_BCN.exists():
        print(f"❌ No se encontró {CSV_BCN}")
        return

    df = pd.read_csv(CSV_BCN, encoding="utf-8")
    print(f"✓ CSV: {len(df)} registros")

    df_valido = df[
        df["url"].notna() & (df["url"].astype(str).str.strip() != "")
        & df["url_minvu"].notna() & (df["url_minvu"].astype(str).str.strip() != "")
    ]
    print(f"✓ Con ambas URLs: {len(df_valido)}")

    max_pares = args.max_pares
    if max_pares is not None and max_pares < len(df_valido):
        df_valido = df_valido.head(max_pares)
        print(f"✓ Limitado a {max_pares} pares")

    metadatas: List[Dict] = []
    procesados = 0
    fallidos = 0

    for idx, row in df_valido.iterrows():
        try:
            m = procesar_registro(row.to_dict(), idx)
            if m:
                metadatas.append(m)
                procesados += 1
            else:
                fallidos += 1
        except Exception as e:
            fallidos += 1
            print(f"  ❌ Error en registro {idx}: {e}")
            import traceback
            traceback.print_exc()

    # Guardar índice
    indice_path = DATASET_DIR / "indice.json"
    with open(indice_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_registros": len(metadatas),
            "fecha_creacion": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "registros": metadatas,
        }, f, ensure_ascii=False, indent=2)

    limpiar_huerfanos(metadatas)

    print(f"\n✅ Completado: {procesados} pares | {fallidos} fallidos")
    print(f"   Dataset: {DATASET_DIR}")


if __name__ == "__main__":
    main()
