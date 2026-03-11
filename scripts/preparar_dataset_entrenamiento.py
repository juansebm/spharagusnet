#!/usr/bin/env python3
"""
Script para preparar dataset de entrenamiento para extracción de texto
desde documentos escaneados de MINVU usando PDFs limpios de LeyChile como referencia.

El dataset consistirá en:
- Imágenes de PDFs escaneados de MINVU (url_minvu)
- Texto limpio extraído de PDFs de LeyChile (url) como ground truth
"""

import csv
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import pandas as pd

# Importaciones para procesamiento de PDFs e imágenes
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_DISPONIBLE = True
except ImportError:
    PDF2IMAGE_DISPONIBLE = False
    print("⚠️  pdf2image no disponible. Instala con: pip install pdf2image")

try:
    import PyPDF2
    PYPDF2_DISPONIBLE = True
except ImportError:
    PYPDF2_DISPONIBLE = False

try:
    import pdfplumber
    PDFPLUMBER_DISPONIBLE = True
except ImportError:
    PDFPLUMBER_DISPONIBLE = False

from PIL import Image
import hashlib

# Configuración
BASE_DIR = Path(__file__).parent.parent
CSV_BCN = BASE_DIR / "data" / "planes_origen_modificaciones_bcn.csv"
DATASET_DIR = BASE_DIR / "data" / "dataset_entrenamiento"
IMAGENES_DIR = DATASET_DIR / "imagenes"
TEXTOS_DIR = DATASET_DIR / "textos"
METADATA_DIR = DATASET_DIR / "metadata"

# Configuración de descarga
TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.0

def crear_directorios():
    """Crea los directorios necesarios para el dataset."""
    for dir_path in [DATASET_DIR, IMAGENES_DIR, TEXTOS_DIR, METADATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def descargar_archivo(url: str, output_path: Path) -> bool:
    """Descarga un archivo desde una URL."""
    try:
        response = requests.get(url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"  ❌ Error descargando {url}: {e}")
        return False

def extraer_texto_leychile(pdf_path: Path) -> str:
    """
    Extrae texto limpio de un PDF de LeyChile.
    Estos PDFs tienen formato estandarizado y texto extraíble.
    """
    texto = ""
    
    # Intentar con pdfplumber primero (mejor para texto estructurado)
    if PDFPLUMBER_DISPONIBLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for pagina in pdf.pages:
                    t = pagina.extract_text()
                    if t:
                        texto += t + "\n\n"
            if texto.strip():
                return texto.strip()
        except Exception:
            pass
    
    # Fallback a PyPDF2
    if PYPDF2_DISPONIBLE:
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for pagina in pdf.pages:
                    t = pagina.extract_text()
                    if t:
                        texto += t + "\n\n"
            return texto.strip()
        except Exception:
            pass
    
    return ""

def convertir_pdf_minvu_a_imagenes(pdf_path: Path, output_dir: Path) -> List[Path]:
    """
    Convierte un PDF escaneado de MINVU a imágenes (una por página).
    Devuelve lista de rutas a las imágenes generadas.
    """
    if not PDF2IMAGE_DISPONIBLE:
        print("  ⚠️  pdf2image no disponible")
        return []
    
    imagenes = []
    try:
        # Convertir PDF a imágenes con alta resolución
        pages = convert_from_path(
            str(pdf_path),
            dpi=300,  # Alta resolución para mejor calidad
            first_page=1,
            last_page=None
        )
        
        pdf_name = pdf_path.stem
        for i, page in enumerate(pages, 1):
            img_path = output_dir / f"{pdf_name}_page_{i:03d}.png"
            page.save(img_path, "PNG")
            imagenes.append(img_path)
        
        return imagenes
    except Exception as e:
        print(f"  ❌ Error convirtiendo PDF a imágenes: {e}")
        return []

def generar_hash_url(url: str) -> str:
    """Genera un hash corto de una URL para usar como nombre de archivo."""
    return hashlib.md5(url.encode()).hexdigest()[:12]

def procesar_registro(row: Dict, idx: int) -> Optional[Dict]:
    """
    Procesa un registro del CSV:
    1. Descarga PDF de LeyChile (url) y extrae texto limpio
    2. Descarga PDF de MINVU (url_minvu) y lo convierte a imágenes
    3. Guarda ambos en el dataset
    """
    url_leychile = row.get("url", "").strip()
    url_minvu = row.get("url_minvu", "").strip()
    comuna = row.get("comuna", "").strip()
    titulo = row.get("titulo", "").strip()
    
    # Validar que ambas URLs existan
    if not url_leychile or not url_minvu:
        print(f"  [{idx}] ⚠️  Faltan URLs, omitiendo")
        return None
    
    if not url_leychile.startswith("http") or not url_minvu.startswith("http"):
        print(f"  [{idx}] ⚠️  URLs inválidas, omitiendo")
        return None
    
    print(f"\n  [{idx}] {comuna} - {titulo[:60]}")
    
    # Generar nombres de archivo basados en hash de URLs
    hash_leychile = generar_hash_url(url_leychile)
    hash_minvu = generar_hash_url(url_minvu)
    
    # Descargar PDF de LeyChile
    pdf_leychile_path = DATASET_DIR / "pdfs_leychile" / f"{hash_leychile}.pdf"
    pdf_leychile_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not pdf_leychile_path.exists():
        print(f"    ⬇️  Descargando PDF LeyChile...")
        if not descargar_archivo(url_leychile, pdf_leychile_path):
            return None
        time.sleep(DELAY_BETWEEN_REQUESTS)
    else:
        print(f"    ✓ PDF LeyChile ya existe")
    
    # Extraer texto de LeyChile
    print(f"    📝 Extrayendo texto de LeyChile...")
    texto_leychile = extraer_texto_leychile(pdf_leychile_path)
    
    if not texto_leychile or len(texto_leychile) < 100:
        print(f"    ⚠️  Texto extraído muy corto o vacío, omitiendo")
        return None
    
    # Guardar texto de LeyChile
    texto_path = TEXTOS_DIR / f"{hash_leychile}.txt"
    with open(texto_path, 'w', encoding='utf-8') as f:
        f.write(texto_leychile)
    
    # Descargar PDF de MINVU
    pdf_minvu_path = DATASET_DIR / "pdfs_minvu" / f"{hash_minvu}.pdf"
    pdf_minvu_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not pdf_minvu_path.exists():
        print(f"    ⬇️  Descargando PDF MINVU...")
        if not descargar_archivo(url_minvu, pdf_minvu_path):
            return None
        time.sleep(DELAY_BETWEEN_REQUESTS)
    else:
        print(f"    ✓ PDF MINVU ya existe")
    
    # Convertir PDF de MINVU a imágenes
    print(f"    🖼️  Convirtiendo PDF MINVU a imágenes...")
    imagenes = convertir_pdf_minvu_a_imagenes(pdf_minvu_path, IMAGENES_DIR)
    
    if not imagenes:
        print(f"    ⚠️  No se pudieron generar imágenes, omitiendo")
        return None
    
    # Crear metadata del registro
    metadata = {
        "id": f"{hash_leychile}_{hash_minvu}",
        "comuna": comuna,
        "titulo": titulo,
        "url_leychile": url_leychile,
        "url_minvu": url_minvu,
        "hash_leychile": hash_leychile,
        "hash_minvu": hash_minvu,
        "texto_path": str(texto_path.relative_to(DATASET_DIR)),
        "imagenes": [str(img.relative_to(DATASET_DIR)) for img in imagenes],
        "num_paginas": len(imagenes),
        "longitud_texto": len(texto_leychile)
    }
    
    # Guardar metadata
    metadata_path = METADATA_DIR / f"{hash_leychile}_{hash_minvu}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"    ✅ Procesado: {len(imagenes)} páginas, {len(texto_leychile)} caracteres de texto")
    
    return metadata

def main():
    """Función principal."""
    print("🚀 Preparación de Dataset para Entrenamiento")
    print("=" * 80)
    
    # Crear directorios
    crear_directorios()
    
    # Leer CSV de BCN
    if not CSV_BCN.exists():
        print(f"❌ No se encontró {CSV_BCN}")
        return
    
    df = pd.read_csv(CSV_BCN, encoding="utf-8")
    print(f"✓ CSV cargado: {len(df)} registros")
    
    # Filtrar solo registros con ambas URLs
    df_valido = df[
        df["url"].notna() & (df["url"].astype(str).str.strip() != "") &
        df["url_minvu"].notna() & (df["url_minvu"].astype(str).str.strip() != "")
    ]
    print(f"✓ Registros con ambas URLs: {len(df_valido)}")
    
    if len(df_valido) == 0:
        print("⚠️  No hay registros válidos para procesar")
        return
    
    # Procesar cada registro
    metadatas = []
    procesados = 0
    fallidos = 0
    
    for idx, row in df_valido.iterrows():
        try:
            metadata = procesar_registro(row.to_dict(), idx)
            if metadata:
                metadatas.append(metadata)
                procesados += 1
            else:
                fallidos += 1
        except Exception as e:
            print(f"  ❌ Error procesando registro {idx}: {e}")
            fallidos += 1
            import traceback
            traceback.print_exc()
    
    # Guardar índice completo del dataset
    indice_path = DATASET_DIR / "indice.json"
    with open(indice_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_registros": len(metadatas),
            "fecha_creacion": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "registros": metadatas
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Proceso completado")
    print(f"   Procesados exitosamente: {procesados}")
    print(f"   Fallidos: {fallidos}")
    print(f"   Dataset guardado en: {DATASET_DIR}")
    print(f"   Índice: {indice_path}")

if __name__ == "__main__":
    main()
