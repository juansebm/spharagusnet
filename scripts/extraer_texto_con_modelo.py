#!/usr/bin/env python3
"""
Script para extraer texto relevante de documentos escaneados de MINVU
usando el modelo entrenado.

Una vez entrenado, este script puede procesar nuevos documentos
solo con las imágenes (sin necesidad del texto de referencia).
"""

import json
from pathlib import Path
from typing import List, Dict
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_DISPONIBLE = True
except ImportError:
    TESSERACT_DISPONIBLE = False

# Configuración
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "texto_extractor"

def cargar_modelo():
    """Carga el modelo entrenado."""
    # Por ahora, usa heurísticas simples
    # En producción, cargaría el modelo entrenado
    return None

def extraer_texto_relevante(imagen_path: Path, modelo=None) -> str:
    """
    Extrae texto relevante de una imagen de documento escaneado.
    
    Args:
        imagen_path: Ruta a la imagen del documento
        modelo: Modelo entrenado (opcional, usa heurísticas si None)
    
    Returns:
        Texto relevante extraído
    """
    if not TESSERACT_DISPONIBLE:
        return ""
    
    # Extraer texto con OCR
    imagen = Image.open(imagen_path)
    ocr_data = pytesseract.image_to_data(
        imagen, 
        lang='spa',
        output_type=Output.DICT
    )
    
    bloques_relevantes = []
    
    for i in range(len(ocr_data['text'])):
        texto = ocr_data['text'][i].strip()
        if not texto:
            continue
        
        conf = int(ocr_data['conf'][i])
        if conf < 30:
            continue
        
        # Heurísticas para filtrar texto irrelevante
        texto_lower = texto.lower()
        
        # Filtrar encabezados comunes
        if any(palabra in texto_lower for palabra in [
            'diario oficial', 'página', 'pág.', 'pag.',
            'biblioteca del congreso', 'bcn', 'ley chile',
            '140 años', 'fecha publicación', 'fecha promulgación',
            'url corta', 'qr'
        ]):
            continue
        
        # Filtrar números de página solos
        if len(texto.strip()) <= 3 and texto.strip().isdigit():
            continue
        
        # Filtrar URLs cortas (probablemente QR codes)
        if ('http' in texto_lower or 'www' in texto_lower) and len(texto) < 50:
            continue
        
        # Filtrar texto en posiciones típicas de encabezados/pies
        y = ocr_data['top'][i]
        h_img = imagen.height
        
        # Si está en los primeros 5% o últimos 5% de la imagen, probablemente es encabezado/pie
        if y < h_img * 0.05 or y > h_img * 0.95:
            # Pero solo si es texto corto o tiene palabras de encabezado
            if len(texto) < 50 or any(palabra in texto_lower for palabra in ['diario', 'oficial', 'bcn']):
                continue
        
        bloques_relevantes.append(texto)
    
    # Unir bloques relevantes
    texto_final = ' '.join(bloques_relevantes)
    
    # Limpiar espacios múltiples
    import re
    texto_final = re.sub(r'\s+', ' ', texto_final)
    texto_final = re.sub(r'\n\s*\n', '\n\n', texto_final)
    
    return texto_final.strip()

def procesar_pdf_minvu(pdf_path: Path, output_txt_path: Path = None) -> str:
    """
    Procesa un PDF escaneado de MINVU y extrae texto relevante.
    
    Args:
        pdf_path: Ruta al PDF
        output_txt_path: Ruta donde guardar el texto extraído (opcional)
    
    Returns:
        Texto relevante extraído
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("⚠️  pdf2image no disponible")
        return ""
    
    # Convertir PDF a imágenes
    pages = convert_from_path(str(pdf_path), dpi=300)
    
    textos_paginas = []
    
    for i, page in enumerate(pages, 1):
        # Guardar página temporalmente
        temp_img_path = pdf_path.parent / f"{pdf_path.stem}_temp_page_{i}.png"
        page.save(temp_img_path, "PNG")
        
        # Extraer texto relevante
        texto_pagina = extraer_texto_relevante(temp_img_path)
        if texto_pagina:
            textos_paginas.append(texto_pagina)
        
        # Eliminar imagen temporal
        temp_img_path.unlink()
    
    # Unir textos de todas las páginas
    texto_completo = '\n\n'.join(textos_paginas)
    
    # Guardar si se especifica
    if output_txt_path:
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(texto_completo)
    
    return texto_completo

def main():
    """Función principal para pruebas."""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python extraer_texto_con_modelo.py <ruta_pdf_minvu> [ruta_salida.txt]")
        return
    
    pdf_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not pdf_path.exists():
        print(f"❌ No se encontró: {pdf_path}")
        return
    
    print(f"📄 Procesando: {pdf_path.name}")
    texto = procesar_pdf_minvu(pdf_path, output_path)
    
    print(f"\n✅ Texto extraído ({len(texto)} caracteres)")
    if output_path:
        print(f"   Guardado en: {output_path}")
    else:
        print("\n--- Texto extraído ---")
        print(texto[:500] + "..." if len(texto) > 500 else texto)

if __name__ == "__main__":
    main()
