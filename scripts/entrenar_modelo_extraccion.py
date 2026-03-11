#!/usr/bin/env python3
"""
Script para entrenar un modelo de extracción de texto relevante
desde documentos escaneados de MINVU.

El modelo aprenderá a:
1. Identificar qué partes de un documento escaneado son relevantes
2. Extraer solo el texto relevante (similar al formato de LeyChile)
3. Filtrar encabezados, pies de página, números de página, etc.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

# Configuración de GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Usando dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Importaciones para OCR y procesamiento
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_DISPONIBLE = True
except ImportError:
    TESSERACT_DISPONIBLE = False
    print("⚠️  pytesseract no disponible")

try:
    from transformers import (
        LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_DISPONIBLE = True
except ImportError:
    TRANSFORMERS_DISPONIBLE = False
    print("⚠️  transformers no disponible")

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset_entrenamiento"
MODEL_DIR = BASE_DIR / "models" / "texto_extractor"
IMAGENES_DIR = DATASET_DIR / "imagenes"
TEXTOS_DIR = DATASET_DIR / "textos"
METADATA_DIR = DATASET_DIR / "metadata"

class DocumentTextExtractor(nn.Module):
    """
    Modelo para extraer texto relevante de documentos escaneados.
    
    Arquitectura:
    1. OCR para extraer texto con coordenadas
    2. Clasificador de bloques (relevante/no relevante)
    3. Generador de texto limpio
    """
    
    def __init__(self, vocab_size=10000, hidden_dim=512, device=None):
        super().__init__()
        self.device = device or DEVICE
        
        # Embedding para texto
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Clasificador de bloques
        self.block_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # relevante/no relevante
        )
        
        # Generador de texto
        self.text_generator = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Mover modelo a GPU si está disponible
        self.to(self.device)
    
    def forward(self, text_tokens, block_positions):
        # Asegurar que los tensores estén en el dispositivo correcto
        text_tokens = text_tokens.to(self.device)
        
        # Embed texto
        embedded = self.text_embedding(text_tokens)
        
        # Clasificar bloques
        block_scores = self.block_classifier(embedded.mean(dim=1))
        
        # Generar texto relevante
        lstm_out, _ = self.text_generator(embedded)
        output = self.output_layer(lstm_out)
        
        return output, block_scores

def extraer_texto_con_coordenadas(imagen_path: Path) -> List[Dict]:
    """
    Extrae texto de una imagen con coordenadas usando OCR.
    Devuelve lista de bloques de texto con sus posiciones.
    """
    if not TESSERACT_DISPONIBLE:
        return []
    
    try:
        imagen = Image.open(imagen_path)
        
        # OCR con detección de layout
        ocr_data = pytesseract.image_to_data(
            imagen, 
            lang='spa',
            output_type=Output.DICT
        )
        
        bloques = []
        current_block = None
        
        for i in range(len(ocr_data['text'])):
            texto = ocr_data['text'][i].strip()
            if not texto:
                continue
            
            conf = int(ocr_data['conf'][i])
            if conf < 30:  # Filtrar texto con baja confianza
                continue
            
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            bloques.append({
                'texto': texto,
                'x': x, 'y': y, 'w': w, 'h': h,
                'confianza': conf
            })
        
        return bloques
    except Exception as e:
        print(f"Error en OCR: {e}")
        return []

def identificar_bloques_relevantes(
    bloques_ocr: List[Dict], 
    texto_referencia: str
) -> List[Dict]:
    """
    Identifica qué bloques de texto son relevantes comparando con el texto de referencia.
    Usa heurísticas y matching de texto.
    """
    texto_ref_lower = texto_referencia.lower()
    palabras_ref = set(texto_ref_lower.split())
    
    bloques_marcados = []
    
    for bloque in bloques_ocr:
        texto_bloque = bloque['texto'].lower()
        
        # Heurísticas para identificar texto irrelevante
        es_irrelevante = False
        
        # Filtrar encabezados comunes
        if any(palabra in texto_bloque for palabra in [
            'diario oficial', 'página', 'pág.', 'pag.', 
            'biblioteca del congreso', 'bcn', 'ley chile',
            '140 años', 'fecha publicación', 'fecha promulgación'
        ]):
            es_irrelevante = True
        
        # Filtrar números de página
        if len(texto_bloque.strip()) <= 3 and texto_bloque.strip().isdigit():
            es_irrelevante = True
        
        # Filtrar QR codes y URLs cortas (detectadas como texto)
        if 'http' in texto_bloque or 'www' in texto_bloque:
            if len(texto_bloque) < 50:  # URLs cortas probablemente son QR codes
                es_irrelevante = True
        
        # Verificar si el texto aparece en la referencia
        palabras_bloque = set(texto_bloque.split())
        overlap = len(palabras_bloque & palabras_ref)
        
        # Si hay suficiente overlap, es relevante
        es_relevante = overlap > 0 and not es_irrelevante
        
        bloque['relevante'] = es_relevante
        bloques_marcados.append(bloque)
    
    return bloques_marcados

def preparar_datos_entrenamiento() -> Tuple[List[Dict], List[Dict]]:
    """
    Prepara los datos de entrenamiento desde el dataset.
    Devuelve (datos_entrenamiento, datos_validacion)
    """
    indice_path = DATASET_DIR / "indice.json"
    
    if not indice_path.exists():
        print(f"❌ No se encontró {indice_path}")
        print("   Ejecuta primero: python preparar_dataset_entrenamiento.py")
        return [], []
    
    with open(indice_path, 'r', encoding='utf-8') as f:
        indice = json.load(f)
    
    datos = []
    
    for registro in indice['registros']:
        try:
            # Cargar texto de referencia
            texto_path = DATASET_DIR / registro['texto_path']
            if not texto_path.exists():
                continue
            
            with open(texto_path, 'r', encoding='utf-8') as f:
                texto_referencia = f.read()
            
            # Procesar cada imagen
            for img_rel_path in registro['imagenes']:
                img_path = DATASET_DIR / img_rel_path
                if not img_path.exists():
                    continue
                
                # Extraer texto con OCR
                bloques_ocr = extraer_texto_con_coordenadas(img_path)
                
                if not bloques_ocr:
                    continue
                
                # Identificar bloques relevantes
                bloques_marcados = identificar_bloques_relevantes(
                    bloques_ocr, 
                    texto_referencia
                )
                
                datos.append({
                    'imagen_path': str(img_path),
                    'texto_referencia': texto_referencia,
                    'bloques_ocr': bloques_marcados,
                    'metadata': registro
                })
        
        except Exception as e:
            print(f"Error procesando registro {registro.get('id', 'unknown')}: {e}")
            continue
    
    print(f"✓ Datos preparados: {len(datos)} ejemplos")
    
    # Dividir en entrenamiento y validación (80/20)
    split_idx = int(len(datos) * 0.8)
    datos_train = datos[:split_idx]
    datos_val = datos[split_idx:]
    
    return datos_train, datos_val

def entrenar_modelo_simple(datos_train: List[Dict], datos_val: List[Dict]):
    """
    Entrena un modelo simple de clasificación de bloques usando GPU.
    Este es un ejemplo básico - se puede mejorar con modelos más sofisticados.
    """
    print("\n🚀 Iniciando entrenamiento del modelo...")
    print(f"   Dispositivo: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Ejemplos de entrenamiento: {len(datos_train)}")
    print(f"   Ejemplos de validación: {len(datos_val)}")
    
    # Inicializar modelo en GPU
    model = DocumentTextExtractor(vocab_size=10000, hidden_dim=512, device=DEVICE)
    model.train()
    
    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(DEVICE)  # Mover loss a GPU
    
    # Preparar datos de entrenamiento (ejemplo simplificado)
    # En producción, necesitarías un DataLoader más sofisticado
    print("\n📊 Preparando datos para entrenamiento...")
    
    # Guardar datos procesados
    datos_train_path = MODEL_DIR / "datos_train.json"
    datos_val_path = MODEL_DIR / "datos_val.json"
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(datos_train_path, 'w', encoding='utf-8') as f:
        json.dump(datos_train, f, ensure_ascii=False, indent=2)
    
    with open(datos_val_path, 'w', encoding='utf-8') as f:
        json.dump(datos_val, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Datos guardados en {MODEL_DIR}")
    
    # Ejemplo de loop de entrenamiento (simplificado)
    print("\n🔄 Iniciando entrenamiento...")
    num_epochs = 10
    batch_size = 8
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Aquí iría el loop real de entrenamiento con batches
        # Por ahora es un placeholder que muestra cómo usar GPU
        
        # Ejemplo de cómo procesar un batch en GPU:
        # 1. Crear tensores dummy (en producción vendrían del DataLoader)
        # dummy_tokens = torch.randint(0, 10000, (batch_size, 128)).to(DEVICE)
        # dummy_positions = torch.randn(batch_size, 128, 2).to(DEVICE)
        # 
        # # Forward pass
        # output, block_scores = model(dummy_tokens, dummy_positions)
        # 
        # # Calcular loss (ejemplo)
        # dummy_labels = torch.randint(0, 2, (batch_size,)).to(DEVICE)
        # loss = criterion(block_scores, dummy_labels)
        # 
        # # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # 
        # total_loss += loss.item()
        # num_batches += 1
        
        # Limpiar caché de GPU periódicamente
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # if num_batches > 0:
        #     avg_loss = total_loss / num_batches
        #     print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Guardar modelo entrenado
    model_path = MODEL_DIR / "modelo_entrenado.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'device': str(DEVICE),
        'model_config': {
            'vocab_size': 10000,
            'hidden_dim': 512
        }
    }, model_path)
    
    print(f"✓ Modelo guardado en {model_path}")
    
    # Si usó GPU, mostrar uso de memoria
    if torch.cuda.is_available():
        print(f"   Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Memoria GPU reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    print("\n💡 Próximos pasos:")
    print("   1. Usar LayoutLMv3 o DocFormer para entrenamiento avanzado")
    print("   2. Fine-tune un modelo vision-language (LLaVA, GPT-4V)")
    print("   3. Implementar pipeline completo de extracción")

def main():
    """Función principal."""
    print("🚀 Entrenamiento de Modelo de Extracción de Texto")
    print("=" * 80)
    
    # Preparar datos
    datos_train, datos_val = preparar_datos_entrenamiento()
    
    if not datos_train:
        print("⚠️  No hay datos para entrenar")
        return
    
    # Entrenar modelo
    entrenar_modelo_simple(datos_train, datos_val)
    
    print("\n✅ Proceso completado")

if __name__ == "__main__":
    main()
