# Extracción de Texto Relevante desde Documentos Escaneados

Este sistema entrena una red neuronal para extraer texto relevante de documentos escaneados de MINVU, aprendiendo del formato limpio de los PDFs de LeyChile.

## Problema

Los PDFs de MINVU (`url_minvu`) son documentos escaneados que contienen:
- ✅ Texto relevante del instrumento de planificación
- ❌ Encabezados del Diario Oficial
- ❌ Números de página
- ❌ Logos y marcas de agua (BCN, Ley Chile)
- ❌ QR codes y URLs cortas
- ❌ Pies de página

Los PDFs de LeyChile (`url`) tienen texto limpio y estructurado, perfecto como referencia.

## Solución

### Arquitectura Propuesta

1. **Dataset de Entrenamiento**
   - Pares de documentos: PDF escaneado (MINVU) + Texto limpio (LeyChile)
   - Conversión de PDFs MINVU a imágenes (una por página)
   - Extracción de texto con OCR y coordenadas
   - Etiquetado de bloques relevantes/no relevantes

2. **Modelo de Extracción**
   - **Opción A: LayoutLMv3 / DocFormer** (Recomendado)
     - Modelos pre-entrenados para documentos
     - Entienden estructura y layout
     - Fine-tuning para clasificar bloques relevantes
   
   - **Opción B: Vision-Language Model**
     - LLaVA, GPT-4V, o Claude Vision
     - Procesan imágenes directamente
     - Extraen texto estructurado
   
   - **Opción C: Pipeline Híbrido**
     - OCR (Tesseract/EasyOCR) para extraer texto
     - Clasificador de bloques (BERT/RoBERTa) para filtrar
     - Post-procesamiento para limpiar

3. **Inferencia**
   - Solo requiere imágenes de documentos MINVU
   - El modelo extrae automáticamente texto relevante
   - Sin necesidad de texto de referencia

## Uso

### Paso 1: Preparar Dataset

```bash
python preparar_dataset_entrenamiento.py
```

Este script:
- Lee `planes_origen_modificaciones_bcn.csv`
- Descarga PDFs de LeyChile y MINVU
- Extrae texto limpio de LeyChile
- Convierte PDFs MINVU a imágenes
- Crea dataset estructurado en `data/dataset_entrenamiento/`

### Paso 2: Entrenar Modelo

```bash
python entrenar_modelo_extraccion.py
```

Este script:
- Procesa el dataset
- Identifica bloques relevantes usando heurísticas
- Prepara datos para entrenamiento
- (Por implementar: entrenamiento real del modelo)

### Paso 3: Extraer Texto de Nuevos Documentos

```bash
python extraer_texto_con_modelo.py <ruta_pdf_minvu> [salida.txt]
```

## Implementación Recomendada

### Usando LayoutLMv3

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Cargar modelo pre-entrenado
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=2  # relevante/no relevante
)

# Fine-tuning con tu dataset
# ...
```

### Usando Donut (End-to-end)

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Entrenar para generar texto limpio desde imágenes
# ...
```

### Usando GPT-4V o Claude Vision

```python
# API call con imagen
response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extrae solo el texto relevante del documento, excluyendo encabezados, pies de página, números de página y logos."},
            {"type": "image_url", "image_url": {"url": imagen_base64}}
        ]
    }]
)
```

## Estructura del Dataset

```
data/dataset_entrenamiento/
├── imagenes/              # Imágenes de PDFs MINVU (una por página)
├── textos/                # Textos limpios de LeyChile
├── pdfs_leychile/         # PDFs originales de LeyChile
├── pdfs_minvu/            # PDFs originales de MINVU
├── metadata/              # Metadata JSON por registro
└── indice.json            # Índice completo del dataset
```

## Próximos Pasos

1. ✅ Scripts base creados
2. ⏳ Implementar entrenamiento real con LayoutLMv3
3. ⏳ Crear pipeline de inferencia optimizado
4. ⏳ Integrar con `download_documentos_minvu.py`
5. ⏳ Evaluar calidad del texto extraído

## Dependencias

```bash
pip install pdf2image pytesseract pillow
pip install transformers torch
pip install pandas requests
```

## Notas

- El modelo actual usa heurísticas simples para identificar texto relevante
- Para producción, se recomienda usar LayoutLMv3 o Donut fine-tuneado
- El dataset debe tener al menos 50-100 pares de documentos para entrenar efectivamente
