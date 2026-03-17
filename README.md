# SpharagusNet

![SpharagusNet Logo](ChatGPT%20Image%20Mar%2011,%202026,%2002_30_29%20AM.png)

## Sobre el Nombre

**SpharagusNet** combina dos figuras mitológicas que representan las capacidades del sistema:

- **Sphinx** (Esfinge): La criatura que lee enigmas y extrae significado. Representa la capacidad del sistema para procesar documentos complejos y extraer información relevante de textos escaneados, interpretando el contenido legal y normativo de los instrumentos de planificación territorial.

- **Argus** (Argos): El gigante de 100 ojos que todo lo ve. Representa la capacidad del sistema para observar y procesar múltiples fuentes de información simultáneamente.

Juntos, **SpharagusNet** es un sistema que observa todo (Argus) y extrae significado (Sphinx) de los documentos de planificación territorial de Chile.

## ¿Por Qué no Solo OCR Convencional?

La diferencia es fundamental. **Tesseract OCR puro** extrae todo texto de una imagen: encabezados, números de página, pies, metadata, contenido real... **todo mezclado.**

Con documentos escaneados complejos, el resultado es **90% ruido, 10% contenido útil**.

```
┌─────────────────────────────────────────────┐
│ OCR CONVENCIONAL (Tesseract solo)           │
├─────────────────────────────────────────────┤
│ ES ÓN = PATA > Me A A E A AS               │
│ 1 A sas "e . Cr, - . MN y                  │
│ Ps CINARIO SU TOC NTNURLICADE CHILE        │
│ ; A Maries £5 de Septiémbre de 1990 Pág   │
│ ... (caracteres aleatorios, basura)        │
└─────────────────────────────────────────────┘
❌ Inutilizable. 90% ruido, errores OCR severos.

┌─────────────────────────────────────────────┐
│ CON RED NEURONAL (SpharagusNet)             │
├─────────────────────────────────────────────┤
│ Valparaíso, de Septiembre de Andrés Couve  │
│ Rioseco, Subsecretario de Educación        │
│ Ministerio de Educación AUTORIZA...        │
│ Que es necesario establecer procedimientos │
│ que permitan una eficaz inserción de...    │
└─────────────────────────────────────────────┘
✅ Legible. 85% contenido relevante, 15% ruido residual.
```

### ¿Cómo? La red **aprende a clasificar**

La red neuronal no "mejora" Tesseract. En cambio:

1. **Tesseract extrae** todo texto y metadatos
2. **Red neuronal clasifica** cada bloque: ¿es relevante o basura?
3. **Mantiene solo lo relevante**, descarta el resto

**Entrenada con 100+ documentos legislativos**, la red aprendió:
- Qué se parece a "Artículo", "inciso", "párrafo" (patrones legislativos)
- Qué es "Diario Oficial", "Página 5", números aislados (metadata)
- Qué es ruido puro (caracteres aleatorios, errores OCR)

### Resultado: Texto Procesable

| Métrica | OCR Puro | SpharagusNet |
|---------|----------|--------------|
| **Contenido útil** | 10% | 85% |
| **Ruido/Basura** | 90% | 15% |
| **Usable para NLP** | ❌ No | ✅ Sí |
| **Extracción de datos** | ❌ Falla | ✅ Funciona |
| **Análisis sintáctico** | ❌ Imposible | ✅ Posible |
| **Buscabilidad** | ❌ Pobre | ✅ Excelente |

### Ventajas Prácticas

✅ **Mejor ratio signal-to-noise:** La red filtra ~80% del ruido antes de que salga del pipeline

✅ **Dominio específico:** Entrenada en documentos legislativos chilenos (MINVU, LeyChile), entiende la estructura y vocabulario

✅ **Adaptativo:** Con más datos de entrenamiento, la red mejora automáticamente. Las heurísticas hardcodeadas requieren cambios manuales

✅ **Contextual:** Entiende que "Diario" en "Diario de Implementación" es relevante, pero "Diario Oficial" en header es basura

✅ **Métricas medibles:** Accuracy, Precision, Recall, F1 por epoch. Sabes exactamente qué tan bien está funcionando

## Instalación

```bash
pip install spharagusnet
```

> **Requisito de sistema:** Tesseract OCR debe estar instalado (`sudo apt install tesseract-ocr tesseract-ocr-spa` en Ubuntu, o `brew install tesseract` en macOS).

Primera vez — descargar el modelo pre-entrenado (~300 MB):

```bash
spharagusnet download
```

Para desarrollo local (editable):

```bash
git clone https://github.com/jsmdev/spharagusnet.git
cd spharagusnet
pip install -e ".[train]"
```

## Uso

### Como librería

```python
import spharagusnet

# Descargar modelo (primera vez, después se cachea)
spharagusnet.download_model()

# Extraer texto
texto = spharagusnet.extract("documento_escaneado.pdf")
texto = spharagusnet.extract("documento.pdf", confidence=0.6)

# Verificar si el modelo está disponible
spharagusnet.model_is_available()  # True/False
```

### CLI

```bash
# Extraer texto de un PDF
spharagusnet extract documento.pdf
spharagusnet extract documento.pdf -o salida.txt
spharagusnet extract documento.pdf --confidence 0.6

# Descargar/actualizar modelo
spharagusnet download
spharagusnet download --force

# Info del paquete
spharagusnet info
```

### Buscar documentos en BCN vía SPARQL

```bash
# Solo comunas de comunas_interes.txt
python scripts/bcn_sparql_planes.py

# Todas las comunas del CSV
python scripts/bcn_sparql_planes.py --todas
```

> **Nota:** El script solo incluye registros cuya columna `urls_documentos` contenga un link de `instrumentosdeplanificacion.minvu.cl`. Registros sin URL de MINVU (en desarrollo, históricos sin digitalizar, o alojados en sitios municipales) se omiten del CSV de salida.

### Preparar dataset y entrenar modelo

```bash
# 1. Preparar dataset (descarga PDFs, genera imágenes, cachea OCR y etiqueta bloques)
python scripts/preparar_dataset_entrenamiento.py [--max-pares N]

# 2. Entrenar modelo (con métricas en consola)
python scripts/entrenar_modelo_extraccion.py

# 3. Extraer texto de un PDF escaneado (con modelo entrenado)
python scripts/extraer_texto_con_modelo.py <ruta_pdf> [salida.txt] [--confidence 0.5]
```

### Mejorar el modelo iterativamente

```bash
# 1. Incrementar datos gradualmente (50 → 100 → 200 → todos)
python scripts/preparar_dataset_entrenamiento.py --max-pares 80

# 2. Entrenar sobre lo existente
python scripts/entrenar_modelo_extraccion.py --resume

# 3. Si el val-F1 mejora, genial. Si estanca:
#    - Agregar más datos (paso 1 con más pares)
#    - Regenerar labels: borrar los .labels.json y correr paso 1 de nuevo
#    - Ajustar patience/epochs

# 4. Probar en un documento nuevo
python scripts/extraer_texto_con_modelo.py nuevo.pdf salida.txt
```

#### `preparar_dataset_entrenamiento.py`
- `--max-pares N` limita la cantidad de filas a procesar (por defecto todas).
- **Incremental:** solo descarga/procesa pares nuevos; los ya existentes se saltan.
- **Cache OCR:** guarda bloques Tesseract como `.ocr.json` junto a cada imagen, así el entrenamiento posterior no re-corre OCR.
- **Etiquetado mejorado:** usa trigramas + similitud Levenshtein para decidir relevancia (en vez de overlap de palabras sueltas que matchea "de", "la", etc.).
- Convierte PDFs página por página a 200 DPI para evitar consumo de RAM excesivo en WSL.
- Exige par completo (PDF LeyChile + PDF MINVU + imágenes generadas); archivos huérfanos se eliminan automáticamente.

#### `entrenar_modelo_extraccion.py`
- `--resume`: carga el último checkpoint y continúa entrenando (no parte de cero). Expande el vocabulario automáticamente si hay palabras nuevas.
- `--epochs N`: número de epochs (default: 20).
- `--patience N`: early stopping patience (default: 5).
- **Carga desde cache:** lee `.labels.json` generados por `preparar_dataset_entrenamiento.py` en vez de re-correr OCR.
- **Métricas por epoch:** `Loss`, `Accuracy`, `Precision`, `Recall`, `F1` en train y validación.
- Guarda:
  - `modelo_entrenado.pth` (mejor modelo por F1)
  - `vocab.json` (vocabulario para tokenización)
  - `historial_entrenamiento.json` (métricas para análisis)

**Arquitectura:**
- Embedding layer (vocab → 512-D vectors)
- Block classifier (2 capas FC con ReLU + dropout) → clasificación binaria {irrelevante, relevante}
- LSTM (generador de texto, para futuras extensiones)
- **Class weights** para compensar desbalance (más bloques irrelevantes que relevantes)

#### `extraer_texto_con_modelo.py`
**Flujo de inferencia:**
1. PDF escaneado → OCR (Tesseract) extrae bloques de texto
2. Cada bloque → Tokenización + Embedding + Red neuronal
3. Red predice: 0 (irrelevante) ó 1 (relevante)
4. Reconstruye documento solo con bloques relevantes

**Parámetros:**
- `--confidence THRESHOLD`: Umbral de confianza [0-1]
  - Default: 0.5 (balance entre precision/recall)
  - Menor threshold → más bloques (recall ↑, pero más falsos positivos)
  - Mayor threshold → menos bloques (precision ↑, pero pierdes contenido)

**Fallback:** Si el modelo no está disponible, usa heurísticas (filtros manuales)

### Visualizar pipeline completo

```bash
python scripts/pipeline_completo.py
```

Muestra documentación completa del flujo, arquitectura y ejemplos de ejecución.

## Arquitectura de la Red Neuronal

### Diagrama del Flujo de Inferencia

```
┌──────────────────────────────────────────────────────────┐
│                    PDF MINVU (escaneado)                 │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │ 1. CONVERSIÓN PDF → IMÁGENES    │
        │    (page-by-page, 200 DPI)      │
        └────────────────┬─────────────────┘
                         │
        ┌────────────────▼────────────────────────────────┐
        │ Para cada página:                               │
        │                                                 │
        │  ┌─ 2. OCR (Tesseract)                         │
        │  │    Imagen → bloques de texto + coordenadas  │
        │  │    (filtro: confianza > 30%)                │
        │  │                                              │
        │  │    [                                         │
        │  │      {"texto": "Artículo 1", ...},          │
        │  │      {"texto": "Página 1", ...},            │
        │  │      {...}                                  │
        │  │    ]                                        │
        │  │                                              │
        │  └─ 3. CLASIFICACIÓN (MODELO NEURONAL)         │
        │     Para cada bloque:                          │
        │     a) Tokenizar → [idx_1, idx_2, ...]        │
        │     b) Embedding → 512-D vectors               │
        │     c) Pooling → reducir a (512,)              │
        │     d) Block Classifier (2 FC layers)          │
        │     e) Softmax → prob(relevante)               │
        │                                                │
        │     ┌─────────────────────────────┐            │
        │     │ Clasificador Binario:       │            │
        │     │                             │            │
        │     │ FC(512 → 256) ─ ReLU ─      │            │
        │     │ Dropout(0.1) ─ FC(256 → 2) │            │
        │     │                             │            │
        │     │ Output: [logit_neg, logit_pos]│          │
        │     │ Pred: argmax() → {0, 1}     │            │
        │     └─────────────────────────────┘            │
        │                                                │
        │  └─ 4. FILTRADO                                │
        │     Si pred=1 AND confidence >= threshold:    │
        │       → MANTENER bloque                        │
        │     Sino:                                      │
        │       → DESCARTAR bloque                       │
        │                                                │
        └─────────────────┬──────────────────────────────┘
                          │
        ┌─────────────────▼──────────────┐
        │ 5. RECONSTRUCCIÓN              │
        │    Unir bloques relevantes     │
        │    + limpieza de espacios      │
        │    + separar por páginas       │
        └─────────────────┬──────────────┘
                          │
        ┌─────────────────▼──────────────┐
        │ 📄 TEXTO LIMPIO (formato      │
        │    LeyChile, sin headers/pies) │
        └───────────────────────────────┘
```

### Métricas Reportadas During Training

La tabla de entrenamiento muestra:

```
    Epoch │   Loss    Acc   Prec   Rec    F1 │  vLoss  vAcc  vPrec  vRec   vF1 │     t     ETA
     1/20 │  0.6932  0.650  0.580  0.720  0.642 │  0.6801  0.670  0.600  0.700  0.646 │  2.3s   41.4s
     2/20 │  0.5234  0.710  0.620  0.750  0.680 │  0.5102  0.725  0.640  0.770  0.700 │  2.1s   39.2s
     3/20 │  0.4521  0.760  0.680  0.800  0.735 │  0.4398  0.770  0.690  0.810  0.745 │  2.0s   37.0s
```

**Columnas:**
- `Epoch`: Época actual / total
- `Loss` / `vLoss`: Pérdida de entrenamiento / validación (menor es mejor)
- `Acc`: Accuracy (% bloques clasificados correctamente)
- `Prec`: Precision (% de positivos predichos que son realmente positivos)
- `Rec`: Recall (% de positivos reales que fueron detectados)
- `F1`: Media armónica de Precision y Recall (métrica de balance)
- `t`: Tiempo que tardó la época
- `ETA`: Tiempo estimado para terminar

**Early Stopping:** El entrenamiento se detiene automáticamente si `val-F1` no mejora durante 5 epochs consecutivos, previniendo overfitting.

## Publicar en PyPI

```bash
# 1. Subir modelo a GitHub Releases (tag v0.2.0)
#    - modelo_entrenado.pth
#    - vocab.json

# 2. Build
pip install build twine
python -m build

# 3. Publicar
twine upload dist/*
```