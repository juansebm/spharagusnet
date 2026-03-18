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
| **Usable para NLP** | ❌ No | ✅ Sí |
| **Extracción de datos** | ❌ Falla | ✅ Funciona |
| **Análisis sintáctico** | ❌ Difícil | ✅ Posible |
| **Buscabilidad** | ❌ Pobre | ✅ Excelente |

### Ventajas Prácticas

✅ **Mejor ratio signal-to-noise:** La red filtra ~80% del ruido antes de que salga del pipeline

✅ **Dominio específico:** Entrenada en documentos legislativos chilenos (MINVU, LeyChile), entiende la estructura y vocabulario

✅ **Adaptativo:** Con más datos de entrenamiento, la red mejora automáticamente. Las heurísticas hardcodeadas requieren cambios manuales

✅ **Contextual:** Entiende que "Diario" en "Diario de Implementación" es relevante, pero "Diario Oficial" en header es basura

✅ **Métricas medibles:** Accuracy, Precision, Recall, F1 por epoch. Sabes exactamente qué tan bien está funcionando

## Instalación

### 1. Dependencia de sistema: Tesseract OCR

`pytesseract` es solo un wrapper Python — el binario `tesseract` debe estar instalado en el sistema operativo:

**Ubuntu / Debian / WSL:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-spa
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows (nativo):**
Descarga el instalador desde [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) y asegúrate de marcar el idioma **Spanish** durante la instalación.

Verifica que quedó bien instalado:
```bash
tesseract --version
tesseract --list-langs   # debe aparecer "spa"
```

### 2. Instalar el paquete

```bash
pip install spharagusnet
```

La primera vez es necesario descargar el modelo pre-entrenado (~300 MB)

```bash
spharagusnet download
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

### Cómo se entrena este modelo

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
