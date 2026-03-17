# Quick Start: SpharagusNet Pipeline

## Visión General

```
1. Preparar Dataset → 2. Entrenar Modelo → 3. Extraer Texto de PDFs
      ↓                        ↓                      ↓
  (30-60 min)            (5-15 min)             (2-5 seg/PDF)
```

---

## Paso 1: Preparar Dataset (30-60 min)

Descarga PDFs de LeyChile (referencia) y MINVU (imágenes), crea un dataset de entrenamiento.

```bash
cd /path/to/spharagusnet

# Procesar los primeros 100 pares (para prueba rápida)
python scripts/preparar_dataset_entrenamiento.py --max-pares 100

# O procesar TODOS (puede tardar horas)
python scripts/preparar_dataset_entrenamiento.py
```

**Outputs:**
```
data/dataset_entrenamiento/
├── imagenes/          # Páginas MINVU en PNG
├── textos/            # Textos de referencia LeyChile
├── metadata/          # JSONs con metadata
└── indice.json        # Índice master
```

**Requisitos previos:**
- `planes_origen_modificaciones_bcn.csv` en el directorio raíz
- Internet (para descargas)
- Espacio disco: ~1-2 GB para 100 pares

---

## Paso 2: Entrenar Modelo (5-15 min)

Entrena el clasificador binario (relevante/irrelevante) usando el dataset.

```bash
python scripts/entrenar_modelo_extraccion.py
```

**Ve en tiempo real:**
```
 Epoch │   Loss    Acc   Prec   Rec    F1 │  vLoss  vAcc  vPrec  vRec   vF1 │     t     ETA
   1/20│  0.6932  0.650  0.580  0.720  0.642 │  0.6801  0.670  0.600  0.700  0.646 │  2.3s   41.4s
   2/20│  0.5234  0.710  0.620  0.750  0.680 │  0.5102  0.725  0.640  0.770  0.700 │  2.1s   39.2s
   3/20│  0.4521  0.760  0.680  0.800  0.735 │  0.4398  0.770  0.690  0.810  0.745 │  2.0s   37.0s
```

**Se detiene automáticamente cuando:**
- val-F1 no mejora durante 5 epochs (early stopping)
- Se completan 20 epochs máximo

**Outputs:**
```
models/texto_extractor/
├── modelo_entrenado.pth          # Red neuronal (pesos)
├── vocab.json                    # Diccionario de palabras
├── historial_entrenamiento.json  # Métricas por epoch
├── datos_train.json              # Datos de entrenamiento
└── datos_val.json                # Datos de validación
```

**Métricas:**
- `Loss`: Qué tan equivocada está la red
- `Accuracy`: % bloques clasificados correctamente
- `Precision`: De los que predigo relevantes, ¿cuántos lo son?
- `Recall`: De los realmente relevantes, ¿cuántos detecto?
- `F1`: Balance de precision y recall

---

## Paso 3: Extraer Texto de PDFs (2-5 seg cada uno)

Usa el modelo entrenado para extraer solo texto relevante de cualquier PDF MINVU.

### Uso Básico

```bash
# Procesar un PDF y mostrar resultado en pantalla
python scripts/extraer_texto_con_modelo.py /ruta/al/pdf.pdf

# Guardar resultado en archivo de texto
python scripts/extraer_texto_con_modelo.py /ruta/al/pdf.pdf salida.txt
```

### Con Threshold de Confianza

```bash
# Más selectivo (solo bloques muy seguros)
python scripts/extraer_texto_con_modelo.py pdf.pdf salida.txt --confidence 0.7

# Menos selectivo (incluye más bloques)
python scripts/extraer_texto_con_modelo.py pdf.pdf salida.txt --confidence 0.3

# Default (balance)
python scripts/extraer_texto_con_modelo.py pdf.pdf salida.txt --confidence 0.5
```

**Parámetro `--confidence`:**
- Rango: 0.0 - 1.0
- 0.3 = Más bloques (recall ↑, precision ↓)
- 0.5 = Balance (default)
- 0.8 = Menos bloques (recall ↓, precision ↑)

### Ejemplo Real

```bash
python scripts/extraer_texto_con_modelo.py \
    /home/user/documentos/decreto_2024.pdf \
    /home/user/decretos_limpios/decreto_2024.txt \
    --confidence 0.6
```

**Output esperado:**
```
================================================================================
🚀 Extracción de Texto con Modelo Entrenado
================================================================================

📦 Cargando modelo...
✅ Modelo cargado desde models/texto_extractor/modelo_entrenado.pth
   Vocab size: 8234 | Hidden dim: 512
   Entrenado hasta epoch 12, val-F1: 0.7843

📄 Procesando: decreto_2024.pdf
   ✓ Página 1: 2345 caracteres
   ✓ Página 2: 1890 caracteres
   ✓ Página 3: 2156 caracteres
   ✓ Guardado en: decretos_limpios/decreto_2024.txt

✅ Extracción completa: 6391 caracteres en 3 páginas
================================================================================
```

---

## Flujo Completo (Ejemplo End-to-End)

```bash
#!/bin/bash

# 1. Preparar dataset con 50 pares (prueba rápida)
echo "📦 Preparando dataset..."
python scripts/preparar_dataset_entrenamiento.py --max-pares 50

# 2. Entrenar
echo "🚀 Entrenando modelo..."
python scripts/entrenar_modelo_extraccion.py

# 3. Procesar varios PDFs
echo "🔄 Extrayendo texto..."
python scripts/extraer_texto_con_modelo.py pdf1.pdf texto1.txt
python scripts/extraer_texto_con_modelo.py pdf2.pdf texto2.txt
python scripts/extraer_texto_con_modelo.py pdf3.pdf texto3.txt

echo "✅ Listo!"
```

---

## Documentación Completa

Para entender la arquitectura, ver:

```bash
# Visual del pipeline completo
python scripts/pipeline_completo.py

# Detalles técnicos
cat IMPLEMENTACION.md
```

---

## Solución de Problemas

### "No se encontró modelo"
```
⚠️  Modelo no encontrado en models/texto_extractor/modelo_entrenado.pth
   Ejecuta primero: python entrenar_modelo_extraccion.py
```
**Solución:** Entrena el modelo primero (Paso 2)

### "No hay datos para entrenar"
```
⚠️  No hay datos para entrenar
```
**Solución:** Prepara el dataset primero (Paso 1)

### "WSL se congela" durante preparación
- Síntoma: Sistema lento, se mata el proceso
- Causa: Alto consumo de memoria en conversión PDF
- **Solución:** Ya implementada (page-by-page + DPI=200)
- Si persiste: Reduce `--max-pares` a un número menor

### Confianza muy baja (~0.5) en todas las predicciones
- Posible causa: Dataset de mala calidad (OCR con errores)
- Verificar: Comparar OCR vs texto de referencia
- Solución: Aumentar dataset, mejorar OCR

---

## Parámetros Clave

| Parámetro | Ubicación | Default | Rango |
|-----------|-----------|---------|-------|
| `MAX_PARES` | preparar_dataset_entrenamiento.py | `None` | 1-∞ |
| `batch_size` | entrenar_modelo_extraccion.py | 32 | 16-128 |
| `num_epochs` | entrenar_modelo_extraccion.py | 20 | 1-∞ |
| `patience` (early stop) | entrenar_modelo_extraccion.py | 5 | 1-∞ |
| `--confidence` | extraer_texto_con_modelo.py | 0.5 | 0.0-1.0 |

---

## Performance Tips

### Acelerar Entrenamiento
- ✅ Usar GPU (CUDA): ~3x más rápido
- ✅ Aumentar `batch_size` a 64-128 (si memoria lo permite)
- ❌ NO: Aumentar `num_epochs` sin monitorear (early stopping lo hace)

### Acelerar Preparación Dataset
- ✅ Usar `--max-pares` para pruebas rápidas
- ✅ Ejecutar en máquina potente con SSD
- ✅ Conexión rápida a internet

### Acelerar Inferencia
- ✅ Procesamiento PDF ya optimizado (page-by-page)
- ✅ Batch inference interno (rápido)
- ❌ NO: Aumentar `--confidence` (no afecta speed, solo cantidad de bloques)

---

## Estructura Final de Archivos

```
spharagusnet/
├── scripts/
│   ├── entrenar_modelo_extraccion.py     (24 KB - entrenamiento)
│   ├── extraer_texto_con_modelo.py       (15 KB - inferencia)
│   ├── pipeline_completo.py              (7 KB - documentación)
│   └── ...
├── data/
│   └── dataset_entrenamiento/
│       ├── imagenes/
│       ├── textos/
│       ├── metadata/
│       └── indice.json
├── models/
│   └── texto_extractor/
│       ├── modelo_entrenado.pth
│       ├── vocab.json
│       └── historial_entrenamiento.json
├── README.md                              (actualizado)
├── IMPLEMENTACION.md                      (detalle técnico)
└── QUICKSTART.md                          (este archivo)
```

---

## Próximos Pasos

1. **Ejecutar quickstart:** `python scripts/pipeline_completo.py`
2. **Preparar dataset:** `python scripts/preparar_dataset_entrenamiento.py --max-pares 50`
3. **Entrenar:** `python scripts/entrenar_modelo_extraccion.py`
4. **Procesar PDFs:** `python scripts/extraer_texto_con_modelo.py pdf.pdf salida.txt`

¡Listo! Ahora puedes extraer texto limpio de cualquier PDF MINVU. 🚀

