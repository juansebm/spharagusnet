#!/usr/bin/env python3
"""
Pipeline completo de extracción de texto desde PDFs de MINVU.

Flujo:
1. Preparar dataset (LeyChile + MINVU) → preparar_dataset_entrenamiento.py
2. Entrenar modelo → entrenar_modelo_extraccion.py
3. Inferencia en nuevos PDFs → extraer_texto_con_modelo.py (este script)
"""

from pathlib import Path
import sys


def main():
    """Demuestra el pipeline completo."""
    base_dir = Path(__file__).parent.parent

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETO: Extracción de Texto de PDFs MINVU")
    print("=" * 80)

    print("\n📋 PASOS DEL PIPELINE:\n")

    print("1️⃣  PREPARAR DATASET")
    print("-" * 80)
    print("   Requiere: planes_origen_modificaciones_bcn.csv")
    print("   Descarga: PDFs de LeyChile (texto limpio) y MINVU (imágenes)")
    print("   Output:  data/dataset_entrenamiento/")
    print("            ├── imagenes/        (páginas MINVU en PNG)")
    print("            ├── textos/          (referencias LeyChile)")
    print("            ├── metadata/        (índice JSON)")
    print("            └── indice.json")
    print("\n   Ejecutar:")
    print("   $ python preparar_dataset_entrenamiento.py [--max-pares N]")

    print("\n" + "-" * 80)
    print("\n2️⃣  ENTRENAR MODELO")
    print("-" * 80)
    print("   Input:   data/dataset_entrenamiento/")
    print("   Output:  models/texto_extractor/")
    print("            ├── modelo_entrenado.pth  (pesos del modelo)")
    print("            ├── vocab.json            (vocabulario)")
    print("            ├── historial_entrenamiento.json (métricas por epoch)")
    print("            ├── datos_train.json")
    print("            └── datos_val.json")
    print("\n   Características:")
    print("   • Clasificador binario: relevante vs irrelevante")
    print("   • Métricas por epoch: Loss, Accuracy, Precision, Recall, F1")
    print("   • Early stopping sobre val-F1")
    print("   • Pesos de clase para compensar desbalance")
    print("\n   Ejecutar:")
    print("   $ python entrenar_modelo_extraccion.py")

    print("\n" + "-" * 80)
    print("\n3️⃣  INFERENCIA: Extraer texto de nuevos PDFs")
    print("-" * 80)
    print("   Input:   PDF MINVU escaneado (cualquiera)")
    print("   Output:  Texto limpio (solo bloques relevantes)")
    print("\n   Proceso por cada página:")
    print("   a) OCR (Tesseract) → extrae bloques de texto")
    print("   b) Tokenización → convierte a índices")
    print("   c) Red neuronal → clasifica cada bloque (0=irrelevante, 1=relevante)")
    print("   d) Reconstrucción → une solo bloques relevantes")
    print("\n   Ejecutar:")
    print("   $ python extraer_texto_con_modelo.py <ruta_pdf> [salida.txt] [--confidence 0.5]")

    print("\n" + "=" * 80)
    print("\n🔄 ARQUITECTURA DEL MODELO:")
    print("""
┌─────────────────────────────────────────────────────────┐
│ INPUT: Bloque OCR (e.g., "Artículo 5 párrafo a")       │
└──────────────────────┬──────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │ 1. TOKENIZACIÓN + EMBEDDING         │
    │    texto → [idx_1, idx_2, ...]      │
    │    → Dense vectors (512-D)          │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │ 2. POOLING (promedia embeddings)    │
    │    (64, 512) → (512)                │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │ 3. BLOCK CLASSIFIER                 │
    │    FC(512→256) + ReLU + Dropout     │
    │    FC(256→2)   ← logits             │
    │    argmax() → predicción {0, 1}     │
    └──────────────────┬───────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│ OUTPUT: Clasificación binaria               │
│  0 = Irrelevante (descarta)                 │
│  1 = Relevante   (mantiene)                 │
└──────────────────────────────────────────────┘
    """)

    print("\n" + "=" * 80)
    print("📊 MÉTRICAS DE ENTRENAMIENTO (por epoch):")
    print("""
  Epoch │   Loss    Acc   Prec   Rec    F1 │  vLoss  vAcc  vPrec  vRec   vF1 │     t     ETA
   1/20│  0.6932  0.650  0.580  0.720  0.642 │  0.6801  0.670  0.600  0.700  0.646 │  2.3s   41.4s
   2/20│  0.5234  0.710  0.620  0.750  0.680 │  0.5102  0.725  0.640  0.770  0.700 │  2.1s   39.2s
   3/20│  0.4521  0.760  0.680  0.800  0.735 │  0.4398  0.770  0.690  0.810  0.745 │  2.0s   37.0s
   ...

Columnas:
  • Epoch: época actual / total
  • Loss/vLoss: pérdida en train / validación
  • Acc/Prec/Rec/F1: accuracy, precision, recall, F1 en train
  • vAcc/vPrec/vRec/vF1: mismas métricas en validación
  • t: tiempo de época
  • ETA: tiempo estimado para terminar

Early stopping: detiene si val-F1 no mejora durante 5 épocas
    """)

    print("\n" + "=" * 80)
    print("⚠️  NOTAS IMPORTANTES:")
    print("""
1. Dataset
   - Requiere OCR válido en las imágenes MINVU (confiance > 30%)
   - La calidad del modelo depende de la calidad del dataset
   - Desbalance: hay más bloques irrelevantes que relevantes

2. Entrenamiento
   - Primera ejecución puede ser lenta (~1-2 horas para 100+ pares)
   - Usa GPU si está disponible (mucho más rápido)
   - Early stopping evita overfitting

3. Inferencia
   - Usa el modelo guardado (no requiere re-entrenamiento)
   - Si el modelo no está disponible, usa heurísticas de fallback
   - confidence_threshold ajusta la selectividad (0.5 default)
     * Menor threshold → más bloques (recall ↑, precision ↓)
     * Mayor threshold → menos bloques (recall ↓, precision ↑)

4. Mejoras futuras
   - Usar LayoutLMv3 (incorpora ubicación en página)
   - Vision-language models (LLaVA, GPT-4V)
   - Fine-tuning con más datos específicos
    """)

    print("\n" + "=" * 80)
    print("🚀 COMENZAR:")
    print("""
1. Preparar dataset:
   $ python preparar_dataset_entrenamiento.py --max-pares 100

2. Entrenar:
   $ python entrenar_modelo_extraccion.py

3. Procesar un PDF:
   $ python extraer_texto_con_modelo.py /ruta/al/pdf.pdf salida.txt --confidence 0.5
    """)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
