#!/usr/bin/env python3
"""
CLI para extraer texto relevante de un PDF escaneado usando SpharagusNet.

Ejemplo:
    python scripts/extraer_texto_con_modelo.py documento.pdf salida.txt --confidence 0.6
"""

import sys
from pathlib import Path

# Paquete local
sys.path.insert(0, str(Path(__file__).parent.parent))
import spharagusnet


def main():
    if len(sys.argv) < 2:
        print("Uso: python extraer_texto_con_modelo.py <pdf> [salida.txt] [--confidence N]")
        return

    pdf_path = Path(sys.argv[1])
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
        else None
    )

    confidence = 0.5
    if "--confidence" in sys.argv:
        idx = sys.argv.index("--confidence")
        if idx + 1 < len(sys.argv):
            try:
                confidence = float(sys.argv[idx + 1])
            except ValueError:
                print("❌ --confidence debe ser un número entre 0 y 1")
                return

    if not pdf_path.exists():
        print(f"❌ No se encontró: {pdf_path}")
        return

    print("=" * 80)
    print("🚀 SpharagusNet — Extracción de Texto")
    print("=" * 80)

    texto = spharagusnet.extract(str(pdf_path), confidence=confidence)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(texto, encoding="utf-8")
        print(f"✅ Guardado en {output_path} ({len(texto)} chars)")
    else:
        print(f"\n--- Primeros 500 caracteres ---")
        print(texto[:500] + ("..." if len(texto) > 500 else ""))

    print("=" * 80)


if __name__ == "__main__":
    main()
