"""
CLI de SpharagusNet.

Uso:
    spharagusnet extract documento.pdf                  # imprime texto
    spharagusnet extract documento.pdf -o salida.txt    # guarda a archivo
    spharagusnet extract documento.pdf --confidence 0.6
    spharagusnet download                               # descarga modelo
    spharagusnet download --force                       # re-descarga
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_extract(args: argparse.Namespace) -> None:
    from spharagusnet.extract import extract
    from spharagusnet.paths import model_is_available

    if not model_is_available():
        print("⚠️  Modelo no encontrado. Descargándolo automáticamente...")
        from spharagusnet.download import download_model
        try:
            download_model()
        except Exception as e:
            print(f"❌ No se pudo descargar: {e}")
            print("   Puedes usar heurísticas (sin modelo) o descargarlo manualmente.")

    texto = extract(args.pdf, confidence=args.confidence, dpi=args.dpi)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(texto, encoding="utf-8")
        print(f"✅ {len(texto)} caracteres → {out}")
    else:
        print(texto)


def cmd_download(args: argparse.Namespace) -> None:
    from spharagusnet.download import download_model
    download_model(force=args.force)


def cmd_info(args: argparse.Namespace) -> None:
    import spharagusnet
    from spharagusnet.paths import (
        MODEL_CACHE_DIR,
        _DEV_DIR,
        get_model_path,
        get_vocab_path,
        model_is_available,
    )

    print(f"SpharagusNet v{spharagusnet.__version__}")
    print(f"  Modelo disponible : {'✅ Sí' if model_is_available() else '❌ No'}")
    print(f"  Modelo path       : {get_model_path()}")
    print(f"  Vocab path        : {get_vocab_path()}")
    print(f"  Cache dir         : {MODEL_CACHE_DIR}")
    print(f"  Dev dir           : {_DEV_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="spharagusnet",
        description="Neural OCR para documentos de planificación territorial chilenos",
    )
    sub = parser.add_subparsers(dest="command")

    # extract
    p_ext = sub.add_parser("extract", help="Extrae texto de un PDF escaneado")
    p_ext.add_argument("pdf", help="Ruta al PDF")
    p_ext.add_argument("-o", "--output", help="Guardar texto en archivo")
    p_ext.add_argument("--confidence", type=float, default=0.5,
                       help="Umbral de confianza [0-1] (default: 0.5)")
    p_ext.add_argument("--dpi", type=int, default=200,
                       help="DPI para rasterización (default: 200)")
    p_ext.set_defaults(func=cmd_extract)

    # download
    p_dl = sub.add_parser("download", help="Descarga el modelo pre-entrenado")
    p_dl.add_argument("--force", action="store_true",
                      help="Re-descargar aunque ya exista")
    p_dl.set_defaults(func=cmd_download)

    # info
    p_info = sub.add_parser("info", help="Muestra información del paquete")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
