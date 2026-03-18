"""
SpharagusNet — Extracción de texto relevante de documentos escaneados.

Uso:
    import spharagusnet

    # Descargar modelo (primera vez)
    spharagusnet.download_model()

    # Extraer texto
    texto = spharagusnet.extract("path/to/scanned.pdf")
    texto = spharagusnet.extract("path/to/scanned.pdf", confidence=0.6)

CLI:
    spharagusnet download
    spharagusnet extract documento.pdf -o salida.txt
    spharagusnet info
"""

from spharagusnet.download import download_model
from spharagusnet.extract import extract, load_model
from spharagusnet.paths import model_is_available

__all__ = ["extract", "load_model", "download_model", "model_is_available"]
__version__ = "0.2.1"
