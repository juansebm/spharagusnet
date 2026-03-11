#!/usr/bin/env python3
"""
Script para descargar documentos de instrumentos de planificación
desde planes_origen_modificaciones.csv para las comunas especificadas
en comunas_interes.txt.

Descarga tanto instrumentos de origen como sus modificaciones.
Solo descarga los PDFs originales sin procesamiento (sin OCR ni conversión a TXT).
"""

import csv
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
from urllib.parse import urlparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

# Tipos de documentos a excluir (planos)
TIPOS_EXCLUIDOS = ["plano", "planos"]

# Configuración
BASE_DIR = Path(__file__).parent.parent
COMUNAS_INTERES_FILE = BASE_DIR / "scripts" / "comunas_interes.txt"
CSV_FILE = BASE_DIR / "data" / "planes_origen_modificaciones.csv"
OUTPUT_DIR = BASE_DIR / "data" / "documentos_prc"

# Configuración de descarga
TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 0.5  # Segundos entre descargas para no sobrecargar el servidor

def load_comunas_interes() -> Set[str]:
    """
    Carga las comunas de interés desde el archivo de texto.
    
    Returns:
        Set con los nombres de las comunas (normalizados a mayúsculas)
    """
    if not COMUNAS_INTERES_FILE.exists():
        console.print(f"[red]❌ Archivo no encontrado: {COMUNAS_INTERES_FILE}[/]")
        return set()
    
    comunas = set()
    with open(COMUNAS_INTERES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            comuna = line.strip().upper()
            if comuna:
                comunas.add(comuna)
    
    console.print(f"[green]✅ Cargadas {len(comunas)} comunas de interés: {', '.join(sorted(comunas))}[/]")
    return comunas

def load_instrumentos_from_csv(comunas_interes: Set[str]) -> List[Dict]:
    """
    Carga instrumentos del CSV filtrados por comunas de interés.
    
    Args:
        comunas_interes: Set con nombres de comunas de interés
        
    Returns:
        Lista de instrumentos (origen y modificaciones)
    """
    if not CSV_FILE.exists():
        console.print(f"[red]❌ Archivo CSV no encontrado: {CSV_FILE}[/]")
        return []
    
    instrumentos = []
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            # Obtener comunas del registro (puede ser una o varias separadas por comas)
            comunas_str = row.get("comunas", "").strip()
            if not comunas_str:
                continue
            
            # Verificar si alguna de las comunas está en la lista de interés
            comunas_registro = [c.strip().upper() for c in comunas_str.split(",")]
            if any(comuna in comunas_interes for comuna in comunas_registro):
                instrumentos.append(row)
    
    console.print(f"[green]✅ Encontrados {len(instrumentos)} instrumentos para las comunas de interés[/]")
    
    # Separar origen y modificaciones
    origen = [i for i in instrumentos if i.get("es_modificacion", "").lower() == "false"]
    modificaciones = [i for i in instrumentos if i.get("es_modificacion", "").lower() == "true"]
    
    console.print(f"[cyan]   - Instrumentos de origen: {len(origen)}[/]")
    console.print(f"[cyan]   - Modificaciones: {len(modificaciones)}[/]")
    
    return instrumentos

def parse_urls(urls_str: str) -> List[str]:
    """
    Parsea las URLs separadas por ' | '.
    
    Args:
        urls_str: String con URLs separadas por ' | '
        
    Returns:
        Lista de URLs
    """
    if not urls_str or not urls_str.strip():
        return []
    
    # Separar por ' | ' y limpiar espacios
    urls = [url.strip() for url in urls_str.split(" | ") if url.strip()]
    return urls


def parse_urls_con_tipos(instrumento: Dict) -> List[Tuple[str, str, str]]:
    """
    Parsea las URLs junto con sus tipos y nombres, filtrando planos.
    
    Args:
        instrumento: Diccionario con datos del instrumento
        
    Returns:
        Lista de tuplas (url, tipo, nombre) excluyendo planos
    """
    urls_str = instrumento.get("urls_documentos", "")
    tipos_str = instrumento.get("tipos_documentos", "")
    nombres_str = instrumento.get("nombres_documentos", "")
    
    if not urls_str or not urls_str.strip():
        return []
    
    urls = [u.strip() for u in urls_str.split(" | ") if u.strip()]
    tipos = [t.strip() for t in tipos_str.split(" | ")] if tipos_str else [""] * len(urls)
    nombres = [n.strip() for n in nombres_str.split(" | ")] if nombres_str else [""] * len(urls)
    
    # Asegurar que las listas tengan el mismo tamaño
    while len(tipos) < len(urls):
        tipos.append("")
    while len(nombres) < len(urls):
        nombres.append("")
    
    # Filtrar planos
    resultado = []
    planos_excluidos = 0
    for url, tipo, nombre in zip(urls, tipos, nombres):
        tipo_lower = tipo.lower().strip()
        # Verificar si es un plano
        if any(excluido in tipo_lower for excluido in TIPOS_EXCLUIDOS):
            planos_excluidos += 1
            continue
        resultado.append((url, tipo, nombre))
    
    if planos_excluidos > 0:
        console.print(f"  [dim]⏭️  Excluidos {planos_excluidos} plano(s)[/]")
    
    return resultado


def get_filename_from_url(url: str, instrumento: Dict) -> str:
    """
    Genera un nombre de archivo descriptivo desde la URL y datos del instrumento.
    
    Args:
        url: URL del documento
        instrumento: Diccionario con datos del instrumento
        
    Returns:
        Nombre de archivo sugerido
    """
    # Obtener nombre del archivo desde la URL
    parsed = urlparse(url)
    original_filename = Path(parsed.path).name
    
    # Obtener información del instrumento
    codigo = instrumento.get("codigo", "").strip()
    denominacion = instrumento.get("denominacion", "").strip()
    es_modificacion = instrumento.get("es_modificacion", "").lower() == "true"
    numero_doc = instrumento.get("numeroDocumento", "").strip()
    
    # Limpiar denominación para usar en nombre de archivo
    denominacion_clean = "".join(c for c in denominacion if c.isalnum() or c in (" ", "-", "_")).strip()
    denominacion_clean = denominacion_clean.replace(" ", "_")[:50]  # Limitar longitud
    
    # Construir nombre
    if codigo:
        prefix = f"{codigo}"
    elif numero_doc and numero_doc != "Sin información" and numero_doc != "No aplica":
        prefix = f"Doc_{numero_doc}"
    else:
        prefix = f"ID_{instrumento.get('id', 'unknown')}"
    
    # Agregar tipo
    tipo = "MOD" if es_modificacion else "ORIGEN"
    
    # Mantener extensión original
    ext = Path(original_filename).suffix or ".pdf"
    
    # Si hay denominación, incluirla
    if denominacion_clean:
        filename = f"{prefix}_{tipo}_{denominacion_clean}{ext}"
    else:
        filename = f"{prefix}_{tipo}_{original_filename}"
    
    # Limpiar caracteres problemáticos
    filename = "".join(c for c in filename if c.isalnum() or c in ("-", "_", "."))
    
    return filename

def download_file(url: str, output_path: Path, instrumento: Dict) -> bool:
    """
    Descarga un archivo desde una URL.
    
    Args:
        url: URL del archivo a descargar
        output_path: Ruta donde guardar el archivo
        instrumento: Diccionario con datos del instrumento (para logs)
        
    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    try:
        response = requests.get(url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    
    except requests.exceptions.RequestException as e:
        console.print(f"[red]  ❌ Error descargando {url}: {e}[/]")
        return False
    except Exception as e:
        console.print(f"[red]  ❌ Error guardando {output_path}: {e}[/]")
        return False


def build_instrumento_metadata(instrumento: Dict, archivos_descargados: List[Dict]) -> Dict:
    """
    Construye el diccionario de metadatos para un instrumento.
    
    Args:
        instrumento: Diccionario con datos del instrumento desde CSV
        archivos_descargados: Lista de archivos descargados para este instrumento
        
    Returns:
        Diccionario con metadatos estructurados
    """
    return {
        # Información del instrumento
        "id": instrumento.get("id"),
        "codigo": instrumento.get("codigo"),
        "denominacion": instrumento.get("denominacion"),
        "planificacion": instrumento.get("planificacion"),
        "tipo": instrumento.get("tipo"),
        
        # Información geográfica
        "region_codigo": instrumento.get("region_codigo"),
        "region_nombre": instrumento.get("region_nombre"),
        "comunas": instrumento.get("comunas"),
        "comunas_codigos": instrumento.get("comunas_codigos"),
        
        # Clasificación
        "clasificacion": instrumento.get("clasificacion"),
        "estado": instrumento.get("estado"),
        "es_modificacion": instrumento.get("es_modificacion", "").lower() == "true",
        "instrumento_origen_id": instrumento.get("instrumento_origen_id"),
        "codigos_modificaciones": instrumento.get("codigos_modificaciones"),
        
        # Fechas
        "fechaInicioVigencia": instrumento.get("fechaInicioVigencia"),
        "fechaPublicacion": instrumento.get("fechaPublicacion"),
        "fechaInicioElaboracion": instrumento.get("fechaInicioElaboracion"),
        
        # Información del documento
        "numeroDocumento": instrumento.get("numeroDocumento"),
        "urlPublicacion": instrumento.get("urlPublicacion"),
        
        # URLs y tipos de documentos (todos)
        "urls_documentos": [u.strip() for u in instrumento.get("urls_documentos", "").split(" | ") if u.strip()],
        "tipos_documentos": [t.strip() for t in instrumento.get("tipos_documentos", "").split(" | ") if t.strip()],
        "nombres_documentos": [n.strip() for n in instrumento.get("nombres_documentos", "").split(" | ") if n.strip()],
        
        # Información adicional
        "modificacionLimiteUrbano": instrumento.get("modificacionLimiteUrbano"),
        "evaluacionAmbientalEstrategica": instrumento.get("evaluacionAmbientalEstrategica"),
        "fechaInicioEae": instrumento.get("fechaInicioEae"),
        "fechaTerminoEae": instrumento.get("fechaTerminoEae"),
        "consultaIndigena": instrumento.get("consultaIndigena"),
        "modificacionIntegral": instrumento.get("modificacionIntegral"),
        "fuente": instrumento.get("fuente"),
        
        # Timestamps del CSV
        "createdAt": instrumento.get("createdAt"),
        "updatedAt": instrumento.get("updatedAt"),
        
        # Archivos descargados localmente
        "archivos_locales": archivos_descargados
    }


def save_comuna_metadata(comuna: str, instrumentos: List[Dict], 
                         archivos_por_instrumento: Dict[str, List[Dict]],
                         comuna_dir: Path):
    """
    Guarda el archivo JSON de metadatos para una comuna.
    
    Args:
        comuna: Nombre de la comuna
        instrumentos: Lista de instrumentos de esta comuna
        archivos_por_instrumento: Diccionario con archivos descargados por ID de instrumento
        comuna_dir: Directorio de la comuna
    """
    metadata_path = comuna_dir / "metadata.json"
    
    # Separar instrumentos de origen y modificaciones
    instrumentos_origen = []
    instrumentos_modificaciones = []
    
    for instrumento in instrumentos:
        instrumento_id = instrumento.get("id", "")
        archivos = archivos_por_instrumento.get(instrumento_id, [])
        metadata = build_instrumento_metadata(instrumento, archivos)
        
        if instrumento.get("es_modificacion", "").lower() == "true":
            instrumentos_modificaciones.append(metadata)
        else:
            instrumentos_origen.append(metadata)
    
    # Ordenar modificaciones por fecha de vigencia
    instrumentos_modificaciones.sort(
        key=lambda x: x.get("fechaInicioVigencia") or "", 
        reverse=False
    )
    
    # Construir estructura final
    metadata_comuna = {
        "comuna": comuna,
        "fecha_actualizacion": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resumen": {
            "total_instrumentos": len(instrumentos),
            "instrumentos_origen": len(instrumentos_origen),
            "modificaciones": len(instrumentos_modificaciones),
            "total_archivos": sum(len(archivos_por_instrumento.get(i.get("id", ""), [])) for i in instrumentos)
        },
        "instrumentos_origen": instrumentos_origen,
        "modificaciones": instrumentos_modificaciones
    }
    
    # Crear directorio si no existe
    comuna_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar JSON
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_comuna, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]✅ Metadata guardada: {metadata_path}[/]")

def download_documentos(instrumentos: List[Dict], comunas_interes: Set[str]):
    """
    Descarga todos los documentos PDF de los instrumentos (excluyendo planos).
    Solo descarga los PDFs originales sin procesamiento.
    
    Args:
        instrumentos: Lista de instrumentos
        comunas_interes: Set con nombres de comunas de interés
    """
    # Crear directorio base
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Organizar por comuna
    instrumentos_por_comuna: Dict[str, List[Dict]] = {}
    
    for instrumento in instrumentos:
        comunas_str = instrumento.get("comunas", "").strip()
        if not comunas_str:
            continue
        
        comunas_registro = [c.strip().upper() for c in comunas_str.split(",")]
        # Asignar a la primera comuna que esté en la lista de interés
        for comuna in comunas_registro:
            if comuna in comunas_interes:
                if comuna not in instrumentos_por_comuna:
                    instrumentos_por_comuna[comuna] = []
                instrumentos_por_comuna[comuna].append(instrumento)
                break
    
    # Contar total de documentos (excluyendo planos)
    total_docs = 0
    total_planos_excluidos = 0
    for instrumento in instrumentos:
        docs_con_tipos = parse_urls_con_tipos(instrumento)
        total_docs += len(docs_con_tipos)
        # Contar planos excluidos
        urls_originales = parse_urls(instrumento.get("urls_documentos", ""))
        total_planos_excluidos += len(urls_originales) - len(docs_con_tipos)
    
    console.print(f"\n[bold cyan]📥 Descargando {total_docs} documentos PDF (excluyendo {total_planos_excluidos} planos)...[/]")
    
    # Descargar documentos
    descargados = 0
    fallidos = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Descargando...", total=total_docs)
        
        for comuna, insts_comuna in instrumentos_por_comuna.items():
            comuna_dir = OUTPUT_DIR / comuna.replace(" ", "_")
            
            # Diccionario para rastrear archivos descargados por instrumento
            archivos_por_instrumento: Dict[str, List[Dict]] = {}
            
            for instrumento in insts_comuna:
                instrumento_id = instrumento.get("id", "")
                
                # Obtener URLs filtradas (sin planos)
                docs_con_tipos = parse_urls_con_tipos(instrumento)
                
                # Inicializar lista de archivos para este instrumento
                if instrumento_id not in archivos_por_instrumento:
                    archivos_por_instrumento[instrumento_id] = []
                
                if not docs_con_tipos:
                    continue
                
                es_modificacion = instrumento.get("es_modificacion", "").lower() == "true"
                tipo_dir = comuna_dir / ("modificaciones" if es_modificacion else "origen")
                tipo_carpeta = "modificaciones" if es_modificacion else "origen"
                
                # Mostrar información del instrumento
                codigo = instrumento.get("codigo", "")
                denominacion = instrumento.get("denominacion", "")[:60]
                tipo = "MODIFICACIÓN" if es_modificacion else "ORIGEN"
                
                console.print(f"\n[bold yellow]{comuna}[/] - [cyan]{tipo}[/] - {denominacion} (Código: {codigo})")
                console.print(f"  [dim]Documentos: {len(docs_con_tipos)} (sin planos)[/]")
                
                for idx, (url, tipo_doc, nombre_doc) in enumerate(docs_con_tipos, 1):
                    filename = get_filename_from_url(url, instrumento)
                    
                    # Si hay múltiples documentos, numerarlos
                    if len(docs_con_tipos) > 1:
                        name_part = Path(filename).stem
                        ext = Path(filename).suffix
                        filename = f"{name_part}_{idx}{ext}"
                    
                    output_path = tipo_dir / filename
                    ruta_relativa = f"{tipo_carpeta}/{filename}"
                    
                    # Registrar archivo en metadata
                    archivo_info = {
                        "nombre": filename,
                        "ruta_relativa": ruta_relativa,
                        "url_origen": url,
                        "tipo_documento": tipo_doc,
                        "nombre_documento": nombre_doc,
                        "indice": idx,
                        "descargado": False,
                        "formato_final": "pdf"
                    }
                    
                    # Verificar si ya existe el PDF
                    pdf_ya_existe = output_path.exists()
                    
                    if pdf_ya_existe:
                        console.print(f"  [dim]⏭️  Ya existe: {filename}[/]")
                        archivo_info["descargado"] = True
                        archivo_info["ya_existia"] = True
                        descargados += 1
                    else:
                        # Descargar
                        console.print(f"  [cyan]⬇️  Descargando: {filename}[/]")
                        if download_file(url, output_path, instrumento):
                            descargados += 1
                            archivo_info["descargado"] = True
                            archivo_info["ya_existia"] = False
                            console.print(f"  [green]✓ Descargado: {filename}[/]")
                        else:
                            fallidos += 1
                            archivo_info["error"] = True
                    
                    archivos_por_instrumento[instrumento_id].append(archivo_info)
                    progress.update(task, advance=1)
                    
                    # Pausa entre descargas
                    if idx < len(docs_con_tipos):
                        time.sleep(DELAY_BETWEEN_REQUESTS)
            
            # Guardar metadata de la comuna después de procesar todos sus instrumentos
            save_comuna_metadata(comuna, insts_comuna, archivos_por_instrumento, comuna_dir)
    
    # Resumen
    console.print(f"\n[bold green]✅ Proceso completado[/]")
    console.print(f"[green]   PDFs descargados: {descargados}[/]")
    if fallidos > 0:
        console.print(f"[red]   Descargas fallidas: {fallidos}[/]")
    console.print(f"[cyan]   Directorio: {OUTPUT_DIR}[/]")

def main():
    """Función principal"""
    console.print("[bold magenta]📥 Descargador de Documentos PRC[/]")
    console.print(f"[bold magenta]Fuente: {CSV_FILE.name}[/]\n")
    
    try:
        # 1. Cargar comunas de interés
        comunas_interes = load_comunas_interes()
        
        if not comunas_interes:
            console.print("[yellow]⚠️  No hay comunas de interés definidas[/]")
            return
        
        # 2. Cargar instrumentos del CSV
        instrumentos = load_instrumentos_from_csv(comunas_interes)
        
        if not instrumentos:
            console.print("[yellow]⚠️  No se encontraron instrumentos para las comunas de interés[/]")
            return
        
        # 3. Descargar documentos
        download_documentos(instrumentos, comunas_interes)
        
        console.print("\n[bold green]✅ Proceso completado exitosamente[/]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/]")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
