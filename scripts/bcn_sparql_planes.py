#!/usr/bin/env python3
"""
bcn_sparql_planes.py
====================
Lee comunas de `comunas_interes.txt`, filtra `planes_origen_modificaciones.csv`
por el campo "comunas" y busca cada documento en BCN vía SPARQL.

Uso:
  python bcn_sparql_planes.py            # solo comunas de comunas_interes.txt
  python bcn_sparql_planes.py --todas    # TODAS las comunas del CSV

Estrategia por registro — 100% determinista, 0 LLM:
  Track 1 — SPARQL con fecha:
    • Con número de decreto: commune + ?numero + fecha exacta.
    • Sin número: commune + 1-2 palabras del título + fecha.
  Track 2 — SPARQL sin fecha (fallback):
    • Idéntico a Track 1 pero sin filtro de fecha.
    • Amplía búsqueda cuando hay discrepancia de fecha entre MINVU y BCN.
  Si ningún track da resultado → registro omitido.

Por qué este diseño:
  • ?numero en BCN coincide con el número del decreto → discriminador más preciso.
  • Commune en ?titulo siempre presente: desambigua números cortos (ej. "4").
  • Sin LLM: reproducible, sin coste, sin alucinaciones de términos sin tilde.

Salida:
  data/planes_origen_modificaciones_bcn.csv
  scripts/LOG_planes_bcn.txt
"""

import argparse
import os
import re
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.error import URLError

from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON

# ── Rutas ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

CSV_INPUT            = PROJECT_ROOT / "data" / "planes_origen_modificaciones.csv"
CSV_OUTPUT           = PROJECT_ROOT / "data" / "planes_origen_modificaciones_bcn.csv"
COMUNAS_INTERES_FILE = SCRIPT_DIR / "comunas_interes.txt"
LOG_FILE             = SCRIPT_DIR / "LOG_planes_bcn.txt"
SPARQL_ENDPOINT      = "http://datos.bcn.cl/sparql"


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE URL PDF LEYCHILE
# ══════════════════════════════════════════════════════════════════════════════

_MESES_ES = {
    1: "ENE", 2: "FEB", 3: "MAR", 4: "ABR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DIC",
}


def _convertir_fecha_corta(fecha_iso: str) -> str:
    """Convierte 'YYYY-MM-DD' → 'DD-MES-YYYY' (ej. '17-ENE-2003')."""
    try:
        from datetime import date as _date
        d = _date.fromisoformat(fecha_iso[:10])
        return f"{d.day:02d}-{_MESES_ES[d.month]}-{d.year}"
    except Exception:
        return ""


def construir_url_pdf(leychile_code: str, numero: str, fecha_iso: str) -> str:
    """
    Construye la URL de descarga directa del PDF en leychile.cl.

    Formato:
      https://nuevo.leychile.cl/servicios/Consulta/Exportar?
        radioExportar=Normas&exportar_formato=pdf&
        nombrearchivo=DTO-{numero}_{fecha_corta}&
        exportar_con_notas_bcn=False&...&
        hddResultadoExportar={leychileCode}.{fecha_iso}.0.0%23

    Devuelve '' si faltan datos esenciales.
    """
    if not leychile_code or not fecha_iso:
        return ""
    fecha_corta = _convertir_fecha_corta(fecha_iso)
    if not fecha_corta:
        return ""
    nombre_archivo = f"DTO-{numero}_{fecha_corta}" if numero else f"DTO-{leychile_code}_{fecha_corta}"
    return (
        "https://nuevo.leychile.cl/servicios/Consulta/Exportar?"
        "radioExportar=Normas&"
        "exportar_formato=pdf&"
        f"nombrearchivo={nombre_archivo}&"
        "exportar_con_notas_bcn=False&"
        "exportar_con_notas_originales=False&"
        "exportar_con_notas_al_pie=False&"
        f"hddResultadoExportar={leychile_code}.{fecha_iso}.0.0%23"
    )


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

# Palabras demasiado genéricas para discriminar en BCN (solo las realmente neutras).
# NOTA: verbos de acción como MODIFICA, APRUEBA, PROMULGA NO están aquí;
#       sí discriminan y deben incluirse en las queries.
_GENERICAS = frozenset({
    "DE", "DEL", "LA", "EL", "LOS", "LAS", "Y", "EN", "A", "AL", "CON",
    "QUE", "SE", "UN", "UNA", "SU", "POR", "PARA", "O", "E", "U",
    "PLAN", "REGULADOR", "COMUNAL", "PRC",
})


def _quitar_acentos(texto: str) -> str:
    for src, dst in [("Á","A"),("É","E"),("Í","I"),("Ó","O"),("Ú","U"),("Ñ","N"),
                     ("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
        texto = texto.replace(src, dst)
    return texto


def _numero_valido(numero: str) -> bool:
    """¿Es el número de decreto un valor real y específico?"""
    malos = {"nan", "sin información", "sin informacion", "no aplica",
             "no corresponde", "sin informacion.", ""}
    return bool(numero) and numero.lower() not in malos and len(numero) >= 1


def _normalizar_numero(numero_doc: str) -> str:
    """
    Normaliza el número de decreto para que coincida con el campo ?numero de BCN.

    BCN almacena solo el número base (ej. "1335 exento"), mientras que MINVU
    puede incluir prefijos tipo "202-1335" (año-número) o "I-45" (región-número).
    Cuando hay guión, usamos únicamente el último segmento.

    Ejemplos:
      "202-1335" → "1335"   ("1335 exento" en BCN → CONTAINS("1335") = True)
      "I-45"     → "45"
      "1370"     → "1370"   (sin cambio)
    """
    n = _quitar_acentos(numero_doc.strip().upper())
    if "-" in n:
        n = n.split("-")[-1].strip()
    return n


def _palabras_clave(denominacion: str, excluir: set, max_n: int = 2) -> list[str]:
    """
    Extrae palabras de `denominacion` que sean útiles para un CONTAINS en SPARQL.
    - Mínimo 4 caracteres.
    - No presentes en `excluir`.
    - Sin acentos (para máxima compatibilidad con BCN).
    """
    palabras: list[str] = []
    for w in denominacion.upper().split():
        w_clean = _quitar_acentos(re.sub(r"[^A-Z0-9]", "", w))
        if len(w_clean) >= 4 and w_clean not in excluir:
            palabras.append(w_clean)
            if len(palabras) >= max_n:
                break
    return palabras


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE QUERIES SPARQL
# ══════════════════════════════════════════════════════════════════════════════

_SPARQL_SELECT = """\
PREFIX bcnnorms: <http://datos.bcn.cl/ontologies/bcn-norms#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?leychileCode ?titulo ?numero ?fechaPublicacion ?documento
WHERE {{
  ?norma bcnnorms:leychileCode ?leychileCode .
  ?norma dc:title ?titulo .
  OPTIONAL {{ ?norma bcnnorms:hasNumber ?numero }}
  OPTIONAL {{ ?norma bcnnorms:publishDate ?fechaPublicacion }}
  OPTIONAL {{ ?norma bcnnorms:hasHtmlDocument ?documento }}
  FILTER(
    {filter_body}
  )
}}
ORDER BY DESC(?fechaPublicacion)"""


def _armar_query(condiciones: list[str]) -> str:
    return _SPARQL_SELECT.format(filter_body=" &&\n    ".join(condiciones))


def construir_query_determinista(
    denominacion: str,
    fecha: str,
    comuna: str,
    numero_doc: str,
) -> str | None:
    """
    Construye query SPARQL sin LLM usando datos estructurados del CSV.

    Dos casos:
      • Con número de decreto: ?numero + fecha exacta — discriminador máximo, sin título.
      • Sin número: commune + 1-2 palabras del título + fecha.

    Devuelve None si no hay suficiente información para construir algo útil.
    """
    c_norm  = _quitar_acentos(comuna.upper())
    excluir = _GENERICAS | {c_norm}

    n_clean = _normalizar_numero(numero_doc) if numero_doc else ""
    if _numero_valido(n_clean):
        # Número + commune + fecha: commune desambigua números cortos (ej. "4")
        conds = [
            f'CONTAINS(UCASE(?titulo), "{c_norm}")',
            f'CONTAINS(UCASE(?numero), "{n_clean}")',
            f'?fechaPublicacion = "{fecha}"^^xsd:date',
        ]
    else:
        # Sin número: commune + palabras clave del título + fecha
        conds = [f'CONTAINS(UCASE(?titulo), "{c_norm}")']
        palabras = _palabras_clave(denominacion, excluir, max_n=2)
        if not palabras:
            return None  # título demasiado genérico → sin info suficiente
        for p in palabras:
            conds.append(f'CONTAINS(UCASE(?titulo), "{p}")')
        conds.append(f'?fechaPublicacion = "{fecha}"^^xsd:date')

    return _armar_query(conds)


def construir_query_sin_fecha(
    denominacion: str,
    fecha: str,
    comuna: str,
    numero_doc: str = "",
) -> str | None:
    """
    Track 2 — igual que Track 1 pero SIN filtro de fecha.
    Amplía la búsqueda cuando Track 1 falla (p.ej. discrepancia de fecha en BCN).
    0 LLM, 0 coste.
    """
    c_norm  = _quitar_acentos(comuna.upper())
    excluir = _GENERICAS | {c_norm}

    n_clean = _normalizar_numero(numero_doc) if numero_doc else ""
    if _numero_valido(n_clean):
        conds = [
            f'CONTAINS(UCASE(?titulo), "{c_norm}")',
            f'CONTAINS(UCASE(?numero), "{n_clean}")',
        ]
    else:
        conds = [f'CONTAINS(UCASE(?titulo), "{c_norm}")']
        palabras = _palabras_clave(denominacion, excluir, max_n=2)
        if not palabras:
            return None
        for p in palabras:
            conds.append(f'CONTAINS(UCASE(?titulo), "{p}")')

    return _armar_query(conds)


# ══════════════════════════════════════════════════════════════════════════════
# SPARQL — EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_query_sparql(query_text: str, max_retries: int = 3, initial_timeout: int = 15) -> list:
    """Ejecuta una query SPARQL con reintentos y backoff exponencial."""
    for attempt in range(max_retries):
        try:
            sparql = SPARQLWrapper(SPARQL_ENDPOINT)
            sparql.setTimeout(initial_timeout + attempt * 5)
            sparql.setQuery(query_text)
            sparql.setReturnFormat(JSON)
            return sparql.query().convert().get("results", {}).get("bindings", [])
        except (URLError, TimeoutError, Exception) as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * 2
                print(f"  ⚠️  Error intento {attempt+1}/{max_retries}: {type(e).__name__} — reintentando en {wait}s…")
                time.sleep(wait)
            else:
                print(f"  ❌  Error tras {max_retries} intentos: {type(e).__name__}: {str(e)[:120]}")
                return []
    return []


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES CSV / DEDUPLICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def leer_comunas_interes() -> list[str]:
    if not COMUNAS_INTERES_FILE.exists():
        print(f"❌ No se encontró {COMUNAS_INTERES_FILE}")
        return []
    return [
        line.strip() for line in
        COMUNAS_INTERES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def leer_todas_comunas_csv(df: pd.DataFrame) -> list[str]:
    """
    Extrae todas las comunas únicas de la columna 'comunas' del CSV de entrada.
    Devuelve lista ordenada alfabéticamente.
    """
    if "comunas" not in df.columns:
        print("❌ Columna 'comunas' no encontrada en el CSV")
        return []
    valores = df["comunas"].dropna().str.strip().str.upper().unique()
    comunas = sorted([v for v in valores if v])
    return comunas


def leer_csv_existente() -> pd.DataFrame:
    return pd.read_csv(CSV_OUTPUT, encoding="utf-8") if CSV_OUTPUT.exists() else pd.DataFrame()


def comuna_ya_procesada(comuna: str, df: pd.DataFrame) -> bool:
    if df.empty or "comuna" not in df.columns:
        return False
    return comuna.upper() in df["comuna"].str.upper().values


def _extraer_valor(binding, campo: str) -> str:
    v = binding.get(campo)
    if v is None:
        return ""
    return v.get("value", "") if isinstance(v, dict) else (str(v) if v else "")


def construir_url_minvu(row: pd.Series | None = None) -> str | None:
    """
    Extrae la URL de MINVU desde la columna 'urls_documentos'.
    
    Busca la primera URL que contenga 'instrumentosdeplanificacion.minvu.cl' en urls_documentos
    y devuelve esa URL completa (no solo la base).
    
    Si no encuentra ninguna URL de MINVU válida en urls_documentos, devuelve None.
    NO usa fallbacks con código ni URL base genérica.
    
    Args:
        row: Serie de pandas con datos del registro del CSV (opcional)
        
    Returns:
        URL completa de MINVU si se encuentra, None si no hay URL válida
    """
    if row is None:
        return None
    
    # Extraer URL de MINVU desde urls_documentos
    urls_documentos = str(row.get("urls_documentos", "")).strip()
    if urls_documentos and urls_documentos.lower() not in ("nan", ""):
        # Separar URLs por ' | '
        urls = [url.strip() for url in urls_documentos.split(" | ") if url.strip()]
        # Buscar la primera URL que contenga el dominio de MINVU
        for url in urls:
            if "instrumentosdeplanificacion.minvu.cl" in url:
                # Devolver la URL completa del documento
                return url
    
    # No hay URL válida de MINVU
    return None


def bindings_a_registros(bindings: list, comuna: str, row_csv: pd.Series = None) -> list[dict]:
    """
    Convierte bindings SPARQL a registros.
    Si hay URL de MINVU válida, crea registros incluso si no hay URL de BCN.
    """
    out = []
    # Construir URL de MINVU una vez para todos los bindings del mismo registro
    url_minvu = construir_url_minvu(row_csv)
    
    # Si no hay URL de MINVU válida, no procesar ningún binding
    if url_minvu is None:
        return []
    
    for b in bindings:
        fecha = _extraer_valor(b, "fechaPublicacion")
        if "T" in fecha:
            fecha = fecha.split("T")[0]
        leychile_code = _extraer_valor(b, "leychileCode")
        numero        = _extraer_valor(b, "numero")
        # Construir URL directa de descarga PDF desde leychile.cl
        url_pdf = construir_url_pdf(leychile_code, numero, fecha)
        
        # Agregar registro incluso si no tiene URL de BCN (pero sí tiene url_minvu)
        out.append({
            "comuna":           comuna,
            "titulo":           _extraer_valor(b, "titulo"),
            "leychileCode":     leychile_code,
            "numero":           numero,
            "fechaPublicacion": fecha,
            "url":              url_pdf if (url_pdf and url_pdf.startswith("http")) else "",
            "url_minvu":        url_minvu,
        })
    return out


def _filtrar_por_fecha(bindings: list, fecha_csv: str, comuna: str, row_csv: pd.Series = None) -> list[dict]:
    """Convierte bindings SPARQL a registros y filtra por fecha exacta."""
    registros = bindings_a_registros(bindings, comuna, row_csv)
    return deduplicar([r for r in registros if r["fechaPublicacion"][:10] == fecha_csv])


def deduplicar(registros: list) -> list:
    """
    Deduplica registros. Si tienen leychileCode, agrupa por ese.
    Si no, agrupa por comuna + titulo + fecha.
    Prioriza registros con ambas URLs válidas, pero acepta solo url_minvu.
    """
    grupos: dict[str, list] = {}
    for r in registros:
        # Usar leychileCode como clave si existe, sino usar comuna+titulo+fecha
        if r.get("leychileCode") and str(r["leychileCode"]).strip():
            key = str(r["leychileCode"])
        else:
            key = f"{r.get('comuna', '')}_{r.get('titulo', '')}_{r.get('fechaPublicacion', '')}"
        grupos.setdefault(key, []).append(r)
    
    out = []
    for grupo in grupos.values():
        # Priorizar registros con ambas URLs válidas
        con_ambas_urls = [
            r for r in grupo 
            if r.get("fechaPublicacion") 
            and r.get("url") and r["url"].startswith("http")
            and r.get("url_minvu") and r["url_minvu"].startswith("http")
        ]
        if con_ambas_urls:
            out.append(con_ambas_urls[0])
        # Si no hay con ambas URLs, buscar con url_minvu válida (aunque no tenga url de BCN)
        else:
            con_minvu = [
                r for r in grupo 
                if r.get("url_minvu") and r["url_minvu"].startswith("http")
            ]
            if con_minvu:
                # Priorizar los que tienen fecha
                con_fecha = [r for r in con_minvu if r.get("fechaPublicacion")]
                out.append(con_fecha[0] if con_fecha else con_minvu[0])
    return out



def guardar_csv(df: pd.DataFrame) -> None:
    cols = ["comuna", "titulo", "leychileCode", "numero", "fechaPublicacion", "url", "url_minvu"]
    if df.empty:
        pd.DataFrame(columns=cols).to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
        print(f"✓ CSV guardado: 0 registros → {CSV_OUTPUT.name}")
        return
    cols_ok = [c for c in cols if c in df.columns]
    df_out  = df[cols_ok].copy()
    # No filtrar por fecha - aceptar registros con url_minvu aunque no tengan fecha
    # Solo mantener registros que tienen url_minvu válida
    if "url_minvu" in df_out.columns:
        df_out = df_out[df_out["url_minvu"].notna() & (df_out["url_minvu"].astype(str).str.strip() != "")]
    orden = [c for c in ["comuna", "fechaPublicacion", "titulo"] if c in df_out.columns]
    if orden:
        df_out = df_out.sort_values(by=orden, ascending=[True, False, True])
    df_out.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"✓ CSV guardado: {len(df_out)} registros → {CSV_OUTPUT.name}")


def agregar_al_df(nuevos: list, df_existente: pd.DataFrame) -> pd.DataFrame:
    if not nuevos:
        return df_existente
    df_n = pd.DataFrame(nuevos)
    if df_existente.empty:
        return df_n
    df_c = pd.concat([df_existente, df_n], ignore_index=True)
    
    # Ordenar por leychileCode si existe, sino por otros campos
    sort_cols = []
    if "leychileCode" in df_c.columns:
        sort_cols.append("leychileCode")
    if "fechaPublicacion" in df_c.columns:
        sort_cols.append("fechaPublicacion")
    if "url" in df_c.columns:
        sort_cols.append("url")
    
    if sort_cols:
        df_c = df_c.sort_values(
            by=sort_cols,
            ascending=[True] * len(sort_cols),
            na_position="last",
        )
    
    # Deduplicar: si hay leychileCode, usar ese; sino usar comuna+titulo+fecha
    if "leychileCode" in df_c.columns:
        # Para registros con leychileCode, deduplicar por ese campo
        con_codigo = df_c[df_c["leychileCode"].notna() & (df_c["leychileCode"].astype(str).str.strip() != "")]
        sin_codigo = df_c[df_c["leychileCode"].isna() | (df_c["leychileCode"].astype(str).str.strip() == "")]
        
        con_codigo_dedup = con_codigo.drop_duplicates(subset=["leychileCode"], keep="first")
        
        # Para registros sin código, deduplicar por comuna+titulo+fecha
        if not sin_codigo.empty and all(col in sin_codigo.columns for col in ["comuna", "titulo", "fechaPublicacion"]):
            sin_codigo_dedup = sin_codigo.drop_duplicates(subset=["comuna", "titulo", "fechaPublicacion"], keep="first")
            df_c = pd.concat([con_codigo_dedup, sin_codigo_dedup], ignore_index=True)
        else:
            df_c = con_codigo_dedup if not con_codigo.empty else sin_codigo
    else:
        # Si no hay columna leychileCode, deduplicar por comuna+titulo+fecha
        if all(col in df_c.columns for col in ["comuna", "titulo", "fechaPublicacion"]):
            df_c = df_c.drop_duplicates(subset=["comuna", "titulo", "fechaPublicacion"], keep="first")
    
    return df_c


# ══════════════════════════════════════════════════════════════════════════════
# LOG
# ══════════════════════════════════════════════════════════════════════════════

def _log(texto: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(texto + "\n")
        f.flush()


def log_inicio(comunas: list, total: int) -> None:
    _log("\n" + "#" * 80)
    _log(f"INICIO  {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"Comunas: {', '.join(comunas)}")
    _log(f"Registros CSV entrada: {total}")
    _log("#" * 80 + "\n")


def log_fin(total: int) -> None:
    _log("\n" + "#" * 80)
    _log(f"FIN     {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"Total registros CSV salida: {total}")
    _log("#" * 80 + "\n")


def log_registro(
    materia: str,
    fecha: str,
    track: str,
    query_det: str | None,
    query_llm: str | None,
    resultados: list,
    fuente: str,
) -> None:
    _log("=" * 80)
    _log(f"HORA:    {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"MATERIA: {materia}")
    _log(f"FECHA:   {fecha}   TRACK: {track}   FUENTE: {fuente}")
    if query_det:
        _log(f"QUERY DETERMINISTA:\n{query_det}\n")
    if query_llm:
        _log(f"QUERY LLM:\n{query_llm}\n")
    _log(f"RESULTADOS: {len(resultados)}")
    for i, b in enumerate(resultados, 1):
        _log(f"  {i}. code={_extraer_valor(b,'leychileCode')}  "
             f"fecha={_extraer_valor(b,'fechaPublicacion')}  "
             f"titulo={_extraer_valor(b,'titulo')[:55]}")
    _log("=" * 80 + "\n")


def _indent(text: str, prefix: str = "     ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE REGISTROS
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar_fecha(valor: str) -> str:
    """Devuelve YYYY-MM-DD o '' si no es válida."""
    s = str(valor or "").strip()
    if s.lower() in ("nan", "sin información", "sin informacion", ""):
        return ""
    if "T" in s:
        s = s.split("T")[0]
    return s[:10]


def procesar_registro(row: pd.Series, comuna: str) -> list[dict]:
    """
    Aplica los 3 tracks para un único registro del CSV.
    Si no encuentra resultados en BCN pero tiene URL de MINVU, crea un registro solo con url_minvu.
    Devuelve lista con 0 ó 1 dict con el resultado.
    """
    denominacion = str(row.get("denominacion", "")).strip()
    fecha        = (_normalizar_fecha(str(row.get("fechaInicioVigencia", "")))
                    or _normalizar_fecha(str(row.get("fechaPublicacion", ""))))
    numero_doc   = str(row.get("numeroDocumento", "")).strip()

    if not denominacion or denominacion.lower() == "nan":
        return []

    # Verificar si hay URL de MINVU válida
    url_minvu = construir_url_minvu(row)
    
    # Sin fecha no podemos validar ningún resultado SPARQL
    if not fecha:
        # Pero si hay URL de MINVU, crear registro sin URL de BCN
        if url_minvu:
            print(f"  ⚠️   Sin fecha, pero guardando con url_minvu")
            return [{
                "comuna":           comuna,
                "titulo":           denominacion,
                "leychileCode":     "",
                "numero":           numero_doc,
                "fechaPublicacion": "",
                "url":              "",
                "url_minvu":        url_minvu,
            }]
        print(f"  —   Sin fecha, omitiendo")
        return []

    query_det  = None
    query_llm  = None
    resultados = []
    track      = ""

    # ── Track 1: SPARQL determinista ──────────────────────────────────────
    query_det = construir_query_determinista(denominacion, fecha, comuna, numero_doc)
    if query_det:
        n_norm  = _normalizar_numero(numero_doc) if numero_doc else ""
        n_label = f"{n_norm} (de {numero_doc})" if n_norm != numero_doc.upper().strip() and n_norm else n_norm or "palabras"
        print(f"  🔢  Track 1 | discriminador: {n_label}")
        print(_indent(query_det))
        resultados = ejecutar_query_sparql(query_det)
        match = _filtrar_por_fecha(resultados, fecha, comuna, row)
        if match:
            track = "determinista"
            log_registro(denominacion, fecha, track, query_det, None, resultados, "BCN")
            print(f"  ✓   Track 1 OK")
            return match

    # ── Track 2: SPARQL sin fecha (0 LLM) ────────────────────────────────
    print(f"  🔎  Track 2 | sin fecha")
    query_llm = construir_query_sin_fecha(denominacion, fecha, comuna, numero_doc)
    if query_llm:
        print(_indent(query_llm))
        resultados = ejecutar_query_sparql(query_llm)
        match = _filtrar_por_fecha(resultados, fecha, comuna, row)
        if match:
            track = "sin-fecha"
            log_registro(denominacion, fecha, track, query_det, query_llm, resultados, "BCN")
            print(f"  ✓   Track 2 OK")
            return match

    # Si no hay resultados en BCN pero sí hay URL de MINVU, crear registro solo con url_minvu
    if url_minvu:
        print(f"  ⚠️   Sin resultado en BCN, pero guardando con url_minvu")
        log_registro(denominacion, fecha, "sin-resultado-bcn", query_det, query_llm, [], "MINVU")
        return [{
            "comuna":           comuna,
            "titulo":           denominacion,
            "leychileCode":     "",
            "numero":           numero_doc,
            "fechaPublicacion": fecha,
            "url":              "",
            "url_minvu":        url_minvu,
        }]

    log_registro(denominacion, fecha, "sin-resultado", query_det, query_llm, [], "—")
    print(f"  —   Sin resultado")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO POR COMUNA
# ══════════════════════════════════════════════════════════════════════════════

def procesar_comuna(
    comuna: str,
    df_input: pd.DataFrame,
    df_existente: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"🏙️   COMUNA: {comuna}")
    print(f"{'='*80}")

    mask       = df_input["comunas"].str.strip().str.upper() == comuna.upper()
    df_filtrado = df_input[mask]
    print(f"✓ Registros encontrados: {len(df_filtrado)}")

    if df_filtrado.empty:
        print("⚠️  Sin registros para esta comuna.")
        return df_existente

    df_actual = df_existente.copy()
    total_registros = 0

    for idx, row in df_filtrado.iterrows():
        denominacion  = str(row.get("denominacion", "")).strip()
        clasificacion = str(row.get("clasificacion", "")).strip()
        numero_doc    = str(row.get("numeroDocumento", "")).strip()
        fecha         = (_normalizar_fecha(str(row.get("fechaInicioVigencia", "")))
                         or _normalizar_fecha(str(row.get("fechaPublicacion", ""))))

        print(f"\n  [{idx}] {clasificacion} | {denominacion[:70]}")
        print(f"         fecha={fecha or '—'}  numero={numero_doc or '—'}")

        resultado = procesar_registro(row, comuna)
        if resultado:
            # Filtrar registros con url_minvu válida (puede no tener url de BCN)
            resultado_valido = [
                r for r in resultado 
                if r.get("url_minvu") and r.get("url_minvu").startswith("http")
            ]
            
            if resultado_valido:
                print(f"  ✓   Encontrado {len(resultado_valido)} registro(s) con url_minvu")
                # Deduplicar antes de agregar
                resultado_dedup = deduplicar(resultado_valido)
                # Agregar al DataFrame actual
                df_actual = agregar_al_df(resultado_dedup, df_actual)
                # Guardar inmediatamente en CSV
                guardar_csv(df_actual)
                total_registros += len(resultado_dedup)
            else:
                print(f"  ⚠️  Sin url_minvu válida, omitiendo")
        else:
            print(f"  —   Sin resultado")

    print(f"\n📊 {comuna}: {total_registros} registros guardados")
    _log(f"\n{'-'*80}\nCOMUNA: {comuna}  CSV:{len(df_filtrado)}  guardados:{total_registros}\n{'-'*80}\n")

    return df_actual


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca documentos de planes reguladores en BCN vía SPARQL.",
    )
    parser.add_argument(
        "--todas", "--all",
        action="store_true",
        dest="todas",
        help="Procesar TODAS las comunas presentes en planes_origen_modificaciones.csv "
             "(ignora comunas_interes.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🚀 BCN SPARQL — planes_origen_modificaciones")
    print("=" * 80)

    if not CSV_INPUT.exists():
        print(f"❌ No se encontró {CSV_INPUT}")
        return
    df_input = pd.read_csv(CSV_INPUT, sep=";", encoding="utf-8", dtype=str)
    df_input.columns = df_input.columns.str.strip()
    print(f"✓ CSV entrada: {len(df_input)} registros  |  cols: {', '.join(df_input.columns[:8])}…")

    requeridas = ["comunas", "denominacion", "fechaInicioVigencia",
                  "numeroDocumento", "urls_documentos"]
    faltantes  = [c for c in requeridas if c not in df_input.columns]
    if faltantes:
        print(f"❌ Faltan columnas: {', '.join(faltantes)}")
        print(f"   Disponibles: {', '.join(df_input.columns)}")
        return

    if args.todas:
        comunas = leer_todas_comunas_csv(df_input)
        if not comunas:
            print("❌ No se encontraron comunas en el CSV")
            return
        print(f"✓ Modo --todas: {len(comunas)} comunas encontradas en el CSV")
    else:
        comunas = leer_comunas_interes()
        if not comunas:
            print("❌ comunas_interes.txt vacío o no encontrado")
            return
    print(f"✓ Comunas a procesar: {', '.join(comunas)}")

    log_inicio(comunas, len(df_input))

    df_existente = leer_csv_existente()
    print(f"✓ CSV salida: {'nuevo archivo' if df_existente.empty else f'{len(df_existente)} registros existentes'}")

    for comuna in comunas:
        if comuna_ya_procesada(comuna, df_existente):
            print(f"\n⏭️  {comuna} ya procesada, saltando…")
            continue
        df_existente = procesar_comuna(comuna, df_input, df_existente)

    print(f"\n✅ Completado  |  {len(df_existente)} registros  →  {CSV_OUTPUT}")
    log_fin(len(df_existente))


if __name__ == "__main__":
    main()
