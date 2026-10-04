from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database import MedicalSynonym

logger = logging.getLogger(__name__)
settings = get_settings()

# Semilla: cada grupo es una idea/concepto con sus formas de escribirlo.
# canonical = forma principal; variants = todas las formas aceptadas (incluye canonical).
SEED_SYNONYMS: dict[str, dict] = {
    "identificacion": {
        "DNI": [
            "DNI",
            "D.N.I.",
            "DNI.",
            "documento nacional de identidad",
            "documento de identidad",
            "num. de documento",
            "nro. de documento",
            "n° documento",
            "numero de documento",
            "carnet de extranjeria",
            "carnet de extranjería",
            "CE",
            "C.E.",
        ],
        "CUIL": [
            "CUIL",
            "C.U.I.L.",
            "codigo unico de identificacion laboral",
        ],
        "CUIT": [
            "CUIT",
            "C.U.I.T.",
            "codigo unico de identificacion tributaria",
        ],
        "pasaporte": [
            "pasaporte",
            "pasap.",
            "passaporte",
        ],
        "licencia de conducir": [
            "licencia de conducir",
            "licencia",
            "registro de conducir",
            "carnet de conducir",
        ],
    },
    "hc": {
        "historia clinica": [
            "historia clinica",
            "historia clinica",
            "historia medica",
            "historia clinica electronica",
            "H.C.",
            "HC",
            "expediente clinico",
            "carpeta de historia clinica",
            "historia",
            "historial",
            "historial clinico",
        ],
        "diagnostico": [
            "diagnostico",
            "dx",
            " Dx",
            "diagnosticos",
            "impresion diagnostica",
            "diagnostico presuntivo",
            "diagnostico definitivo",
        ],
        "antecedente": [
            "antecedente",
            "antecedentes",
            "ant.",
            "antecedentes personales",
            "antecedentes familiares",
        ],
        "alergia": [
            "alergia",
            "alergias",
            "alerjico",
            "alergico",
            "hipersensibilidad",
        ],
        "indicacion": [
            "indicacion",
            "indicaciones",
            "ind.",
            "indicado",
        ],
        "evolucion": [
            "evolucion",
            "evol.",
            "evoluciones",
            "se evoluciona",
        ],
    },
    "medicamentos": {
        "medicamento": [
            "medicamento",
            "medicamentos",
            "medicacion",
            "medicación",
            "farmaco",
            "fármaco",
            "droga",
            "producto medicamentoso",
        ],
        "dosis": [
            "dosis",
            "dosis diaria",
            "posologia",
            "dosis a administrar",
        ],
        "via de administracion": [
            "via de administracion",
            "via",
            "viaa",
            "modo de administracion",
        ],
        "receta": [
            "receta",
            "recetario",
            "orden medica",
            "prescripcion",
            "formula",
        ],
    },
    "signos": {
        "frecuencia cardiaca": [
            "frecuencia cardiaca",
            "fc",
            "pulso",
            "latidos",
        ],
        "presion arterial": [
            "presion arterial",
            "pa",
            "tension arterial",
            "ta",
        ],
        "temperatura": [
            "temperatura",
            "temp",
            "t",
            "termometria",
        ],
        "saturacion de oxigeno": [
            "saturacion de oxigeno",
            "sato2",
            "spo2",
            "saturacion",
        ],
        "peso": [
            "peso",
            "peso corporal",
            "kg",
        ],
        "talla": [
            "talla",
            "altura",
        ],
        "imc": [
            "imc",
            "indice de masa corporal",
            "indice de masa",
        ],
    },
    "paciente": {
        "paciente": [
            "paciente",
            "usuario",
            "afiliado",
            "beneficiario",
            "cliente",
        ],
        "nombre": [
            "nombre",
            "nombres",
            "apellido y nombre",
            "apellido y nombres",
            "apellidos y nombres",
        ],
        "fecha de nacimiento": [
            "fecha de nacimiento",
            "f. nacimiento",
            "fnac",
            "fecha nac",
            "nacimiento",
        ],
        "edad": [
            "edad",
            "anos",
            "años",
        ],
        "sexo": [
            "sexo",
            "genero",
            "gender",
        ],
        "domicilio": [
            "domicilio",
            "direccion",
            "dir.",
            "domicilio actual",
            "domicilio particular",
        ],
        "telefono": [
            "telefono",
            "tel.",
            "tel",
            "celular",
            "contacto telefonico",
        ],
    },
    "estudios": {
        "laboratorio": [
            "laboratorio",
            "lab.",
            "analisis clinico",
            "estudio de laboratorio",
        ],
        "radiografia": [
            "radiografia",
            "radiografia",
            "rx",
            "placa",
            "radiologico",
        ],
        "ecografia": [
            "ecografia",
            "ecotomografia",
            "ultrasonido",
            "eco",
        ],
        "tomografia": [
            "tomografia",
            "tc",
            "tac",
            "tomografo",
        ],
        "resonancia magnetica": [
            "resonancia magnetica",
            "rm",
            "rmn",
            "resonancia",
        ],
        "electrocardiograma": [
            "electrocardiograma",
            "ecg",
            "ekg",
        ],
    },
    "internacion": {
        "internacion": [
            "internacion",
            "internado",
            "ingreso",
            "admision hospitalaria",
            "hospitalizacion",
        ],
        "alta medica": [
            "alta medica",
            "alta",
            "egreso",
            "alta hospitalaria",
        ],
        "dieta": [
            "dieta",
            "régimen",
            "regimen alimentario",
        ],
        "signos vitales": [
            "signos vitales",
            "sv",
            "signos",
        ],
        "estado general": [
            "estado general",
            "eg",
            "estado clinico",
        ],
    },
}


def normalize_term(s: str) -> str:
    """minúsculas, sin acentos, sin puntuación, espacios colapsados."""
    if not s:
        return ""
    t = unicodedata.normalize("NFD", str(s))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"_", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compact_term(s: str) -> str:
    """normalize + sin espacios (para 'd n i' → 'dni')."""
    return normalize_term(s).replace(" ", "")


def build_seed_rows() -> list[tuple[str, str, str]]:
    """[(canonical, synonym, category), ...] a partir de SEED_SYNONYMS."""
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, groups in SEED_SYNONYMS.items():
        for canonical, variants in groups.items():
            all_forms = [canonical, *variants]
            for v in all_forms:
                v = str(v).strip()
                if not v:
                    continue
                key = (canonical, v)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((canonical, v, category))
            # también enlazar cada variante ↔ canonical para búsqueda inversa
            for v in variants:
                v = str(v).strip()
                if not v or v == canonical:
                    continue
                key = (v, canonical)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((v, canonical, category))
    return rows


async def seed_synonyms(db: AsyncSession) -> int:
    """Inserta semilla en medical_synonyms si aún no existe ese par."""
    rows = build_seed_rows()
    count = 0
    existing = set()
    result = await db.execute(
        select(MedicalSynonym.canonical, MedicalSynonym.synonym)
    )
    for c, s in result.all():
        existing.add((c, s))
    for canonical, synonym, category in rows:
        if (canonical, synonym) in existing:
            continue
        db.add(MedicalSynonym(canonical=canonical, synonym=synonym, category=category))
        count += 1
    if count:
        await db.commit()
        logger.info(f"Seeded {count} synonym rows")
    else:
        logger.info("Synonym seed already present")
    return count


def get_seed_groups() -> list[dict]:
    """Grupos listos para la API (solo semilla, sin DB)."""
    groups = []
    for category, cat_groups in SEED_SYNONYMS.items():
        for canonical, variants in cat_groups.items():
            forms: list[str] = []
            seen: set[str] = set()
            for v in [canonical, *variants]:
                n = normalize_term(v)
                if n and n not in seen:
                    seen.add(n)
                    forms.append(v)
            groups.append({
                "canonical": canonical,
                "category": category,
                "variants": forms,
            })
    return groups


async def get_groups_from_db(db: AsyncSession) -> list[dict] | None:
    """Si la tabla tiene datos, agrupa por canonical."""
    result = await db.execute(select(MedicalSynonym))
    rows = result.scalars().all()
    if not rows:
        return None
    by_can: dict[str, dict] = {}
    for r in rows:
        cat = r.category or "general"
        g = by_can.setdefault(r.canonical, {
            "canonical": r.canonical,
            "category": cat,
            "variants": [],
        })
        seen = {normalize_term(v) for v in g["variants"]}
        n = normalize_term(r.synonym)
        if n not in seen:
            g["variants"].append(r.synonym)
    return list(by_can.values())


async def get_synonym_groups(db: AsyncSession | None = None) -> list[dict]:
    if db is not None:
        try:
            db_groups = await get_groups_from_db(db)
            if db_groups:
                return db_groups
        except Exception as e:
            logger.warning(f"DB synonym load failed, using seed: {e}")
    return get_seed_groups()


def expand_term(term: str, groups: list[dict] | None = None) -> list[str]:
    """Devuelve todas las formas normalizadas equivalentes a term."""
    if groups is None:
        groups = get_seed_groups()
    n = normalize_term(term)
    c = compact_term(term)
    if not n:
        return []
    variants: set[str] = {n}
    if c:
        variants.add(c)
    for g in groups:
        g_forms = [g["canonical"], *g["variants"]]
        g_norms = {normalize_term(v) for v in g_forms}
        g_compact = {compact_term(v) for v in g_forms}
        if n in g_norms or (c and c in g_compact):
            for v in g_forms:
                vn = normalize_term(v)
                if vn:
                    variants.add(vn)
                vc = compact_term(v)
                if vc:
                    variants.add(vc)
    # también: si term es subcadena de un canonical corto (ej "dni" en "documento...")
    # no ampliar por substring sola para evitar ruido; solo grupos explícitos.
    return sorted(variants)


async def load_synonyms_from_xlsx(db: AsyncSession) -> int:
    xlsx_path = Path(os.getenv("SYNONYM_XLSX_PATH", "data/palabras clave.xlsx"))
    if not xlsx_path.exists():
        logger.info("palabras clave.xlsx not found, skipping synonym import")
        return 0

    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
        count = 0
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row and row[0]:
                    canonical = str(row[0]).strip()
                    synonym = str(row[1]).strip() if len(row) > 1 and row[1] else canonical
                    if canonical:
                        existing = await db.execute(
                            select(MedicalSynonym).where(
                                MedicalSynonym.canonical == canonical,
                                MedicalSynonym.synonym == synonym,
                            )
                        )
                        if not existing.scalar_one_or_none():
                            db.add(MedicalSynonym(
                                canonical=canonical,
                                synonym=synonym,
                                category=sheet_name,
                            ))
                            count += 1
        await db.commit()
        wb.close()
        logger.info(f"Loaded {count} synonyms from xlsx")
        return count
    except Exception as e:
        logger.error(f"Failed to load synonyms: {e}")
        return 0


async def get_synonyms_for_query(query: str, db: AsyncSession) -> list[str]:
    groups = await get_synonym_groups(db)
    n = normalize_term(query)
    extras: list[str] = []
    for g in groups:
        forms = [g["canonical"], *g["variants"]]
        norms = {normalize_term(v) for v in forms}
        if n in norms or any(x in n for x in norms if len(x) >= 4):
            for v in forms:
                if normalize_term(v) != n:
                    extras.append(v)
    return extras
