"""Medical entity extraction with deterministic Spanish patterns and optional spaCy."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)
_nlp = None

MEDICAL_ENTITY_PATTERNS = {
    "MEDICATION": [
        r"\b(paracetamol|ibuprofeno|amoxicilina|metformina|losartan|atorvastatina|"
        r"omeprazol|azitromicina|ceftriaxona|dexametasona|prednisona|insulina|"
        r"enrofloxacina|doxiciclina|clindamicina|metronidazol|diclofenaco|naproxeno|"
        r"aspirina|clopidogrel|warfarina|heparina|salbutamol|fluticasona|montelukast|"
        r"levotiroxina|metoprolol|amlodipino|nifedipino|furosemida|espironolactona|"
        r"sertralina|fluoxetina|duloxetina|cefalexina)\b",
    ],
    "DIAGNOSIS": [
        r"\b(hipertensi[o\u00f3]n|diabetes|neumon[i\u00eda]|bronquitis|asma|"
        r"insuficiencia card[i\u00ed]aca|arritmia|infarto|ictus|gastroenteritis|"
        r"pancreatitis|colelitiasis|anemia|leucemia|linfoma|nefropat[i\u00eda]|"
        r"hepatopat[i\u00eda]|COVID[-\s]?19|gripe|resfriado|hipotiroidismo|"
        r"hipertiroidismo|obesidad|sobrepeso|depresi[o\u00f3]n|ansiedad|lumbalgia|"
        r"cervicalgia|cistitis|pielonefritis)\b",
    ],
    "PROCEDURE": [
        r"\b(radiograf[i\u00eda]|ecograf[i\u00eda]|tomograf[i\u00eda]|resonancia|"
        r"endoscopia|colonoscopia|biopsia|ecocardiograma|electrocardiograma|"
        r"an[a\u00e1]lisis de sangre|hemograma|punci[o\u00f3]n lumbar|broncoscopia|"
        r"cirug[i\u00eda]|operaci[o\u00f3]n|intervenci[o\u00f3]n|cateterismo|"
        r"angiograf[i\u00eda])\b",
    ],
    "VITAL_SIGN": [
        r"\b(presi[o\u00f3]n arterial|tasa card[i\u00ed]aca|frecuencia card[i\u00ed]aca|"
        r"temperatura corporal|saturaci[o\u00f3]n de ox[i\u00ed]geno|peso|talla|IMC|"
        r"[i\u00ed]ndice de masa corporal|glucemia|colesterol|triglic[e\u00e9]ridos|"
        r"creatinina|urea|[a\u00e1]cido [u\u00far]ico)\b",
    ],
    "CLINICAL_VALUE": [
        r"\b(?:presi[o\u00f3]n arterial|glucosa|glucemia|temperatura|saturaci[o\u00f3]n|"
        r"frecuencia card[i\u00ed]aca|peso|talla|IMC|creatinina|hemoglobina)"
        r"\s*[:=]?\s*\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?"
        r"\s*(?:mmHg|mg/dL|g/dL|\u00b0C|%|kg|cm|lpm)?\b",
    ],
    "BODY_PART": [
        r"\b(cabeza|cuello|t[o\u00f3]rax|abdomen|pelvis|pulm[o\u00f3]n|coraz[o\u00f3]n|"
        r"h[i\u00ed]gado|ri\u00f1[o\u00f3]n|bazo|p[a\u00e1]ncreas|intestino|est[o\u00f3]mago|"
        r"columna vertebral|articulaci[o\u00f3]n|miembro superior|miembro inferior|"
        r"extremidad|cerebro|hueso|m[u\u00fa]sculo)\b",
    ],
}


@dataclass
class MedicalEntity:
    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float = 0.85
    icd10_code: str | None = None
    atc_code: str | None = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            try:
                _nlp = spacy.load("es_core_news_sm")
            except OSError:
                logger.info("spaCy Spanish model not found; using deterministic patterns")
                _nlp = spacy.blank("es")
        except Exception as exc:
            logger.info("spaCy unavailable: %s", exc)
            _nlp = None
    return _nlp


def extract_entities_regex(text: str) -> list[MedicalEntity]:
    entities = []
    for entity_type, patterns in MEDICAL_ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    MedicalEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )
    return entities


def extract_entities_spacy(text: str) -> list[MedicalEntity]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    return [
        MedicalEntity(ent.text, _map_spacy_label(ent.label_), ent.start_char, ent.end_char, 0.80)
        for ent in nlp(text).ents
        if ent.label_ not in {"PER", "PERSON"}
    ]


def _map_spacy_label(label: str) -> str:
    return {
        "LOC": "BODY_PART",
        "ORG": "INSTITUTION",
        "MISC": "MEDICAL_TERM",
        "DATE": "DATE",
        "NUM": "VALUE",
    }.get(label, "OTHER")


def extract_medical_entities(text: str) -> list[MedicalEntity]:
    entities = extract_entities_regex(text) + extract_entities_spacy(text)
    return _deduplicate_entities(sorted(entities, key=lambda entity: (entity.start_char, -entity.end_char)))


def _deduplicate_entities(entities: list[MedicalEntity]) -> list[MedicalEntity]:
    deduped: list[MedicalEntity] = []
    for entity in entities:
        overlaps = [item for item in deduped if entity.start_char < item.end_char and entity.end_char > item.start_char]
        if not overlaps:
            deduped.append(entity)
        elif entity.confidence > max(item.confidence for item in overlaps):
            deduped = [item for item in deduped if item not in overlaps]
            deduped.append(entity)
    return sorted(deduped, key=lambda entity: entity.start_char)


async def list_medical_entities(
    db,
    entity_type: str | None,
    icd10_code: str | None,
    atc_code: str | None,
    chunk_id,
    page: int,
    size: int,
) -> dict:
    from sqlalchemy import func, select

    from src.models.entity import MedicalEntity as MedicalEntityModel

    query = select(MedicalEntityModel)
    if entity_type:
        query = query.where(MedicalEntityModel.entity_type == entity_type)
    if icd10_code:
        query = query.where(MedicalEntityModel.icd10_code == icd10_code)
    if atc_code:
        query = query.where(MedicalEntityModel.atc_code == atc_code)
    if chunk_id:
        query = query.where(MedicalEntityModel.chunk_id == chunk_id)
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
    rows = (
        await db.execute(
            query.order_by(MedicalEntityModel.created_at.desc()).offset((page - 1) * size).limit(size)
        )
    ).scalars().all()
    return {
        "items": [
            {
                "id": str(row.id),
                "chunk_id": str(row.chunk_id),
                "entity_text": row.entity_text,
                "entity_type": row.entity_type,
                "icd10_code": row.icd10_code,
                "atc_code": row.atc_code,
                "confidence": row.confidence,
                "start_char": row.start_char,
                "end_char": row.end_char,
            }
            for row in rows
        ],
        "total": total,
        "page": page,
        "size": size,
    }


async def get_entities_stats(db) -> dict:
    from sqlalchemy import func, select

    from src.models.entity import MedicalEntity as MedicalEntityModel

    rows = await db.execute(
        select(MedicalEntityModel.entity_type, func.count(MedicalEntityModel.id)).group_by(
            MedicalEntityModel.entity_type
        )
    )
    return dict(rows.all())
