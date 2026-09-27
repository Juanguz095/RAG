"""Conservative, local PII masking for medical documents."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnonymizationResult:
    text: str
    replacements: int
    entity_types: tuple[str, ...]


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d[\s-]?){8,14}\d(?!\d)")),
    ("DNI", re.compile(r"\b(?:DNI|RUT|C[ÉE]DULA|DOCUMENTO)\s*[:#-]?\s*[A-Z0-9.-]{6,15}\b", re.I)),
    ("PATIENT_ID", re.compile(r"\b(?:HC|HISTORIA\s+CL[IÍ]NICA|PACIENTE\s*ID)\s*[:#-]?\s*[A-Z0-9-]{3,20}\b", re.I)),
    ("ADDRESS", re.compile(r"\b(?:DIRECCI[ÓO]N|DOMICILIO)\s*[:#-]?\s*[^\n,;]{5,100}", re.I)),
    ("PERSON", re.compile(r"\b(?:PACIENTE|NOMBRE|APELLIDOS?)\s*:\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ'’-]*(?:\s+[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ'’-]*){1,3}", re.I)),
)


def _same_length_mask(label: str, value: str) -> str:
    if len(value) <= len(label):
        return label[: len(value)]
    return label + " " * (len(value) - len(label))


def anonymize_text(text: str) -> AnonymizationResult:
    """Mask high-confidence PII while preserving original character offsets."""
    matches: list[tuple[int, int, str, str]] = []
    for entity_type, pattern in _PATTERNS:
        matches.extend(
            (match.start(), match.end(), entity_type, match.group())
            for match in pattern.finditer(text)
        )
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    accepted: list[tuple[int, int, str, str]] = []
    last_end = -1
    for match in matches:
        if match[0] >= last_end:
            accepted.append(match)
            last_end = match[1]

    masked = text
    entity_types: set[str] = set()
    for start, end, entity_type, value in reversed(accepted):
        masked = masked[:start] + _same_length_mask(f"[{entity_type}]", value) + masked[end:]
        entity_types.add(entity_type)
    return AnonymizationResult(masked, len(accepted), tuple(sorted(entity_types)))
