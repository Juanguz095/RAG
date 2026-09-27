"""Medical synonyms dictionary based on clinical document fields.

Each key maps to a canonical term and a list of aliases/synonyms used in
medical documents. Used for fuzzy search with percentage matching.
"""

# Canonical terms -> list of aliases/synonyms (all lowercase)
MEDICAL_SYNONYMS: dict[str, list[str]] = {
    # Diagnóstico
    "diagnostico": [
        "dx", "dg", "diagnóstico", "diagnosticos", "diagnósticos",
        "impresion diagnostica", "impresión diagnóstica",
        "juicio clinico", "juicio clínico",
        "valoracion", "valoración", "evaluacion", "evaluación",
        "hallazgos", "conclusion", "conclusión",
        "problemas", "problema activo", "epicrisis",
        "enfermedad actual", "presuntivo", "definitivo",
        "compatible con", "probable", "sospecha de",
        "se concluye", "se evidencia",
        "cie", "cie10", "cie-10", "cid", "cid10", "cid-10",
        "plan", "conducta",
    ],
    # Historia clínica
    "historia_clinica": [
        "hc", "h.c", "historia clinica", "historia clínica",
        "nro historia", "numero historia", "número de historia",
        "expediente clinico", "expediente clínico",
    ],
    # Datos del paciente
    "edad": ["ed", "edad", "anos", "años"],
    "dni": ["dni", "documento", "documento de identidad", "cedula"],
    "nombre_completo": [
        "nombre completo", "nombres", "nombres y apellidos",
        "apellido paterno", "apellido materno",
    ],
    # Número de formato
    "numero_formato": ["numero formato", "número formato", "formato"],
    # Códigos institucionales
    "renaes": ["renaes"],
    "renipress": ["renipress"],
    "hrdlm": ["hrdlm"],
    "minsa": ["minsa"],
    "sis": ["sis"],
    "fua": ["fua"],
    "ce": ["ce"],
    # Fechas
    "fecha_atencion": [
        "fecha de atencion", "fecha de atención",
        "fecha de atención", "fecha consignada",
    ],
    "fecha_egreso": ["fecha de egreso"],
    # Profesional
    "responsable": [
        "responsable de la atencion", "responsable de la atención",
        "medico tratante", "doctor", "dr",
    ],
    "especialidad": [
        "especialidad", "oftalmologia", "oftalmología",
        "oftalmo", "traumatologia", "traumatología",
        "ortopedia", "medicina interna", "cardiologia",
    ],
    "cmp": ["cmp"],
    "rne": ["rne"],
    # Oftalmología
    "ojo_derecho": ["od", "ojo derecho"],
    "ojo_izquierdo": ["oi", "ojo izquierdo"],
    "ambos_ojos": ["ao", "ambos ojos"],
    # Catarata
    "catarata": [
        "catarata", "catarata senil", "catarata od", "catarata oi",
        "catarata bilateral",
    ],
    "cirugia_catarata": [
        "cirugia de catarata", "cirugía de catarata",
        "extraccion de catarata", "extracción de catarata",
        "facoemulsificacion", "facoemulsificación",
    ],
    "lio": [
        "lio", "implante lio", "implante de lio",
        "lente intraocular",
    ],
    "lentes": ["eecc", "lentes", "gafas", "anteojos"],
    # Procedimientos oftálmicos
    "anestesia_local": ["anestesia local"],
    "oclusion_ocular": ["oclusión ocular", "oclusion ocular"],
    "agudeza_visual": ["agudeza visual", "av", "agudeza visual central"],
    "campimetria": ["campimetria", "campimetría"],
    "fondo_ojo": ["fondo de ojo", "fondo de ojos"],
    "oftalmoscopia": ["oftalmoscopia", "oftalmoscopía"],
    # Medicamentos comunes
    "amoxicilina": ["amoxicilina"],
    "metformina": ["metformina"],
    "naproxeno": ["naproxeno"],
    "omeprazol": ["omeprazol"],
    "paracetamol": ["paracetamol"],
    "ciprofloxacino": ["ciprofloxacino"],
    "enalapril": ["enalapril"],
    "ibuprofeno": ["ibuprofeno"],
    "levotiroxina": ["levotiroxina"],
    # Procedimientos médicos generales
    "oximetria": ["oximetria", "oximetría"],
    "sonda_nasogastrica": ["sonda nasogastrica", "sonda nasogástrica"],
    "sonda_foley": ["sonda foley"],
    "fototerapia": ["fototerapia"],
    "transfusion": ["transfusion", "transfusión"],
    "consejeria_nutricional": ["consejeria nutricional", "consejería nutricional"],
    "ecografia": ["ecografia", "ecografía"],
    "radiografia": ["radiografia", "radiografía"],
    "perfil_neonatal": ["perfil neonatal"],
    # Vacunas
    "bcg": ["bcg"],
    "hepatitis_b": ["hepatitis b"],
    "influenza": ["influenza"],
    "antitetanica": ["antitetanica", "antitetánica"],
    "sarampion": ["sarampion", "sarampión"],
    "rubeola": ["rubeola", "rubéola"],
    "parotiditis": ["parotiditis"],
    # signos vitales y valores clinicos
    "signos_vitales": [
        "signos vitales", "tension arterial", "tensión arterial",
        "frecuencia cardiaca", "frecuencia respiratoria",
        "temperatura", "peso", "talla", "imc",
    ],
}


def expand_query(query: str) -> list[str]:
    """Expand a search query with synonyms.

    Returns a list of expanded terms including the original query and any
    matching synonyms found in the dictionary.
    """
    query_lower = query.lower().strip()
    terms = [query_lower]

    # Direct match in synonym values
    for _canonical, aliases in MEDICAL_SYNONYMS.items():
        for alias in aliases:
            if query_lower in alias or alias in query_lower:
                terms.extend(aliases[:5])
                break

    # Check if query matches a canonical key
    if query_lower in MEDICAL_SYNONYMS:
        terms.extend(MEDICAL_SYNONYMS[query_lower][:5])

    return list(dict.fromkeys(terms))


def match_score(query: str, text: str) -> float:
    """Calculate a fuzzy match score (0.0 - 1.0) between query and text.

    Uses a combination of:
    - Exact substring match (1.0)
    - Prefix match (0.9)
    - Trigram-style character overlap
    - Synonym match bonus
    """
    query_lower = query.lower().strip()
    text_lower = text.lower().strip()

    if not query_lower or not text_lower:
        return 0.0

    # Exact substring match
    if query_lower in text_lower:
        return 1.0

    # Check synonym expansion
    expanded = expand_query(query_lower)
    for term in expanded[1:]:
        if term in text_lower:
            return 0.95

    # Prefix match
    words = text_lower.split()
    for word in words:
        if word.startswith(query_lower):
            return 0.85

    # Character overlap (trigram-like)
    q_set = set(query_lower)
    t_set = set(text_lower)
    overlap = len(q_set & t_set)
    total = len(q_set | t_set)
    if total == 0:
        return 0.0
    char_score = overlap / total

    # Word-level match
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    word_overlap = len(query_words & text_words)
    word_score = word_overlap / len(query_words) if query_words else 0.0

    return max(char_score * 0.7, word_score * 0.85)
