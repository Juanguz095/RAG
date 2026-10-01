"""Dataset sintético (RAG-058 — PLAN-006 Fase 5, spec §68).

Genera PDFs 100% ficticios (formularios médicos de juguete, nombres
inventados, datos clínicos de prueba) para demos y QA — sin datos reales
de pacientes. Solo libs locales (reportlab). 100% offline.
"""
from __future__ import annotations

import random
from pathlib import Path

NOMBRES = ["Ana Demo Pérez", "Luis Ficticio García", "Mara Prueba López",
           "Óscar Ejemplo Torres", "Nina Muestra Ramírez"]
DX = ["CTD-x-001 demo", "RUTINA CONTROL NIÑOS", "ATENCIÓN OBSTÉTRICA DEMO"]
MEDS = ["PARACETAMOL 500 mg TAB", "SOLUCIÓN ORAL DEMO 250 ml",
        "VITAMINA D 1000 UI CAP", "SUERO ORAL JUGUETE 500 ml"]


def _pdf_form(output: Path, idx: int) -> Path:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    rnd = random.Random(42 + idx)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"sintetico_demo_{idx + 1:02d}.pdf"
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 755, f"DEMO - FORMULARIO DE ATENCION SINTETICO #{idx + 1:02d}")
    c.setFont("Helvetica", 10)
    lines = [
        "ESTABLECIMIENTO: CENTRO DEMO (LOCAL)",
        "HISTORIA CLINICA: HC-00000000 (sinte<tico>), SIN DATOS REALES",
        f"PACIENTE: {rnd.choice(NOMBRES)}",
        f"DNI: {rnd.randint(10000000, 99999999)}X (FICTICIO)",
        f"TIPO DE ATENCION: {rnd.choice(DX)}",
        "RELACION MADRE - HIJO: LA ATENCION SIN COMPLICACIONES (DEMO)",
        "",
        "MEDICAMENTOS (NOMBRE | CONCENTRACION | CANTIDAD DISPENSADA):",
    ]
    for m in rnd.sample(MEDS, 3):
        lines.append(f"  {m} | dosis demo | {rnd.randint(1, 30)}")
    lines += ["", "FIRMA RESPONSABLE: (DEMO - SIN VALOR LEGAL)"]
    y = 730
    for ln in lines:
        c.drawString(72, y, ln)
        y -= 14
    c.showPage()
    c.save()
    return path


def generate(out_dir: Path, count: int = 3) -> list[Path]:
    return [_pdf_form(Path(out_dir), i) for i in range(count)]


if __name__ == "__main__":
    import sys

    destino = Path(sys.argv[1] if len(sys.argv) > 1 else "./datossinteticos")
    pdfs = generate(destino, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    print(f"Generados {len(pdfs)} PDFs sintéticos en {destino}")
