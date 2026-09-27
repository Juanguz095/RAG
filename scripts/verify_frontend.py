#!/usr/bin/env python3
"""Verifica la integridad del frontend servido tras el rewrite de Búsqueda Inteligente."""
import urllib.request

t = urllib.request.urlopen("http://localhost:8000/").read().decode()

checks = {
    "occ-header CSS": ".occ-header" in t,
    "occ-item CSS": ".occ-item" in t,
    "result-action CSS": ".result-action {" in t,
    "findWordOccurrences": "function findWordOccurrences" in t,
    "goToOccurrence": "function goToOccurrence" in t,
    "highlightAllOccurrences": "function highlightAllOccurrences" in t,
    "centerOnBox": "function centerOnBox" in t,
    "buildOccContext": "function buildOccContext" in t,
    "renderOccList": "function renderOccList" in t,
    "aparece header": "aparece" in t,
    "sin /api/v1/search": "/api/v1/search" not in t,
    "sin goToSearchResult": "goToSearchResult" not in t,
    "sin highlightPdfText": "highlightPdfText" not in t,
    "sin rel-level": "rel-level" not in t,
    "sin class=search-result": 'class="search-result"' not in t,
    "populatePageFilter() call": "populatePageFilter();" in t,
    "sin populatePageFilter(docChunks)": "populatePageFilter(docChunks)" not in t,
    "sin type-filter": "filter-type" not in t,
    "wheel zoom ctrlKey": "ctrlKey" in t and "pdfZoom" in t,
    "applyPdfZoom": "function applyPdfZoom" in t,
    "autoOpenFirst": "autoOpenFirst" in t,
    "clearPdfMarks": "function clearPdfMarks" in t,
}

fails = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("OK  " if v else "FAIL") + " " + k)
print("---")
print("TODO OK" if not fails else str(len(fails)) + " FALLOS: " + ", ".join(fails))
