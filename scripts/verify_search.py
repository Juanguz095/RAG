import urllib.request
t = urllib.request.urlopen('http://localhost:8000/').read().decode()
checks = {
    'sin filter-type': 'filter-type' not in t,
    'sin type_filter JS': 'type_filter' not in t,
    'hint sin doc': 'Abre un documento de la izquierda' in t,
    'input disabled': 'disabled onkeydown' in t,
    'placeholder context doc': 'Buscar en este documento' in t,
    'Sugerencias label': 'Sugerencias' in t,
    'docExtracted guard': 'docExtracted = null' in t,
    'snippet result': 'c.snippet' in t,
    'rel Alta': "['high', 'Alta']" in t,
    'rel Media': "['mid', 'Media']" in t,
    'Ver en el PDF': 'Ver en el PDF' in t,
    '1er resultado badge': '1er resultado' in t,
    'empty examples': 'tryExample(this)' in t,
    'btn asistente': 'Preguntale al asistente' in t,
    'askFromSearch': 'function askFromSearch' in t,
    'guard no doc': 'if (!selectedDocId) return;' in t,
    'search-hint css': '.search-hint' in t,
    'rel-level css': '.rel-level' in t,
}
fails = 0
for k, v in checks.items():
    print(('OK  ' if v else 'FAIL') + ' ' + k)
    if not v:
        fails += 1
print('---')
print('TODO OK' if fails == 0 else str(fails) + ' FALLOS')
