#!/usr/bin/env python3
"""Check JS brace balance in served frontend."""
import re
import urllib.request

t = urllib.request.urlopen("http://localhost:8000/").read().decode()
scripts = re.findall(r"<script>(.*?)</script>", t, re.S)
inline = scripts[-1] if scripts else ""
s = re.sub(r"//[^\n]*", "", inline)
s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
s = re.sub(r"`(?:[^`\\]|\\.)*`", "``", s)
print("braces", s.count("{"), s.count("}"), "diff", s.count("{") - s.count("}"))
print("parens", s.count("("), s.count(")"), "diff", s.count("(") - s.count(")"))
print("brackets", s.count("["), s.count("]"), "diff", s.count("[") - s.count("]"))
print("script chars", len(inline))
