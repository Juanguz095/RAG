#!/usr/bin/env python3
"""Extract inline script and syntax-check with node."""
import re
import pathlib
import subprocess
import sys
import urllib.request

t = urllib.request.urlopen("http://localhost:8000/").read().decode()
scripts = re.findall(r"<script>(.*?)</script>", t, re.S)
inline = scripts[-1] if scripts else ""
if not inline:
    print("NO SCRIPT")
    sys.exit(1)

out = pathlib.Path(__file__).resolve().parent / "_frontend_check.js"
# Wrap so bare top-level declarations are ok; use check syntax only
out.write_text(inline, encoding="utf-8")
r = subprocess.run(
    ["node", "--check", str(out)],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
print("stdout:", r.stdout)
print("stderr:", r.stderr)
print("code:", r.returncode)
