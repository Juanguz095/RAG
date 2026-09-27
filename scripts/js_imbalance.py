#!/usr/bin/env python3
"""Find approximate location of brace/paren imbalance in served JS."""
import re
import urllib.request

t = urllib.request.urlopen("http://localhost:8000/").read().decode()
scripts = re.findall(r"<script>(.*?)</script>", t, re.S)
inline = scripts[-1] if scripts else ""

# Strip strings/comments character by character with state machine
i = 0
n = len(inline)
out = []
mode = None  # None, "'", '"', '`', '//', '/*'
while i < n:
    c = inline[i]
    nxt = inline[i + 1] if i + 1 < n else ""
    if mode is None:
        if c == "/" and nxt == "/":
            mode = "//"
            i += 2
            continue
        if c == "/" and nxt == "*":
            mode = "/*"
            i += 2
            continue
        if c in "'\"`":
            mode = c
            i += 1
            continue
        out.append((i, c))
        i += 1
    elif mode == "//":
        if c == "\n":
            mode = None
            out.append((i, c))
        i += 1
    elif mode == "/*":
        if c == "*" and nxt == "/":
            mode = None
            i += 2
            continue
        i += 1
    else:  # string
        if c == "\\":
            i += 2
            continue
        if c == mode:
            mode = None
        i += 1

line = 1
line_starts = [0]
for idx, ch in out:
    if ch == "\n":
        line_starts.append(idx)

def lineno(pos):
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1

stack = []
for pos, ch in out:
    if ch in "{([":
        stack.append((ch, pos))
    elif ch in "})]":
        pair = {"}": "{", ")": "(", "]": "["}[ch]
        if stack and stack[-1][0] == pair:
            stack.pop()
        else:
            print(f"UNMATCHED {ch} at char {pos} line {lineno(pos)}")
if stack:
    for ch, pos in stack:
        print(f"UNCLOSED {ch} at char {pos} line {lineno(pos)}")
print("mode end:", mode)
print("done")
