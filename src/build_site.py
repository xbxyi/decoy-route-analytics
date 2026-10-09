"""Assemble a standalone static site in site/ from app_profiles/ (for Vercel or any static host).

    python build_site.py
    npx vercel deploy ../site --prod      # or: --temporary to deploy without logging in

app_profiles/index.html is written for a host that supplies the document skeleton; this wraps it
in a full HTML document and copies the data files next to it.
"""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "app_profiles")
OUT = os.path.join(HERE, "..", "site")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Which NFL receivers drag defenders away from the target: decoy routes from 2021 player tracking.">
<style>html{-webkit-text-size-adjust:100%}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    page = open(os.path.join(SRC, "index.html"), encoding="utf-8").read()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(HEAD + page + "\n</body>\n</html>\n")
    n = 0
    for name in os.listdir(SRC):
        if name.endswith(".json"):
            shutil.copy2(os.path.join(SRC, name), os.path.join(OUT, name))
            n += 1
    size = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT))
    print(f"site/: index.html + {n} data files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
