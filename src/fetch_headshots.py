"""Download player headshots for the Film Room cards (private hackathon demo use).

    python fetch_headshots.py

nflId -> ESPN id via the nflverse player table (out/nflverse_players.csv), then ESPN's
transparent-background PNG, shrunk to a small WebP. Falls back to nflverse's NFL.com
headshot URL. Bundles them per team as data URIs: app_profiles/faces_<TEAM>.json
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "headshots")
APP = os.path.join(HERE, "..", "app_profiles")
XWALK = os.path.join(HERE, "..", "out", "nflverse_players.csv")
XWALK_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
ESPN = "https://a.espncdn.com/i/headshots/nfl/players/full/{}.png"
WIDTH = 240


def get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read() if r.status == 200 else None
    except Exception:
        return None


def fetch(job) -> tuple[int, str]:
    nid, espn, fallback = job
    path = os.path.join(OUT, f"{nid}.webp")
    if os.path.exists(path):
        return nid, "cached"
    for url, src in ((ESPN.format(espn) if espn else None, "espn"), (fallback, "nfl")):
        if not url:
            continue
        raw = get(url)
        if not raw:
            continue
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception:
            continue
        if im.width < 40:
            continue
        im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
        im.save(path, "WEBP", quality=80, method=6)
        return nid, src
    return nid, "missing"


def main():
    if not os.path.exists(XWALK):
        urllib.request.urlretrieve(XWALK_URL, XWALK)
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(os.path.join(APP, "index.json"), encoding="utf-8"))
    team = {p["id"]: p["team"] for p in idx["players"]}
    nv = (pl.read_csv(XWALK, infer_schema_length=0)
            .filter(pl.col("nfl_id").is_not_null())
            .with_columns(pl.col("nfl_id").cast(pl.Int64, strict=False)))
    look = {r["nfl_id"]: r for r in nv.select(["nfl_id", "espn_id", "headshot"]).iter_rows(named=True)}
    jobs = [(n, (look.get(n) or {}).get("espn_id"), (look.get(n) or {}).get("headshot")) for n in team]
    stats = defaultdict(int)
    with ThreadPoolExecutor(16) as ex:
        for k, (nid, src) in enumerate(ex.map(fetch, jobs)):
            stats[src] += 1
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} {dict(stats)}", flush=True)
    print("headshots:", dict(stats))

    faces = defaultdict(dict)
    for n, tm in team.items():
        path = os.path.join(OUT, f"{n}.webp")
        if os.path.exists(path):
            faces[tm][n] = "data:image/webp;base64," + base64.b64encode(open(path, "rb").read()).decode()
    total = 0
    for tm, f in faces.items():
        p = os.path.join(APP, f"faces_{tm}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(f, fh, separators=(",", ":"))
        total += os.path.getsize(p)
    print(f"bundled {sum(len(f) for f in faces.values())} faces into {len(faces)} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
