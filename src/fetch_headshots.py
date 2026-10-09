"""Download player headshots for the Film Room cards (private hackathon demo use).

    python fetch_headshots.py

Source order: the NFL.com headshot from the 2021 roster (player in his 2021 kit), then
ESPN's current headshot, then nflverse's latest NFL.com headshot. All are transparent PNGs.
Every photo is cropped to the player and scaled to the same head-to-chest height on an
identical canvas, so faces read the same size on every card.
Bundles them per team as data URIs: app_profiles/faces_<TEAM>.json
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
ROSTER = os.path.join(HERE, "..", "out", "roster_2021.csv")
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2021.csv"
ESPN = "https://a.espncdn.com/i/headshots/nfl/players/full/{}.png"
CANVAS = (240, 200)       # output size
SUBJECT_H = 196           # head-top to bottom edge of the photo, in output pixels


def get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read() if r.status == 200 else None
    except Exception:
        return None


def fit(im: Image.Image) -> Image.Image:
    """Crop to the player, scale to a fixed head-to-chest height, bottom-centre on the canvas."""
    a = im.getchannel("A").point(lambda v: 255 if v > 24 else 0)
    box = a.getbbox()
    if not box:
        return im.resize(CANVAS, Image.LANCZOS)
    im = im.crop(box)
    k = SUBJECT_H / im.height
    im = im.resize((max(1, round(im.width * k)), SUBJECT_H), Image.LANCZOS)
    out = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    if im.width > CANVAS[0]:                      # broad shoulders: trim the sides, keep the head centred
        x = (im.width - CANVAS[0]) // 2
        im = im.crop((x, 0, x + CANVAS[0], im.height))
    out.alpha_composite(im, ((CANVAS[0] - im.width) // 2, CANVAS[1] - im.height))
    return out


def fetch(job) -> tuple[int, str]:
    nid, y2021, espn, fallback = job
    path = os.path.join(OUT, f"{nid}.webp")
    if os.path.exists(path):
        return nid, "cached"
    for url, src in ((y2021, "nfl2021"), (ESPN.format(espn) if espn else None, "espn"), (fallback, "nfl-latest")):
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
        if im.getchannel("A").getextrema()[0] > 200:      # no transparency -> skip, card needs a cut-out
            continue
        fit(im).save(path, "WEBP", quality=82, method=6)
        return nid, src
    return nid, "missing"


def main():
    for f, u in ((XWALK, XWALK_URL), (ROSTER, ROSTER_URL)):
        if not os.path.exists(f):
            urllib.request.urlretrieve(u, f)
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(os.path.join(APP, "index.json"), encoding="utf-8"))
    team = {p["id"]: p["team"] for p in idx["players"]}
    nv = (pl.read_csv(XWALK, infer_schema_length=0)
            .filter(pl.col("nfl_id").is_not_null())
            .with_columns(pl.col("nfl_id").cast(pl.Int64, strict=False)))
    look = {r["nfl_id"]: r for r in nv.select(["nfl_id", "espn_id", "headshot"]).iter_rows(named=True)}
    ro = (pl.read_csv(ROSTER, infer_schema_length=0)
            .filter(pl.col("gsis_it_id").is_not_null() & pl.col("headshot_url").is_not_null())
            .with_columns(pl.col("gsis_it_id").cast(pl.Int64, strict=False)))
    y21 = dict(zip(ro["gsis_it_id"], ro["headshot_url"]))
    jobs = [(n, y21.get(n), (look.get(n) or {}).get("espn_id"), (look.get(n) or {}).get("headshot")) for n in team]
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
