"""README media: an animated GIF of a player's best real decoy play, drawn like the site's pitch.

    python make_readme_media.py              # Davante Adams (41282), his season-best real decoy play
    python make_readme_media.py 41282

Real tracking only (10 Hz, snap-0.5 s to the throw). The lured defenders, the 5-yard circle and the
tethers come from the same pipeline the site uses (app_profiles/film_<TEAM>.json -> me.decoy_ids).
Writes docs/media/decoy_play.gif
"""
from __future__ import annotations

import json
import os
import sys

import polars as pl
from PIL import Image, ImageDraw, ImageFont

from bdb import BDB
from decoy import throw_frames

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app_profiles")
OUT = os.path.join(HERE, "..", "docs", "media")

TURF, TURF2 = (31, 74, 49), (27, 66, 43)
CHALK = (255, 255, 255, 110)
OFF, DEF = (233, 239, 234), (11, 18, 34)
PYLON, YELLOW, DECOY, LOS = (255, 106, 19), (245, 204, 26), (56, 198, 244), (90, 150, 255)
SCALE = 14                     # pixels per yard
FONT = "C:/Windows/Fonts/segoeuib.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def short(n: str) -> str:
    p = n.split(" ")
    return f"{p[0][0]}. {' '.join(p[1:])}" if len(p) > 1 else n


def main(pid: int = 41282):
    d = BDB()
    idx = json.load(open(os.path.join(APP, "index.json"), encoding="utf-8"))
    me = next(p for p in idx["players"] if p["id"] == pid)
    film = json.load(open(os.path.join(APP, f"film_{me['team']}.json"), encoding="utf-8"))
    q = next(x for x in film["plays"][str(pid)] if x["best"])
    gid, playid = (int(v) for v in q["key"].split("-"))
    lured = set(q["me"]["decoy_ids"])
    names = {int(k): v for k, v in idx["names"].items()}
    roles = {r["nflId"]: (r["pff_role"] or "").lower() for r in
             d.pff().filter((pl.col("gameId") == gid) & (pl.col("playId") == playid)).iter_rows(named=True)}

    _, tr = throw_frames(d, gid)
    tr = tr.filter((pl.col("playId") == playid) & (pl.col("frameId") >= pl.col("snapFrame") - 5)
                   & (pl.col("frameId") <= pl.col("throwFrame")))
    snapf, thr = tr["snapFrame"][0], tr["throwFrame"][0]
    frames = sorted(tr["frameId"].unique().to_list())
    pos = {(f, n if n is not None else 0): (x, y) for f, n, x, y in tr.select(["frameId", "nflId", "x", "y"]).iter_rows()}
    xs = [x for (f, n), (x, y) in pos.items() if n]
    x0, x1 = min(xs) - 3, max(xs) + 3
    W, H = int(53.3 * SCALE), int((x1 - x0) * SCALE)
    X = lambda y: y * SCALE
    Y = lambda x: (x1 - x) * SCALE
    los = pos.get((snapf, 0), (min(xs) + 8, 0))[0]
    fd = los + (q.get("togo") or 0)
    ids = sorted({n for (f, n) in pos if n})
    tgt = q["target"]

    imgs = []
    for f in frames:
        im = Image.new("RGB", (W, H), TURF)
        dr = ImageDraw.Draw(im, "RGBA")
        for yl in range(int(x0 // 5 * 5), int(x1) + 5, 5):
            dr.rectangle([0, Y(yl + 5), W, Y(yl)], fill=TURF if (yl // 5) % 2 else TURF2)
            if 10 <= yl <= 110:
                dr.line([0, Y(yl), W, Y(yl)], fill=CHALK, width=2)
                if yl % 10 == 0 and 10 < yl < 110:
                    lab = str(50 - abs(50 - (yl - 10)))
                    for yy in (7, 46.3):
                        dr.text((X(yy), Y(yl)), lab, fill=CHALK, font=font(20), anchor="mm")
        dr.line([0, Y(los), W, Y(los)], fill=LOS, width=4)
        if q.get("togo") and fd < x1:
            dr.line([0, Y(fd), W, Y(fd)], fill=YELLOW, width=4)
        at_throw = f >= thr
        mp = pos.get((f, pid))
        trail = [pos[(g, pid)] for g in frames if g <= f and (g, pid) in pos]
        if len(trail) > 1:
            dr.line([(X(y), Y(x)) for x, y in trail], fill=PYLON + (180,), width=3)
        if at_throw and mp:
            r5 = 5 * SCALE
            dr.ellipse([X(mp[1]) - r5, Y(mp[0]) - r5, X(mp[1]) + r5, Y(mp[0]) + r5], fill=DECOY + (40,), outline=DECOY, width=2)
            for c in lured:
                cp = pos.get((f, c))
                if cp:
                    dr.line([X(mp[1]), Y(mp[0]), X(cp[1]), Y(cp[0])], fill=DECOY, width=3)
        r = 9
        for n in ids:
            p = pos.get((f, n))
            if not p:
                continue
            role = roles.get(n, "")
            dfn = role in ("coverage", "pass rush")
            fill = PYLON if n == pid else YELLOW if n == tgt else DEF if dfn else OFF
            outline = DECOY if (at_throw and n in lured) or n == pid else (255, 255, 255) if dfn else (0, 0, 0)
            rr = r + 2 if n == pid else r
            dr.ellipse([X(p[1]) - rr, Y(p[0]) - rr, X(p[1]) + rr, Y(p[0]) + rr], fill=fill, outline=outline,
                       width=3 if outline == DECOY else 2)
        b = pos.get((f, 0))
        if b:
            dr.ellipse([X(b[1]) - 6, Y(b[0]) - 4, X(b[1]) + 6, Y(b[0]) + 4], fill=(122, 74, 34), outline=(255, 255, 255))
        lab = font(17)
        def label(n, txt, col):
            p = pos.get((f, n))
            if p:
                dr.text((X(p[1]) + 14, Y(p[0])), txt, fill=col, font=lab, anchor="lm", stroke_width=3, stroke_fill=(0, 0, 0))
        label(pid, short(me["name"]) + (f" · lured {len(lured)}" if at_throw else ""), DECOY if at_throw else (255, 255, 255))
        label(tgt, short(names.get(tgt, "")) + " · target", YELLOW)
        t = (f - snapf) / 10
        dr.text((12, 10), "THROW" if at_throw else f"{t:+.1f} s", fill=(255, 255, 255), font=font(28), stroke_width=3, stroke_fill=(0, 0, 0))
        imgs.append(im)

    hold = [imgs[-1]] * 18                                  # rest on the throw frame (1.8 s) before looping
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "decoy_play.gif")
    imgs[0].save(path, save_all=True, append_images=imgs[1:] + hold, duration=100, loop=0, optimize=True)
    print(f"{path}: {len(imgs)} real frames + hold, {W}x{H}, {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"play: wk {q['week']} Q{q['q']} {q['clock']} — {q['desc']}")
    print(f"lured: {[names.get(c) for c in lured]}  score {q['me']['decoy_score']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 41282)
