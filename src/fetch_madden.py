"""Official Madden NFL 22 ratings (EA) for every player in the dataset -> app_profiles/madden.json

    python fetch_madden.py

Pulls the launch ratings and the week-8 roster update (our tracking covers weeks 1-8) from EA's
public ratings API, matches players by name + birth date (then name + team, then unique name),
and keeps the overall plus six position-specific attributes for the card.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import polars as pl

from bdb import BDB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
APP = os.path.join(HERE, "..", "app_profiles")
API = "https://ratings-api.ea.com/v2/entities/m22-ratings?filter=iteration:{it}&limit=1000&offset={off}"
ITERS = {"launch": "launch-ratings", "wk8": "week-8"}

TEAM = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI",
        "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
        "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC", "Rams": "LA", "Chargers": "LAC",
        "Raiders": "LV", "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
        "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA", "49ers": "SF", "Buccaneers": "TB",
        "Titans": "TEN", "Football Team": "WAS", "Washington": "WAS"}

# Madden position -> six card attributes (label, EA field)
A = lambda *xs: [(x, f) for x, f in xs]
ATTRS = {
    "QB": A(("THP", "throwPower_rating"), ("SAC", "throwAccuracyShort_rating"), ("MAC", "throwAccuracyMid_rating"),
            ("DAC", "throwAccuracyDeep_rating"), ("TUP", "throwUnderPressure_rating"), ("AWR", "awareness_rating")),
    "HB": A(("SPD", "speed_rating"), ("AGI", "agility_rating"), ("BCV", "bCVision_rating"), ("BTK", "breakTackle_rating"),
            ("CAR", "carrying_rating"), ("CTH", "catching_rating")),
    "FB": A(("SPD", "speed_rating"), ("STR", "strength_rating"), ("LBK", "leadBlock_rating"), ("RBK", "runBlock_rating"),
            ("CAR", "carrying_rating"), ("CTH", "catching_rating")),
    "WR": A(("SPD", "speed_rating"), ("CTH", "catching_rating"), ("SRR", "shortRouteRunning_rating"),
            ("DRR", "deepRouteRunning_rating"), ("REL", "release_rating"), ("CIT", "catchInTraffic_rating")),
    "TE": A(("SPD", "speed_rating"), ("CTH", "catching_rating"), ("SRR", "shortRouteRunning_rating"),
            ("CIT", "catchInTraffic_rating"), ("RBK", "runBlock_rating"), ("PBK", "passBlock_rating")),
    "OL": A(("PBK", "passBlock_rating"), ("PBP", "passBlockPower_rating"), ("PBF", "passBlockFinesse_rating"),
            ("RBK", "runBlock_rating"), ("STR", "strength_rating"), ("AWR", "awareness_rating")),
    "DL": A(("PMV", "powerMoves_rating"), ("FMV", "finesseMoves_rating"), ("BSH", "blockShedding_rating"),
            ("STR", "strength_rating"), ("PUR", "pursuit_rating"), ("SPD", "speed_rating")),
    "LB": A(("SPD", "speed_rating"), ("TAK", "tackle_rating"), ("PUR", "pursuit_rating"), ("PRC", "playRecognition_rating"),
            ("ZCV", "zoneCoverage_rating"), ("PMV", "powerMoves_rating")),
    "CB": A(("SPD", "speed_rating"), ("MCV", "manCoverage_rating"), ("ZCV", "zoneCoverage_rating"), ("PRS", "press_rating"),
            ("AGI", "agility_rating"), ("PRC", "playRecognition_rating")),
    "S": A(("SPD", "speed_rating"), ("ZCV", "zoneCoverage_rating"), ("MCV", "manCoverage_rating"), ("TAK", "tackle_rating"),
           ("PRC", "playRecognition_rating"), ("HPW", "hitPower_rating")),
    "K": A(("KPW", "kickPower_rating"), ("KAC", "kickAccuracy_rating"), ("AWR", "awareness_rating")),
}
GROUP = {"QB": "QB", "HB": "HB", "FB": "FB", "WR": "WR", "TE": "TE", "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL",
         "RT": "OL", "LE": "DL", "RE": "DL", "DT": "DL", "LOLB": "LB", "MLB": "LB", "ROLB": "LB", "CB": "CB",
         "FS": "S", "SS": "S", "K": "K", "P": "K"}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = SUFFIX.sub("", re.sub(r"[.'\-]", "", s))
    return re.sub(r"[^a-z]", "", s)


def dob(s) -> str | None:
    """'6/15/1993' or '1993-06-15' -> '1993-06-15'"""
    if not s:
        return None
    s = str(s)[:10]
    if "/" in s:
        m, d, y = s.split("/")
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return s


def pull(it: str) -> list[dict]:
    cache = os.path.join(OUT, f"m22_{it}_all.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    docs, off = [], 0
    while True:
        req = urllib.request.Request(API.format(it=it, off=off), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        docs += page["docs"]
        off += len(page["docs"])
        if not page["docs"] or off >= page["count"]:
            break
    json.dump(docs, open(cache, "w", encoding="utf-8"))
    return docs


def main():
    d = BDB()
    idx = json.load(open(os.path.join(APP, "index.json"), encoding="utf-8"))
    ours = {p["id"]: p for p in idx["players"]}
    bio = {r["nflId"]: r for r in d.players().iter_rows(named=True)}

    rated = {}
    for key, it in ITERS.items():
        docs = pull(it)
        by_nd, by_nt, by_n, by_ld = {}, {}, {}, {}
        for x in docs:
            n = norm(f"{x['firstName']} {x['lastName']}")
            by_nd.setdefault((n, dob(x.get("plyrBirthdate"))), []).append(x)
            by_nt.setdefault((n, TEAM.get(x.get("team"))), []).append(x)
            by_n.setdefault(n, []).append(x)
            by_ld.setdefault((norm(x["lastName"]), dob(x.get("plyrBirthdate"))), []).append(x)
        how = {"name+dob": 0, "name+team": 0, "last+dob": 0, "name": 0, "none": 0}
        for nid, p in ours.items():
            n = norm(p["name"])
            b = bio.get(nid, {})
            hit = by_nd.get((n, dob(b.get("birthDate"))))
            tag = "name+dob"
            if not hit:
                hit, tag = by_nt.get((n, p["team"])), "name+team"
            if not hit and b.get("birthDate"):        # nicknames: Nicholas/Nick, Shaquille/Shaq
                hit, tag = by_ld.get((norm(p["name"].split(" ")[-1]), dob(b.get("birthDate")))), "last+dob"
            if not hit:
                hit, tag = by_n.get(n), "name"
                if hit and len(hit) > 1:
                    hit = None
            if not hit:
                how["none"] += 1
                continue
            how[tag] += 1
            x = hit[0]
            pos = x.get("position")
            grp = GROUP.get(pos, "OL")
            r = rated.setdefault(nid, {"mpos": pos, "arch": x.get("archetype")})
            r[key] = x.get("overall_rating")
            if key == "wk8" or "attrs" not in r:
                r["attrs"] = [[lab, x.get(f)] for lab, f in ATTRS[grp] if x.get(f) is not None]
                r["mpos"], r["arch"], r["team"] = pos, x.get("archetype"), TEAM.get(x.get("team"))
        print(f"{it}: {len(docs)} EA players, matched {how}")
    # the card uses the week-8 overall when he was on a week-8 roster, otherwise launch
    for r in rated.values():
        r["ovr"] = r.get("wk8") or r.get("launch")
    with open(os.path.join(APP, "madden.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in rated.items()}, f, separators=(",", ":"), ensure_ascii=False)
    print(f"rated {len(rated)} of {len(ours)} players")
    top = sorted(((v["ovr"] or 0, ours[k]["name"], v["mpos"]) for k, v in rated.items()), reverse=True)[:12]
    print(top)
    miss = [ours[k]["name"] for k in ours if k not in rated]
    print("unmatched sample:", miss[:15])


if __name__ == "__main__":
    main()
