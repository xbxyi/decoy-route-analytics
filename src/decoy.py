"""Decoy Efficiency — which route runners drag defenders away from the actual target?

For every pass play with an identifiable target:
  * freeze the field at the throw (pass_forward / autoevent_passforward)
  * every route runner who is NOT the target is a decoy
  * a coverage defender is "attached" to a decoy if within ATTACH_YDS of him
    (and not also within ATTACH_YDS of the target — he isn't pulled away, he's there)
  * decoy score = attached defenders x mean distance from those defenders to the target
                = sum of yards of coverage pulled off the target

    python decoy.py            # all games -> out/decoy_reps.parquet, out/decoy_plays.parquet
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import polars as pl

from bdb import BDB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

ATTACH_YDS = 5.0          # close-cover radius; also the teammate zone and the "on the target" radius
BAIT_CLOSE_YDS = 7.0      # a defender assigned to the decoy and this close at the throw is baited outright
BAIT_MAX_YDS = 15.0       # a baited defender can be up to this far from the decoy at the throw
BAIT_WINDOW = 15          # frames before the throw used to judge baiting (1.5 s at 10 Hz)
PURSUIT_BASE = 0.25       # pursuit (cosine) needed just outside the close-cover radius ...
PURSUIT_PER_YD = 0.03     # ... rising this much per extra yard: farther defenders need clearer chasing
THROW_EVENTS = ("pass_forward", "autoevent_passforward")

# "pass short left to M.Evans", "pass incomplete deep right to A.St. Brown",
# "pass short middle intended for K.Hinton INTERCEPTED"
_TARGET_RE = re.compile(
    r"pass (?:incomplete )?(?:(?:short|deep) (?:left|middle|right) )?(?:to|intended for) "
    r"([A-Z][A-Za-z]{0,2}\.\s?(?:St\.\s)?[A-Z][A-Za-z'\-]+)")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def parse_target(desc: str | None) -> str | None:
    """Last named target in the description (reviewed plays repeat the call; last one stands)."""
    if not desc:
        return None
    m = _TARGET_RE.findall(desc)
    return m[-1] if m else None


def _match(abbrev: str, cands: list[tuple[int, str]]) -> int | None:
    """'M.Evans' -> nflId among this play's route runners."""
    first, _, last = abbrev.partition(".")
    last_n, first_n = _norm(last), _norm(first)
    # strip trailing words that leaked from the sentence ("K.Hinton INTERCEPTED" can't happen
    # thanks to the regex, but "D.Metcalf For" could) by trying progressively shorter tails
    hits = [(i, n) for i, n in cands if _norm(n).endswith(last_n)]
    if not hits:
        tail = last.split()[0] if last.split() else last
        hits = [(i, n) for i, n in cands if _norm(n.split()[-1]).startswith(_norm(tail))]
    if len(hits) > 1:
        hits = [(i, n) for i, n in hits if _norm(n).startswith(first_n[:1])] or hits
    return hits[0][0] if len(hits) == 1 else None


def targets(d: BDB) -> pl.DataFrame:
    """gameId, playId, targetId for every pass play whose target is a route runner."""
    plays = d.plays().filter(pl.col("passResult").is_in(["C", "I", "IN"]))
    pff = d.pff().filter(pl.col("pff_role").str.to_lowercase() == "pass route")
    names = dict(zip(d.players()["nflId"], d.players()["displayName"]))
    runners = (pff.group_by(["gameId", "playId"]).agg(pl.col("nflId"))
                  .rows_by_key(["gameId", "playId"], unique=True))
    rows = []
    for g, p, desc in plays.select(["gameId", "playId", "playDescription"]).iter_rows():
        ab = parse_target(desc)
        rr = runners.get((g, p))
        if not ab or not rr:
            continue
        tid = _match(ab, [(i, names.get(i, "")) for i in rr[0]])
        if tid is not None:
            rows.append((g, p, tid, ab))
    return pl.DataFrame(rows, schema=["gameId", "playId", "targetId", "targetAbbrev"], orient="row")


def throw_frames(d: BDB, gid: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(every player at the throw frame, snap->throw frames for animation)."""
    tr = d.standardize(d.tracking(gid))
    ev = tr.filter(pl.col("event").is_not_null()).select(["playId", "frameId", "event"]).unique()
    thr = (ev.filter(pl.col("event").is_in(THROW_EVENTS))
             .group_by("playId").agg(pl.col("frameId").min().alias("throwFrame")))
    snp = (ev.filter(pl.col("event").is_in(["ball_snap", "autoevent_ballsnap"]))
             .group_by("playId").agg(pl.col("frameId").min().alias("snapFrame")))
    tr = tr.join(thr, on="playId").join(snp, on="playId", how="left")
    at = tr.filter(pl.col("frameId") == pl.col("throwFrame"))
    return at, tr


def _paths(win: pl.DataFrame) -> dict:
    """(gameId, playId) -> {nflId: (n_frames, 2) positions over the bait window, NaN where missing}."""
    out = {}
    for (g, p), w in win.group_by(["gameId", "playId"]):
        frames = sorted(w["frameId"].unique().to_list())
        fi = {f: i for i, f in enumerate(frames)}
        paths = {}
        for n, f, x, y in w.select(["nflId", "frameId", "x", "y"]).iter_rows():
            if n is None:
                continue
            a = paths.setdefault(n, np.full((len(frames), 2), np.nan))
            a[fi[f]] = (x, y)
        out[(g, p)] = paths
    return out


def baited(paths: dict, decoy: int, routes: list, cov: list, at_throw: dict, on_target: dict) -> list[int]:
    """Coverage defenders baited by this decoy over the last 1.5 s before the throw.

    A defender is baited when all hold:
      1. responsibility — of all route runners, the decoy is the one he stayed closest to on
         average over the window (so he is not credited to the wrong receiver);
      2. within BAIT_MAX_YDS (15) of the decoy at the throw;
      3. close (inside BAIT_CLOSE_YDS, 7, at the throw) OR pursuing: averaged over frames where he
         is moving, his velocity points at the decoy or mirrors the decoy's velocity, with the bar
         rising with distance: cosine >= 0.25 + 0.03 per yard beyond 7 (0.31 at 9 yd, 0.49 at 15);
      4. not within ATTACH_YDS of the target at the throw (then he is covering the target).
    """
    D = paths.get(decoy)
    if D is None:
        return []
    vD = np.diff(D, axis=0) / 0.1
    out = []
    for c in cov:
        C = paths.get(c)
        if C is None or on_target.get(c):
            continue
        mean_d = {r: np.nanmean(np.hypot(*(C - paths[r]).T)) for r in routes if r in paths}
        if not mean_d or min(mean_d, key=mean_d.get) != decoy:
            continue
        cx, cy = at_throw[c]
        dx, dy = at_throw[decoy]
        d_now = float(np.hypot(cx - dx, cy - dy))
        if d_now >= BAIT_MAX_YDS:
            continue
        if d_now < BAIT_CLOSE_YDS:
            out.append(c)
            continue
        vC = np.diff(C, axis=0) / 0.1
        to_D = (D - C)[1:]
        sp_c, sp_d = np.hypot(*vC.T), np.hypot(*vD.T)
        cos_at = (vC * to_D).sum(1) / (sp_c * np.hypot(*to_D.T) + 1e-9)
        cos_mirror = np.where(sp_d > 1, (vC * vD).sum(1) / (sp_c * sp_d + 1e-9), -1)
        moving = sp_c > 1
        need = PURSUIT_BASE + PURSUIT_PER_YD * (d_now - BAIT_CLOSE_YDS)
        if moving.sum() >= 3 and np.nanmean(np.maximum(cos_at, cos_mirror)[moving]) >= need:
            out.append(c)
    return out


def play_reps(at: pl.DataFrame, snap: pl.DataFrame, pff: pl.DataFrame, tgt: dict, paths: dict | None = None) -> list[dict]:
    """One row per decoy route per play, plus a play summary row (nflId = None).

    Two scores per decoy:
      score_raw  — the brief as written: attached defenders x their distance to the target.
                   Confounded by throw depth (a deep target is far from everyone).
      score      — drag: for each attached defender, how many yards FURTHER from the target
                   he is at the throw than at the snap (floored at 0). Yards of coverage
                   the decoy actually pulled off the target.
      score_adj  — score_raw / (1 + teammates inside the decoy's 5-yd zone). If other offensive
                   players are standing in his zone, the defenders there may be covering them,
                   so the decoy gets a share of the credit, not all of it.
    """
    key = ["gameId", "playId", "nflId"]
    roles = pff.select(key + ["pff_role"])
    at = at.join(roles, on=key, how="left")
    s0 = {(g, p, n): (x, y) for g, p, n, x, y in
          snap.select(key + ["x", "y"]).iter_rows() if n is not None}
    los = {(g, p): x for g, p, x in
           snap.filter(pl.col("nflId").is_null()).select(["gameId", "playId", "x"]).iter_rows()}
    out = []
    for (g, p), grp in at.group_by(["gameId", "playId"]):
        tid = tgt.get((g, p))
        if tid is None or (g, p, tid) not in s0:
            continue
        role = grp["pff_role"].str.to_lowercase()
        route = grp.filter(role == "pass route")
        cov = grp.filter(role == "coverage")
        mates = grp.filter(role.is_in(["pass route", "pass block"]))     # offense minus the QB
        M = mates.select(["x", "y"]).to_numpy()
        mid = mates["nflId"].to_numpy()
        t = route.filter(pl.col("nflId") == tid)
        if t.is_empty() or cov.is_empty():
            continue
        T = np.array([t["x"][0], t["y"][0]])
        T0 = np.array(s0[(g, p, tid)])
        C = cov.select(["x", "y"]).to_numpy()
        cid = cov["nflId"].to_numpy()
        C0 = np.array([s0.get((g, p, int(i)), (np.nan, np.nan)) for i in cid])
        d_to_t = np.hypot(*(C - T).T)
        d0_to_t = np.hypot(*(C0 - T0).T)
        drag = np.nan_to_num(np.clip(d_to_t - d0_to_t, 0, None))
        on_target = d_to_t < ATTACH_YDS
        decoys = route.filter(pl.col("nflId") != tid)
        P = (paths or {}).get((g, p), {})
        xy = {n: (x, y) for n, x, y in grp.select(["nflId", "x", "y"]).iter_rows() if n is not None}
        ontgt = {int(c): bool(o) for c, o in zip(cid, on_target)}
        rids = route["nflId"].to_list()
        cidx = {int(c): i for i, c in enumerate(cid)}
        tot_raw = tot = 0.0
        n_multi = 0
        for nid, x, y in decoys.select(["nflId", "x", "y"]).iter_rows():
            dd = np.hypot(C[:, 0] - x, C[:, 1] - y)
            near5 = (dd < ATTACH_YDS) & ~on_target
            if P:
                bait = baited(P, nid, rids, [int(c) for c in cid], xy, ontgt)
                att = np.zeros(len(cid), bool)
                att[[cidx[b] for b in bait]] = True
            else:
                att = near5
            n = int(att.sum())
            raw = float(d_to_t[att].sum())
            tm = int(((np.hypot(M[:, 0] - x, M[:, 1] - y) < ATTACH_YDS) & (mid != nid)).sum())
            sc = float(drag[att].sum())
            tot_raw += raw
            tot += sc
            n_multi += n >= 2
            out.append(dict(gameId=g, playId=p, nflId=nid, targetId=tid, attached=n,
                            score_raw=round(raw, 2), score=round(sc, 2),
                            teammates=tm, score_adj=round(raw / (1 + tm), 2),
                            dist_to_target=round(float(np.hypot(x - T[0], y - T[1])), 2),
                            depth=round(x - los.get((g, p), np.nan), 2),
                            width=round(abs(y - 26.65), 2), ncov=len(cid),
                            n_route=route.height, attached5=int(near5.sum()),
                            bait_radius=round(float(dd[att].max()), 2) if n else 0.0,
                            attachedIds=[int(i) for i in cid[att]]))
        out.append(dict(gameId=g, playId=p, nflId=None, targetId=tid,
                        score_raw=round(tot_raw, 2), score=round(tot, 2),
                        target_sep=round(float(d_to_t.min()), 2), multi_decoys=n_multi,
                        air_depth=round(float(T[0] - los.get((g, p), np.nan)), 2)))
    return out


def run(d: BDB | None = None):
    d = d or BDB()
    tg = targets(d)
    tgt = dict(zip(zip(tg["gameId"], tg["playId"]), tg["targetId"]))
    pff = d.pff()
    rows = []
    for k, gid in enumerate(d.game_ids()):
        at, tr = throw_frames(d, gid)
        snap = tr.filter(pl.col("frameId") == pl.col("snapFrame"))
        win = tr.filter((pl.col("frameId") <= pl.col("throwFrame"))
                        & (pl.col("frameId") >= pl.col("throwFrame") - BAIT_WINDOW))
        rows += play_reps(at, snap, pff, tgt, _paths(win))
        if k % 20 == 0:
            print(f"  game {k + 1}/{len(d.game_ids())}  rows={len(rows):,}", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    reps = (df.filter(pl.col("nflId").is_not_null())
              .drop(["target_sep", "multi_decoys", "air_depth"]))
    plays = (df.filter(pl.col("nflId").is_null())
               .select(["gameId", "playId", "targetId", "score_raw", "score",
                        "target_sep", "multi_decoys", "air_depth"]))
    os.makedirs(OUT, exist_ok=True)
    reps.write_parquet(os.path.join(OUT, "decoy_reps.parquet"))
    plays.write_parquet(os.path.join(OUT, "decoy_plays.parquet"))
    print(f"targets matched: {tg.height:,}  decoy reps: {reps.height:,}  plays: {plays.height:,}")
    return reps, plays


if __name__ == "__main__":
    run()
