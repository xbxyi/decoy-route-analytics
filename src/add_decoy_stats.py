"""Add the brief's Decoy Efficiency numbers to every route runner in app_profiles/index.json.

    python decoy.py            # per-route decoy geometry -> out/decoy_reps.parquet
    python add_decoy_stats.py  # merge into the Film Room index

Per decoy route (a route run while the ball went to someone else), at the throw:
  defenders  = coverage defenders within 5 yd of him (and not within 5 yd of the target)
  score      = defenders x their distance from the target, in yards  (the brief's formula)
Per player: decoy routes, average score, average defenders drawn, share of routes drawing 2+.
"""
from __future__ import annotations

import json
import os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = os.path.join(HERE, "..", "out", "decoy_reps.parquet")
INDEX = os.path.join(HERE, "..", "app_profiles", "index.json")
QUALIFY = 40


def main():
    r = pl.read_parquet(REPS)
    agg = (r.group_by("nflId").agg(pl.len().alias("decoys"),
                                   pl.col("score_raw").mean().alias("decoy_score"),
                                   pl.col("attached").mean().alias("decoy_drawn"),
                                   (pl.col("attached") >= 2).mean().alias("decoy_multi"),
                                   (pl.col("attached") >= 2).sum().alias("decoy_multi_n")))
    stats = {row["nflId"]: row for row in agg.iter_rows(named=True)}
    q = agg.filter(pl.col("decoys") >= QUALIFY)
    pools = {c: q[c].to_numpy() for c in ("decoy_score", "decoy_multi", "decoy_drawn")}
    per_game = (r.group_by(["nflId", "gameId"]).agg(pl.len().alias("n"),
                                                    (pl.col("attached") >= 1).sum().alias("drew"),
                                                    (pl.col("attached") >= 2).sum().alias("multi"),
                                                    pl.col("score_raw").max().alias("best"),
                                                    pl.col("score_raw").mean().alias("avg")))
    games = {(row["nflId"], row["gameId"]): row for row in per_game.iter_rows(named=True)}
    best = r.group_by("nflId").agg(pl.col("score_raw").max().alias("best"))
    best = dict(best.iter_rows())

    idx = json.load(open(INDEX, encoding="utf-8"))
    n = 0
    for p in idx["players"]:
        rt = p["roles"].get("route")
        st = stats.get(p["id"])
        if not rt or not st:
            continue
        for g in p["games"]:
            pg = games.get((p["id"], g.get("gameId")))
            if g.get("role") == "route" and pg:
                g["decoy"] = dict(n=pg["n"], drew=pg["drew"], multi=pg["multi"], best=round(pg["best"], 1), avg=round(pg["avg"], 1))
        rt.update(decoy_best=round(best[p["id"]], 1), decoys=st["decoys"], decoy_score=round(st["decoy_score"], 1), decoy_drawn=round(st["decoy_drawn"], 2),
                  decoy_multi=round(st["decoy_multi"], 3), decoy_multi_n=st["decoy_multi_n"])
        for c, pool in pools.items():
            v = st[c]
            rt[f"p_{c}"] = (round(100 * ((pool < v).mean() + 0.5 * (pool == v).mean()))
                            if st["decoys"] >= QUALIFY and len(pool) else None)
        n += 1
    idx["meta"]["decoy_qualify"] = QUALIFY
    json.dump(idx, open(INDEX, "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
    top = sorted((p for p in idx["players"] if p["roles"].get("route", {}).get("p_decoy_score") is not None),
                 key=lambda p: -p["roles"]["route"]["decoy_score"])[:8]
    print(f"decoy stats added for {n} route runners; top by brief score:",
          [(p["name"], p["roles"]["route"]["decoy_score"], p["roles"]["route"]["decoys"]) for p in top])


if __name__ == "__main__":
    main()
