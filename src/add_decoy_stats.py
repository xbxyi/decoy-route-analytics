"""Add the brief's Decoy Efficiency numbers to every route runner in app_profiles/index.json.

    python decoy.py            # per-route decoy geometry -> out/decoy_reps.parquet
    python add_decoy_stats.py  # merge into the Film Room index

Per decoy route (a route run while the ball went to someone else), at the throw:
  defenders  = coverage defenders within 5 yd of him (and not within 5 yd of the target)
  score      = defenders x their distance from the target, in yards  (the brief's formula),
               divided by (1 + teammates inside his 5-yd zone) so crowded spots share the credit
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


def coach_view(r: pl.DataFrame) -> tuple[dict, dict]:
    """Per player, for coaches: what his decoy work does for the target, and where he lures best.

    impact   — on his decoy routes, the targeted receiver's separation at the throw (nearest coverage
               defender), completion rate and yards, split by whether he lured 2+ defenders or 0-1
    coverage — defenders lured per route vs man and vs zone (PFF coverage call)
    depth    — defenders lured per route by how deep he was at the throw (yards past the line)
    weekly   — average decoy score per week
    League averages are returned for the same splits so each player can be read against them.
    """
    from bdb import BDB
    d = BDB()
    plays = (pl.read_parquet(os.path.join(HERE, "..", "out", "decoy_plays.parquet")).select(["gameId", "playId", "target_sep"])
               .join(d.plays().select(["gameId", "playId", "passResult", "playResult", "pff_passCoverageType", "week"]),
                     on=["gameId", "playId"]))
    x = (r.join(plays, on=["gameId", "playId"])
          .with_columns((pl.col("passResult") == "C").cast(pl.Float64).alias("comp"),
                        pl.when(pl.col("attached") >= 2).then(pl.lit("lure2")).otherwise(pl.lit("lure01")).alias("lg"),
                        pl.when(pl.col("depth") < 5).then(pl.lit("short")).when(pl.col("depth") < 15)
                          .then(pl.lit("mid")).otherwise(pl.lit("deep")).alias("db"),
                        pl.col("pff_passCoverageType").fill_null("Other").alias("cov")))
    rnd = lambda v, k=2: None if v is None else round(float(v), k)
    out = {}
    imp = x.group_by(["nflId", "lg"]).agg(pl.len().alias("n"), pl.col("target_sep").mean().alias("sep"),
                                          pl.col("comp").mean().alias("comp"), pl.col("playResult").mean().alias("ypp"))
    cov = x.filter(pl.col("cov").is_in(["Man", "Zone"])).group_by(["nflId", "cov"]).agg(pl.len().alias("n"), pl.col("attached").mean().alias("lured"))
    dep = x.group_by(["nflId", "db"]).agg(pl.len().alias("n"), pl.col("attached").mean().alias("lured"))
    wk = x.group_by(["nflId", "week"]).agg(pl.len().alias("n"), pl.col("score_adj").mean().alias("score")).sort("week")
    for row in imp.iter_rows(named=True):
        out.setdefault(row["nflId"], {}).setdefault("impact", {})[row["lg"]] = dict(
            n=row["n"], sep=rnd(row["sep"]), comp=rnd(row["comp"], 3), ypp=rnd(row["ypp"], 1))
    for row in cov.iter_rows(named=True):
        out.setdefault(row["nflId"], {}).setdefault("coverage", {})[row["cov"].lower()] = dict(n=row["n"], lured=rnd(row["lured"]))
    for row in dep.iter_rows(named=True):
        out.setdefault(row["nflId"], {}).setdefault("depth", {})[row["db"]] = dict(n=row["n"], lured=rnd(row["lured"]))
    for row in wk.iter_rows(named=True):
        out.setdefault(row["nflId"], {}).setdefault("weekly", []).append(dict(week=row["week"], n=row["n"], score=rnd(row["score"], 1)))
    league = dict(
        coverage={k.lower(): rnd(v) for k, v in x.filter(pl.col("cov").is_in(["Man", "Zone"])).group_by("cov").agg(pl.col("attached").mean()).iter_rows()},
        depth={k: rnd(v) for k, v in x.group_by("db").agg(pl.col("attached").mean()).iter_rows()},
        impact={k: dict(sep=rnd(a), comp=rnd(b, 3), ypp=rnd(c, 1)) for k, a, b, c in
                x.group_by("lg").agg(pl.col("target_sep").mean(), pl.col("comp").mean(), pl.col("playResult").mean()).iter_rows()},
        score=rnd(x["score_adj"].mean(), 1))
    return out, league


def main():
    r = pl.read_parquet(REPS)
    coach, league = coach_view(r)
    agg = (r.group_by("nflId").agg(pl.len().alias("decoys"),
                                   pl.col("score_adj").mean().alias("decoy_score"),
                                   pl.col("attached").mean().alias("decoy_drawn"),
                                   (pl.col("attached") >= 2).mean().alias("decoy_multi"),
                                   (pl.col("attached") >= 2).sum().alias("decoy_multi_n"),
                                   ((pl.col("attached") >= 2) & (pl.col("score") > 0)).sum().alias("decoy_real_n")))
    stats = {row["nflId"]: row for row in agg.iter_rows(named=True)}
    q = agg.filter(pl.col("decoys") >= QUALIFY)
    pools = {c: q[c].to_numpy() for c in ("decoy_score", "decoy_multi", "decoy_drawn")}
    per_game = (r.group_by(["nflId", "gameId"]).agg(pl.len().alias("n"),
                                                    (pl.col("attached") >= 1).sum().alias("drew"),
                                                    (pl.col("attached") >= 2).sum().alias("multi"),
                                                    pl.col("score_adj").max().alias("best"),
                                                    pl.col("score_adj").mean().alias("avg"),
                                                    ((pl.col("attached") >= 2) & (pl.col("score") > 0)).sum().alias("real")))
    games = {(row["nflId"], row["gameId"]): row for row in per_game.iter_rows(named=True)}
    best = r.group_by("nflId").agg(pl.col("score_adj").max().alias("best"))
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
                g["decoy"] = dict(n=pg["n"], drew=pg["drew"], multi=pg["multi"], real=pg["real"],
                                  best=round(pg["best"], 1), avg=round(pg["avg"], 1))
        rt["coach"] = coach.get(p["id"], {})
        rt.update(decoy_best=round(best[p["id"]], 1), decoys=st["decoys"], decoy_score=round(st["decoy_score"], 1), decoy_drawn=round(st["decoy_drawn"], 2),
                  decoy_multi=round(st["decoy_multi"], 3), decoy_multi_n=st["decoy_multi_n"], decoy_real_n=st["decoy_real_n"])
        for c, pool in pools.items():
            v = st[c]
            rt[f"p_{c}"] = (round(100 * ((pool < v).mean() + 0.5 * (pool == v).mean()))
                            if st["decoys"] >= QUALIFY and len(pool) else None)
        n += 1
    idx["meta"]["decoy_qualify"] = QUALIFY
    idx["meta"]["league"] = league
    json.dump(idx, open(INDEX, "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
    top = sorted((p for p in idx["players"] if p["roles"].get("route", {}).get("p_decoy_score") is not None),
                 key=lambda p: -p["roles"]["route"]["decoy_score"])[:8]
    print(f"decoy stats added for {n} route runners; top by brief score:",
          [(p["name"], p["roles"]["route"]["decoy_score"], p["roles"]["route"]["decoys"]) for p in top])


if __name__ == "__main__":
    main()
