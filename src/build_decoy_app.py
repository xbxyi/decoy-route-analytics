"""Build app/decoy.json for the Decoy Efficiency web app.

    python decoy.py              # per-rep geometry -> out/
    python build_decoy_app.py    # expectation model, leaderboard, showcase plays -> app/decoy.json

Re-runnable on any season with the same BDB file layout.
"""
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from bdb import BDB
from decoy import ATTACH_YDS, throw_frames

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
APP = os.path.join(HERE, "..", "app")

MIN_DECOYS = 80           # decoy routes to qualify for the leaderboard
SHOWCASE = 3              # animated plays per player
FEATURES = ["depth", "width", "dist_to_target", "ncov", "n_route", "man", "is_te", "is_rb"]


def expected_attachment(r: pl.DataFrame) -> pl.DataFrame:
    """Cross-fitted (odd weeks predict even, and vice versa) so no rep grades itself."""
    X = r.select(FEATURES).to_numpy()
    y = r["attached"].to_numpy().astype(float)
    odd = (r["week"] % 2 == 1).to_numpy()
    exp = np.zeros(len(y))
    for fit in (odd, ~odd):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        exp[~fit] = m.fit(X[fit], y[fit]).predict(X[~fit])
    r2 = 1 - ((y - exp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"expected-attachment model R^2 (out of fold): {r2:.3f}")
    return r.with_columns(pl.Series("exp_att", exp)).with_columns(
        (pl.col("attached") - pl.col("exp_att")).alias("goe"))


def split_half(r: pl.DataFrame, col: str) -> float:
    a = (r.with_columns((pl.col("week") % 2).alias("h"))
          .group_by(["nflId", "h"]).agg(pl.col(col).mean(), pl.len())
          .filter(pl.col("len") >= 40).pivot(on="h", index="nflId", values=col).drop_nulls())
    return round(a.select(pl.corr("0", "1")).item(), 3)


def validation(d: BDB, r: pl.DataFrame) -> dict:
    """Does decoy gravity open up the target? Compare within (coverage count x route count x depth)."""
    pff = d.pff()
    ncov = (pff.filter(pl.col("pff_role").str.to_lowercase() == "coverage")
               .group_by(["gameId", "playId"]).agg(pl.len().alias("ncov")))
    pp = (pl.read_parquet(os.path.join(OUT, "decoy_plays.parquet"))
            .join(d.plays().select(["gameId", "playId", "passResult", "playResult"]),
                  on=["gameId", "playId"])
            .join(ncov, on=["gameId", "playId"])
            .join(r.group_by(["gameId", "playId"]).agg(pl.col("attached").sum().alias("att"),
                                                        pl.col("goe").sum().alias("goe"),
                                                        pl.col("n_route").first()),
                  on=["gameId", "playId"])
            .with_columns((pl.col("passResult") == "C").cast(pl.Float64).alias("comp"),
                          pl.col("air_depth").cut([0, 5, 10, 20]).alias("db")))
    cell = ["ncov", "n_route", "db"]
    cols = ["score_raw", "score", "att", "goe", "target_sep", "comp", "playResult"]
    rr = pp.with_columns([(pl.col(c) - pl.col(c).mean().over(cell)).alias(c) for c in cols])
    corr = lambda a, b: round(rr.select(pl.corr(a, b)).item(), 3)
    # quintile table on raw formula within depth band — shows the depth confound plainly
    q = (pp.with_columns(pl.when(pl.col("air_depth") < 5).then(pl.lit("Behind / at sticks (<5 yds)"))
                           .when(pl.col("air_depth") < 15).then(pl.lit("Intermediate (5–15)"))
                           .otherwise(pl.lit("Deep (15+)")).alias("band"))
           .with_columns((pl.col("score_raw").rank() / pl.len()).over("band").alias("pct")))
    bands = []
    for band in ["Behind / at sticks (<5 yds)", "Intermediate (5–15)", "Deep (15+)"]:
        b = q.filter(pl.col("band") == band)
        top, bot = b.filter(pl.col("pct") > 0.8), b.filter(pl.col("pct") <= 0.2)
        bands.append(dict(band=band, n=b.height,
                          comp_top=round(top["comp"].mean(), 3), comp_bot=round(bot["comp"].mean(), 3),
                          sep_top=round(top["target_sep"].mean(), 2), sep_bot=round(bot["target_sep"].mean(), 2)))
    out = dict(plays=pp.height,
               r_sep=dict(raw=corr("score_raw", "target_sep"), drag=corr("score", "target_sep"),
                          attached=corr("att", "target_sep"), goe=corr("goe", "target_sep")),
               r_comp=dict(raw=corr("score_raw", "comp"), drag=corr("score", "comp"),
                           attached=corr("att", "comp"), goe=corr("goe", "comp")),
               bands=bands)
    print("validation:", json.dumps(out, indent=1))
    return out


def sensitivity(r: pl.DataFrame, d: BDB) -> list[dict]:
    """Re-score attachment at other radii from stored distances is impossible (we keep counts only),
    so rebuild counts for a sample of games at 3/5/7 yds and report rank stability of the top list."""
    import decoy
    from decoy import targets, play_reps
    tg = targets(d)
    tgt = dict(zip(zip(tg["gameId"], tg["playId"]), tg["targetId"]))
    pff = d.pff()
    games = d.game_ids()
    frames = []
    for gid in games:
        at, tr = throw_frames(d, gid)
        frames.append((at, tr.filter(pl.col("frameId") == pl.col("snapFrame"))))
    res = {}
    for rad in (3.0, 5.0, 7.0):
        decoy.ATTACH_YDS = rad
        rows = []
        for at, sn in frames:
            rows += [x for x in play_reps(at, sn, pff, tgt) if x.get("nflId") is not None]
        df = pl.DataFrame(rows, infer_schema_length=None)
        res[rad] = (df.group_by("nflId").agg(pl.col("attached").mean().alias(f"a{int(rad)}"), pl.len())
                      .filter(pl.col("len") >= MIN_DECOYS).drop("len"))
    decoy.ATTACH_YDS = ATTACH_YDS
    m = res[3.0].join(res[5.0], on="nflId").join(res[7.0], on="nflId")
    rank = lambda c: m[c].rank().to_numpy()
    sp = lambda a, b: round(float(np.corrcoef(rank(a), rank(b))[0, 1]), 3)
    out = [dict(radius=3, rank_corr_vs_5=sp("a3", "a5"), mean=round(m["a3"].mean(), 2)),
           dict(radius=5, rank_corr_vs_5=1.0, mean=round(m["a5"].mean(), 2)),
           dict(radius=7, rank_corr_vs_5=sp("a7", "a5"), mean=round(m["a7"].mean(), 2))]
    print("sensitivity:", out)
    return out


def showcase_frames(d: BDB, picks: pl.DataFrame) -> dict:
    """snap -> throw (+0.5 s) at 5 Hz for each showcase play: [[id, x, y], ...] per frame."""
    pff = d.pff().select(["gameId", "playId", "nflId", "pff_role"])
    out = {}
    for gid in picks["gameId"].unique().to_list():
        _, tr = throw_frames(d, gid)
        want = picks.filter(pl.col("gameId") == gid)["playId"].unique().to_list()
        tr = (tr.filter(pl.col("playId").is_in(want)
                        & (pl.col("frameId") >= pl.col("snapFrame"))
                        & (pl.col("frameId") <= pl.col("throwFrame") + 5)
                        & (((pl.col("frameId") - pl.col("snapFrame")) % 2 == 0)
                           | (pl.col("frameId") == pl.col("throwFrame"))))
                .join(pff, on=["gameId", "playId", "nflId"], how="left"))
        for (pid,), g in tr.group_by(["playId"]):
            g = g.sort(["frameId", "nflId"], nulls_last=True)
            snap, thr = g["snapFrame"][0], g["throwFrame"][0]
            frames, ids = [], {}
            for (fid,), f in g.group_by(["frameId"], maintain_order=True):
                frames.append([fid - snap] + [[int(n) if n is not None else 0, round(x, 1), round(y, 1)]
                                              for n, x, y in f.select(["nflId", "x", "y"]).iter_rows()])
            for n, role, team in g.select(["nflId", "pff_role", "team"]).unique().iter_rows():
                if n is not None:
                    ids[int(n)] = (role or "").lower().replace("pass ", "")
            out[f"{gid}-{pid}"] = dict(throw=thr - snap, frames=frames, roles=ids)
    return out


def main():
    d = BDB()
    P = d.plays()
    pl_ = d.players().select(["nflId", "displayName", "officialPosition"])
    r = (pl.read_parquet(os.path.join(OUT, "decoy_reps.parquet"))
           .join(P.select(["gameId", "playId", "pff_passCoverageType", "week"]), on=["gameId", "playId"])
           .join(pl_, on="nflId", how="left")
           .with_columns((pl.col("pff_passCoverageType") == "Man").cast(pl.Int8).alias("man"),
                         (pl.col("officialPosition") == "TE").cast(pl.Int8).alias("is_te"),
                         (pl.col("officialPosition").is_in(["RB", "FB"])).cast(pl.Int8).alias("is_rb"))
           .drop_nulls(["depth", "width", "dist_to_target", "ncov", "n_route"]))
    r = expected_attachment(r)
    rel = dict(attached=split_half(r, "attached"), goe=split_half(r, "goe"),
               raw=split_half(r, "score_raw"), drag=split_half(r, "score"))
    print("split-half reliability:", rel)
    val = validation(d, r)
    sens = sensitivity(r, d)

    # team per player = the team they ran routes for most
    tm = (r.join(P.select(["gameId", "playId", "possessionTeam"]), on=["gameId", "playId"])
            .group_by(["nflId", "possessionTeam"]).len().sort("len", descending=True)
            .group_by("nflId").agg(pl.col("possessionTeam").first().alias("team")))
    lb = (r.group_by("nflId").agg(
              pl.len().alias("decoys"),
              pl.col("attached").mean().alias("drawn"),
              pl.col("exp_att").mean().alias("expected"),
              pl.col("goe").mean().alias("goe"),
              (pl.col("attached") >= 2).mean().alias("multi_rate"),
              (pl.col("attached") >= 2).sum().alias("multi_n"),
              pl.col("score").mean().alias("drag"),
              pl.col("score_raw").mean().alias("raw"),
              pl.col("man").mean().alias("man_share"))
            .filter(pl.col("decoys") >= MIN_DECOYS)
            .join(pl_, on="nflId").join(tm, on="nflId")
            .sort("goe", descending=True)
            .with_columns(pl.int_range(1, pl.len() + 1).alias("rank")))
    print(lb.select(["rank", "displayName", "officialPosition", "team", "decoys", "drawn",
                     "expected", "goe", "multi_rate"]).head(15))

    # showcase: each qualified player's biggest decoy reps (most defenders drawn, then most drag)
    pk = (r.filter(pl.col("nflId").is_in(lb["nflId"]) & (pl.col("attached") >= 1))
            .sort(["attached", "score"], descending=True)
            .group_by("nflId", maintain_order=True).head(SHOWCASE))
    frames = showcase_frames(d, pk)
    names = dict(zip(pl_["nflId"], pl_["displayName"]))
    pinfo = P.select(["gameId", "playId", "playDescription", "week", "gameDate", "possessionTeam",
                      "defensiveTeam", "quarter", "gameClock", "down", "yardsToGo", "passResult",
                      "playResult", "pff_passCoverage", "pff_passCoverageType"])
    pk = pk.join(pinfo.drop("week"), on=["gameId", "playId"])
    plays_by_player: dict[int, list] = {}
    for row in pk.iter_rows(named=True):
        key = f"{row['gameId']}-{row['playId']}"
        if key not in frames:
            continue
        plays_by_player.setdefault(row["nflId"], []).append(dict(
            key=key, week=row["week"], date=row["gameDate"], off=row["possessionTeam"],
            dff=row["defensiveTeam"], q=row["quarter"], clock=row["gameClock"],
            down=row["down"], togo=row["yardsToGo"], desc=row["playDescription"],
            result=row["passResult"], yds=row["playResult"],
            coverage=row["pff_passCoverage"], ctype=row["pff_passCoverageType"],
            target=row["targetId"], targetName=names.get(row["targetId"], ""),
            attached=row["attached"], attachedIds=row["attachedIds"],
            attachedNames=[names.get(i, "") for i in row["attachedIds"]],
            drag=row["score"], expected=round(row["exp_att"], 2)))

    players = []
    for row in lb.iter_rows(named=True):
        players.append(dict(
            id=row["nflId"], name=row["displayName"], pos=row["officialPosition"], team=row["team"],
            rank=row["rank"], decoys=row["decoys"], drawn=round(row["drawn"], 3),
            expected=round(row["expected"], 3), goe=round(row["goe"], 3),
            multi_rate=round(row["multi_rate"], 3), multi_n=row["multi_n"],
            drag=round(row["drag"], 2), raw=round(row["raw"], 1), man_share=round(row["man_share"], 2),
            plays=plays_by_player.get(row["nflId"], [])))

    used = {p["key"] for pl_list in plays_by_player.values() for p in pl_list}
    data = dict(
        meta=dict(season=2021, weeks="1–8", games=len(d.game_ids()), throws=int(r.select(pl.struct("gameId", "playId").n_unique()).item()),
                  decoy_routes=r.height, attach_yds=ATTACH_YDS, min_decoys=MIN_DECOYS,
                  qualified=len(players)),
        reliability=rel, validation=val, sensitivity=sens,
        players=players,
        frames={k: v for k, v in frames.items() if k in used},
        names={int(k): v for k, v in names.items()})
    os.makedirs(APP, exist_ok=True)
    path = os.path.join(APP, "decoy.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {path}  {os.path.getsize(path) / 1e6:.2f} MB  players={len(players)}  plays={len(used)}")


if __name__ == "__main__":
    main()
