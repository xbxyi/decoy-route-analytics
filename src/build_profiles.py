"""Player profiles for every player in the dataset -> app_profiles/

    python build_profiles.py

One pass over all tracking. For each play: freeze the field at the throw (or the sack /
last frame when there is no throw) and score every player by the role PFF charted him in:

  route     targets, catches, yards, separation at the throw, top speed
  rush      pressures / sacks / hits / hurries (PFF), top speed
  block     pressures / sacks allowed, beaten (PFF)
  coverage  distance to nearest route runner at the throw; when he was the closest
            defender to the target: targets, completions, yards, INTs allowed
  qb        attempts, completions, yards, sacks, time to throw, pressured dropbacks

Output: app_profiles/index.json (every player, summary + percentiles + game log)
        app_profiles/film_<TEAM>.json (animated showcase plays for that team's players)
"""
from __future__ import annotations

import html
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import polars as pl

from bdb import BDB
from decoy import targets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "app_profiles")
THROW = ("pass_forward", "autoevent_passforward")
END = ("qb_sack", "qb_strip_sack", "run", "fumble", "pass_shovel")
MPH = 2.04545          # yd/s -> mph
QUALIFY = 40           # snaps in a role to get percentiles
PRE_SNAP = 15          # replay frames before the snap (1.5 s)
POST_KEY = 30          # replay frames after the throw / end event (3 s)
PER_GAME = 1           # replay plays stored per player per game: his best real decoy play
REAL_DECOY_MIN = 2     # defenders within 5 yd at the throw for a play to count as a real decoy play
SEASON_BEST = 3        # flagged as season highlights

ROLE = {"pass route": "route", "pass rush": "rush", "pass block": "block",
        "coverage": "coverage", "pass": "qb"}


def play_rows(d: BDB, gid: int, pff: pl.DataFrame, plays: dict, tgt: dict):
    tr = d.standardize(d.tracking(gid))
    ev = tr.filter(pl.col("event").is_not_null()).select(["playId", "frameId", "event"]).unique()
    snap = (ev.filter(pl.col("event").is_in(["ball_snap", "autoevent_ballsnap"]))
              .group_by("playId").agg(pl.col("frameId").min().alias("snap")))
    thr = ev.filter(pl.col("event").is_in(THROW)).group_by("playId").agg(pl.col("frameId").min().alias("thr"))
    end = ev.filter(pl.col("event").is_in(END)).group_by("playId").agg(pl.col("frameId").min().alias("endev"))
    last = tr.group_by("playId").agg(pl.col("frameId").max().alias("last"))
    fr = (snap.join(thr, on="playId", how="left").join(end, on="playId", how="left").join(last, on="playId")
              .with_columns(pl.coalesce("thr", "endev", "last").alias("key")))
    tr = (tr.join(fr, on="playId")
            .filter((pl.col("frameId") >= pl.col("snap") - PRE_SNAP) & (pl.col("frameId") <= pl.col("key") + POST_KEY))
            .join(pff, on=["gameId", "playId", "nflId"], how="left"))
    rows, team_of = [], {}
    for (pid,), g in tr.group_by(["playId"]):
        p = plays.get((gid, pid))
        if p is None:
            continue
        snapf, key = g["snap"][0], g["key"][0]
        thrown = g["thr"][0] is not None
        tt = (key - snapf) / 10 if thrown else None
        top = (g.filter((pl.col("frameId") >= snapf) & (pl.col("frameId") <= key)).group_by("nflId").agg(pl.col("s").max())
                .drop_nulls().rows())
        topspd = dict(top)
        at = g.filter((pl.col("frameId") == key) & pl.col("nflId").is_not_null())
        role = {n: ROLE.get((r or "").lower()) for n, r in at.select(["nflId", "pff_role"]).iter_rows()}
        xy = {n: (x, y) for n, x, y in at.select(["nflId", "x", "y"]).iter_rows()}
        for n, t, j in at.select(["nflId", "team", "jerseyNumber"]).iter_rows():
            team_of.setdefault(n, defaultdict(int))[(t, j)] += 1
        lab = {r[0]: r for r in at.select(["nflId", "pff_hit", "pff_hurry", "pff_sack", "pff_beatenByDefender",
                                            "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"]).iter_rows()}
        routes = [n for n, r in role.items() if r == "route"]
        covs = [n for n, r in role.items() if r == "coverage"]
        R = np.array([xy[n] for n in routes]) if routes else np.zeros((0, 2))
        C = np.array([xy[n] for n in covs]) if covs else np.zeros((0, 2))
        target = tgt.get((gid, pid))
        res, yds = p["passResult"], p["playResult"]
        primary = None
        if target in xy and len(C):
            primary = covs[int(np.argmin(np.hypot(*(C - np.array(xy[target])).T)))]
        pressured = any((lab[n][1] or 0) + (lab[n][2] or 0) + (lab[n][3] or 0) > 0
                        for n, r in role.items() if r == "rush")
        base = dict(gameId=gid, playId=pid, tt=tt)
        for n, r in role.items():
            if r is None:
                continue
            row = dict(base, nflId=n, role=r, top=topspd.get(n))
            if r == "route":
                sep = float(np.hypot(*(C - np.array(xy[n])).T).min()) if len(C) else None
                tg = n == target
                row.update(sep=sep, target=tg, catch=tg and res == "C", yds=yds if tg and res == "C" else 0)
            elif r == "coverage":
                near = float(np.hypot(*(R - np.array(xy[n])).T).min()) if len(R) else None
                pr = n == primary
                row.update(near=near, targeted=pr, comp=pr and res == "C", yds=yds if pr and res == "C" else 0,
                           int=pr and res == "IN")
            elif r == "rush":
                h, hu, s = (lab[n][1] or 0), (lab[n][2] or 0), (lab[n][3] or 0)
                row.update(hit=h, hurry=hu, sack=s, pressure=int(h + hu + s > 0))
            elif r == "block":
                h, hu, s = (lab[n][5] or 0), (lab[n][6] or 0), (lab[n][7] or 0)
                row.update(allowed=int(h + hu + s > 0), sack_allowed=s, beaten=lab[n][4] or 0)
            elif r == "qb":
                row.update(att=res in ("C", "I", "IN"), comp=res == "C", yds=yds if res == "C" else 0,
                           sack=res == "S", int=res == "IN", pressured=pressured)
            rows.append(row)
    return rows, team_of, tr


def frames_for(tr: pl.DataFrame, pid: int) -> dict:
    """Compact replay: ids once, then per frame [t, x1*10, y1*10, x2*10, ...] (ball id 0)."""
    g = tr.filter(pl.col("playId") == pid)
    snapf, key = g["snap"][0], g["key"][0]
    g = g.filter(((pl.col("frameId") - snapf + 1000) % 2 == 0) | (pl.col("frameId") == key)).with_columns(pl.col("nflId").fill_null(0))
    ids = sorted(g["nflId"].unique().to_list())
    pos = {(f, n): (x, y) for f, n, x, y in g.select(["frameId", "nflId", "x", "y"]).iter_rows()}
    frames = []
    for fid in sorted(g["frameId"].unique().to_list()):
        row = [fid - snapf]
        for n in ids:
            x, y = pos.get((fid, n), (None, None))
            row += [round(x * 10), round(y * 10)] if x is not None else [-1, -1]
        frames.append(row)
    roles = {int(n): ROLE.get((r or "").lower(), "") for n, r in
             g.select(["nflId", "pff_role"]).unique().iter_rows() if n}
    return dict(key=key - snapf, thrown=g["thr"][0] is not None, ids=ids, frames=frames, roles=roles)


AGG = {
    "route": [("routes", pl.len()), ("targets", pl.col("target").sum()), ("catches", pl.col("catch").sum()),
              ("yds", pl.col("yds").sum()), ("sep", pl.col("sep").mean()),
              ("sep_tgt", pl.col("sep").filter(pl.col("target")).mean()), ("top", pl.col("top").max())],
    "rush": [("rushes", pl.len()), ("pressures", pl.col("pressure").sum()), ("sacks", pl.col("sack").sum()),
             ("hits", pl.col("hit").sum()), ("hurries", pl.col("hurry").sum()), ("top", pl.col("top").max())],
    "block": [("blocks", pl.len()), ("allowed", pl.col("allowed").sum()),
              ("sacks_allowed", pl.col("sack_allowed").sum()), ("beaten", pl.col("beaten").sum())],
    "coverage": [("snaps", pl.len()), ("near", pl.col("near").mean()), ("targeted", pl.col("targeted").sum()),
                 ("comp", pl.col("comp").sum()), ("yds", pl.col("yds").sum()), ("ints", pl.col("int").sum()),
                 ("top", pl.col("top").max())],
    "qb": [("dropbacks", pl.len()), ("att", pl.col("att").sum()), ("comp", pl.col("comp").sum()),
           ("yds", pl.col("yds").sum()), ("sacks", pl.col("sack").sum()), ("ints", pl.col("int").sum()),
           ("tt", pl.col("tt").mean()), ("pressured", pl.col("pressured").mean())],
}
# rate stats (derived) and which direction is good — percentiles are within role
RATES = {
    "route": [("catch_rate", lambda r: r["catches"] / max(r["targets"], 1), True),
              ("tgt_share", lambda r: r["targets"] / r["routes"], True),
              ("ypr", lambda r: r["yds"] / r["routes"], True)],
    "rush": [("pressure_rate", lambda r: r["pressures"] / r["rushes"], True),
             ("sack_rate", lambda r: r["sacks"] / r["rushes"], True),
             ("hit_rate", lambda r: r["hits"] / r["rushes"], True)],
    "block": [("allowed_rate", lambda r: r["allowed"] / r["blocks"], False),
              ("sack_allowed_rate", lambda r: r["sacks_allowed"] / r["blocks"], False),
              ("beaten_rate", lambda r: r["beaten"] / r["blocks"], False)],
    "coverage": [("comp_allowed", lambda r: r["comp"] / max(r["targeted"], 1), False),
                 ("ypc_allowed", lambda r: r["yds"] / r["snaps"], False),
                 ("int_rate", lambda r: r["ints"] / r["snaps"], True)],
    "qb": [("comp_pct", lambda r: r["comp"] / max(r["att"], 1), True),
           ("sack_rate", lambda r: r["sacks"] / r["dropbacks"], False),
           ("ypa", lambda r: r["yds"] / max(r["att"], 1), True),
           ("int_rate", lambda r: r["ints"] / max(r["att"], 1), False)],
}
PCT = {"route": [("sep", True), ("tgt_share", True), ("yds", True), ("top", True), ("catch_rate", True), ("ypr", True)],
       "rush": [("pressure_rate", True), ("pressures", True), ("sacks", True), ("top", True), ("sack_rate", True), ("hit_rate", True)],
       "block": [("allowed_rate", False), ("beaten", False), ("sack_allowed_rate", False), ("beaten_rate", False), ("blocks", True)],
       "coverage": [("near", False), ("comp_allowed", False), ("ints", True), ("top", True), ("ypc_allowed", False), ("snaps", True)],
       "qb": [("comp_pct", True), ("yds", True), ("sack_rate", False), ("tt", False), ("ypa", True), ("int_rate", False)]}


EXTRA = ["sep", "target", "catch", "sack", "hit", "hurry", "allowed", "tt", "int", "targeted", "comp", "near",
         "decoy_n", "decoy_score", "decoy_drag", "decoy_raw", "decoy_tm", "decoy_ids", "decoy_radius"]
DECOY_REPS = os.path.join(HERE, "..", "out", "decoy_reps.parquet")   # from decoy.py


def play_score(role: str) -> pl.Expr:
    """How good a play was for this player in this job — used to pick replays per game."""
    c = lambda n: pl.col(n).cast(pl.Float64).fill_null(0)
    return {
        # route runners: their best DECOY plays — the brief's score (defenders x yards from target)
        "route": c("decoy_score") * 10 + c("decoy_n"),
        "rush": c("sack") * 1000 + c("hit") * 500 + c("hurry") * 300 + c("top"),
        "block": (1 - c("allowed")) * 1000 + c("tt") * 10,
        "coverage": c("int") * 2000 + (c("targeted") * (1 - c("comp"))) * 1000 + (10 - c("near")),
        "qb": c("comp") * 1000 + c("yds") * 10 - c("sack") * 500 - c("int") * 800,
    }[role]


def main():
    d = BDB()
    pff = d.pff().select(["gameId", "playId", "nflId", "pff_role", "pff_hit", "pff_hurry", "pff_sack",
                          "pff_beatenByDefender", "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"])
    P = d.plays()
    plays = {(r["gameId"], r["playId"]): r for r in P.iter_rows(named=True)}
    tg = targets(d)
    tgt = dict(zip(zip(tg["gameId"], tg["playId"]), tg["targetId"]))

    rows, teams, tracks = [], defaultdict(lambda: defaultdict(int)), {}
    gids = d.game_ids()
    for k, gid in enumerate(gids):
        r, t, tr = play_rows(d, gid, pff, plays, tgt)
        rows += r
        for n, c in t.items():
            for tj, v in c.items():
                teams[n][tj] += v
        tracks[gid] = tr.select(["playId", "frameId", "nflId", "x", "y", "snap", "thr", "key", "pff_role"])
        if k % 20 == 0:
            print(f"  game {k + 1}/{len(gids)}  rows={len(rows):,}", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    dr = (pl.read_parquet(DECOY_REPS)
            .select(["gameId", "playId", "nflId", "attached", "score_adj", "score_raw", "score", "teammates",
                     "attachedIds", "bait_radius"])
            .rename({"attached": "decoy_n", "score_adj": "decoy_score", "score_raw": "decoy_raw",
                     "score": "decoy_drag", "teammates": "decoy_tm", "attachedIds": "decoy_ids",
                     "bait_radius": "decoy_radius"}))
    df = df.join(dr, on=["gameId", "playId", "nflId"], how="left")
    print(f"player-plays: {df.height:,}")

    games = d.games().select(["gameId", "week", "gameDate", "homeTeamAbbr", "visitorTeamAbbr"])
    names = d.players().select(["nflId", "displayName", "officialPosition"])
    bio = {r["nflId"]: r for r in d.players().iter_rows(named=True)}
    tm_count = defaultdict(lambda: defaultdict(int))
    jersey = {}
    for n, c in teams.items():
        for (tm, j), v in c.items():
            tm_count[n][tm] += v
        jersey[n] = max(c, key=c.get)[1]
    team_of = {n: max(c, key=c.get) for n, c in tm_count.items()}

    # role summaries + percentiles
    summ = {}
    for role, agg in AGG.items():
        s = df.filter(pl.col("role") == role).group_by("nflId").agg([e.alias(a) for a, e in agg])
        recs = s.to_dicts()
        for rec in recs:
            for name, fn, _ in RATES[role]:
                rec[name] = fn(rec)
        q = [r for r in recs if r[agg[0][0]] >= QUALIFY]
        for stat, hib in PCT[role]:
            vals = np.array([r[stat] for r in q if r[stat] is not None], dtype=float)
            for rec in recs:
                v = rec[stat]
                if v is None or not len(vals) or rec[agg[0][0]] < QUALIFY:
                    rec[f"p_{stat}"] = None
                    continue
                pr = (vals < v).mean() + 0.5 * (vals == v).mean()
                rec[f"p_{stat}"] = round(100 * (pr if hib else 1 - pr))
        summ[role] = {r["nflId"]: r for r in recs}
        print(f"  {role}: {len(recs)} players, {len(q)} qualified")

    # game log
    gl = (df.join(games, on="gameId")
            .group_by(["nflId", "gameId", "week", "gameDate", "homeTeamAbbr", "visitorTeamAbbr", "role"])
            .agg(pl.len().alias("n"),
                 pl.col("target").sum().alias("targets") if "target" in df.columns else pl.lit(0),
                 pl.col("catch").sum().alias("catches"),
                 pl.col("yds").sum().alias("yds"),
                 pl.col("pressure").sum().alias("pressures"),
                 pl.col("sack").sum().alias("sacks"),
                 pl.col("allowed").sum().alias("allowed"),
                 pl.col("targeted").sum().alias("targeted"),
                 pl.col("comp").sum().alias("comp"),
                 pl.col("att").sum().alias("att"),
                 pl.col("int").sum().alias("ints"))
            .sort(["nflId", "week"]))
    log = defaultdict(list)
    for r in gl.iter_rows(named=True):
        log[r["nflId"]].append(r)

    # showcase plays per player (primary role)
    snaps_by_role = df.group_by(["nflId", "role"]).len()
    primary = (snaps_by_role.sort("len", descending=True).group_by("nflId").agg(pl.col("role").first()))
    primary = dict(primary.iter_rows())
    picks = []
    for role in ["route"]:      # the app is about decoys: replays are decoy plays, for anyone who ran routes
        # a REAL decoy play: not the target, 2+ coverage defenders within 5 yd at the throw, and those
        # defenders were pulled away from the target since the snap (drag > 0)
        sub_ = (df.filter((pl.col("role") == role) & (pl.col("decoy_n").fill_null(0) >= REAL_DECOY_MIN)
                          & (pl.col("decoy_drag").fill_null(0) > 0))
                  .with_columns(play_score(role).alias("score")))
        per_game = (sub_.sort("score", descending=True)
                        .group_by(["nflId", "gameId"], maintain_order=True).head(PER_GAME))
        best = (sub_.sort("score", descending=True).group_by("nflId", maintain_order=True).head(SEASON_BEST)
                    .select(["nflId", "gameId", "playId"]).with_columns(pl.lit(True).alias("best")))
        cols = ["nflId", "gameId", "playId", "role", "score"] + [c for c in EXTRA if c in sub_.columns]
        per_game = (pl.concat([per_game.select(cols),
                               sub_.join(best, on=["nflId", "gameId", "playId"]).select(cols)],
                              how="diagonal_relaxed").unique(["nflId", "gameId", "playId"])
                      .join(best, on=["nflId", "gameId", "playId"], how="left"))
        picks.append(per_game)
    picks = pl.concat(picks, how="diagonal_relaxed").with_columns(pl.col("best").fill_null(False))
    print(f"replay plays: {picks.height:,}")

    nm = dict(zip(names["nflId"], names["displayName"]))
    pos = dict(zip(names["nflId"], names["officialPosition"]))
    film = defaultdict(lambda: {"frames": {}, "plays": defaultdict(list)})
    for r_ in picks.sort("score", descending=True).iter_rows(named=True):
        n, gid, pid, role, best = r_["nflId"], r_["gameId"], r_["playId"], r_["role"], r_["best"]
        tm = team_of.get(n, "UNK")
        k = f"{gid}-{pid}"
        if k not in film[tm]["frames"]:
            film[tm]["frames"][k] = frames_for(tracks[gid], pid)
        p = plays[(gid, pid)]
        film[tm]["plays"][n].append(dict(
            key=k, gameId=gid, role=role, best=best, week=p["week"], q=p["quarter"], clock=p["gameClock"],
            down=p["down"], togo=p["yardsToGo"], desc=p["playDescription"], result=p["passResult"],
            yds=p["playResult"], coverage=p["pff_passCoverage"], target=tgt.get((gid, pid)),
            off=p["possessionTeam"], dff=p["defensiveTeam"],
            me={c: (round(r_[c], 2) if isinstance(r_[c], float) else r_[c]) for c in EXTRA
                if r_.get(c) not in (None, False, 0)}))

    os.makedirs(OUT, exist_ok=True)
    players = []
    for n in sorted(set(df["nflId"].to_list())):
        roles = {}
        for role in AGG:
            s = summ[role].get(n)
            if s:
                roles[role] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "nflId"}
        tm = team_of.get(n, "UNK")
        g = []
        for r in log.get(n, []):
            opp = r["visitorTeamAbbr"] if r["homeTeamAbbr"] == tm else r["homeTeamAbbr"]
            g.append({k: v for k, v in dict(gameId=r["gameId"], week=r["week"], date=r["gameDate"], opp=opp,
                                            home=r["homeTeamAbbr"] == tm, role=r["role"], n=r["n"],
                                            targets=r["targets"], catches=r["catches"], yds=r["yds"],
                                            pressures=r["pressures"], sacks=r["sacks"], allowed=r["allowed"],
                                            targeted=r["targeted"], comp=r["comp"], att=r["att"],
                                            ints=r["ints"]).items() if v not in (None, 0, False) or k in ("week", "n", "gameId")})
        b = bio.get(n, {})
        age = None
        if b.get("birthDate"):
            try:
                y, m_, d_ = (int(x) for x in str(b["birthDate"])[:10].split("-"))
                age = 2021 - y - ((m_, d_) > (9, 9))
            except ValueError:
                age = None
        players.append(dict(id=n, name=nm.get(n, str(n)), pos=pos.get(n), team=tm, primary=primary.get(n),
                            num=jersey.get(n), ht=b.get("height"), wt=b.get("weight"), college=html.unescape(b["collegeName"]) if b.get("collegeName") else None, age=age,
                            roles=roles, games=g))
    meta = dict(season=2021, weeks="1–8", games=len(gids), plays=int(df.select(pl.struct("gameId", "playId").n_unique()).item()),
                players=len(players), qualify=QUALIFY)
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, players=players, names={int(k): v for k, v in nm.items()}), f,
                  separators=(",", ":"), ensure_ascii=False)
    for tm, fl in film.items():
        with open(os.path.join(OUT, f"film_{tm}.json"), "w", encoding="utf-8") as f:
            json.dump({"frames": fl["frames"], "plays": {str(k): v for k, v in fl["plays"].items()}}, f,
                      separators=(",", ":"), ensure_ascii=False)
    tot = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT))
    print(f"wrote {OUT}: {len(players)} players, {len(film)} film files, {tot / 1e6:.1f} MB total, "
          f"index {os.path.getsize(os.path.join(OUT, 'index.json')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
