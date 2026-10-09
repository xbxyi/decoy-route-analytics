"""Big Data Bowl 2023 (pass rush / pass protection) — data layer.

122 games, Weeks 1-8 of the 2021 season. One tracking file per game.

    from bdb import BDB
    d = BDB()                      # finds data/ automatically
    plays   = d.plays()            # plays joined to games
    tr      = d.tracking(gameId)   # one game of tracking
    snap    = d.at_snap(gameId)    # one row per player at ball_snap
    pff     = d.pff()              # PFF roles / pressure labels

Uses polars (fast on 10Hz tracking); .to_pandas() on anything if you prefer.
"""
from __future__ import annotations

import functools
import os
import glob
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data():
    for c in (os.path.join(HERE, "..", "data", "data"),
              os.path.join(HERE, "..", "data"),
              os.path.join(HERE, "data")):
        c = os.path.abspath(c)
        if os.path.isfile(os.path.join(c, "plays.csv")):
            return c
    raise FileNotFoundError("couldn't find the data dir (expected plays.csv inside it)")


class BDB:
    def __init__(self, data_dir: str | None = None):
        self.dir = os.path.abspath(data_dir) if data_dir else _find_data()
        self.tracking_dir = os.path.join(self.dir, "tracking")

    # ── raw tables ─────────────────────────────────────────────────────────
    @functools.lru_cache(maxsize=1)
    def games(self) -> pl.DataFrame:
        return pl.read_csv(os.path.join(self.dir, "games.csv"))

    @functools.lru_cache(maxsize=1)
    def players(self) -> pl.DataFrame:
        return pl.read_csv(os.path.join(self.dir, "players.csv"),
                           infer_schema_length=10000, null_values=["NA", ""])

    @functools.lru_cache(maxsize=1)
    def pff(self) -> pl.DataFrame:
        return pl.read_csv(os.path.join(self.dir, "pffScoutingData.csv"),
                           infer_schema_length=10000, null_values=["NA", ""])

    @functools.lru_cache(maxsize=1)
    def plays(self) -> pl.DataFrame:
        """plays.csv joined to games.csv (adds season/week/teams)."""
        p = pl.read_csv(os.path.join(self.dir, "plays.csv"), infer_schema_length=10000,
                        null_values=["NA", ""])
        return p.join(self.games(), on="gameId", how="left")

    # ── tracking ───────────────────────────────────────────────────────────
    def game_ids(self) -> list[int]:
        out = []
        for f in sorted(glob.glob(os.path.join(self.tracking_dir, "tracking_*.csv"))):
            try:
                out.append(int(os.path.basename(f)[9:-4]))
            except ValueError:
                pass
        return out

    def tracking(self, game_id: int) -> pl.DataFrame:
        """One game of tracking.

        NB: the CSVs use the literal "NA" (ball rows have no nflId/o/dir), which makes
        polars type those columns as String. Declaring them null and casting is the
        difference between numeric features and a dtype error later.
        """
        f = os.path.join(self.tracking_dir, f"tracking_{game_id}.csv")
        tr = pl.read_csv(f, infer_schema_length=10000, null_values=["NA", ""])
        casts = {"nflId": pl.Int64, "jerseyNumber": pl.Int64,
                 "o": pl.Float64, "dir": pl.Float64,
                 "x": pl.Float64, "y": pl.Float64, "s": pl.Float64,
                 "a": pl.Float64, "dis": pl.Float64}
        return tr.with_columns([pl.col(c).cast(t, strict=False)
                                for c, t in casts.items() if c in tr.columns])

    def tracking_many(self, game_ids=None, week=None) -> pl.DataFrame:
        """Several games at once. Pass week=1 to grab a whole week."""
        if week is not None:
            g = self.games().filter(pl.col("week") == week)["gameId"].to_list()
            game_ids = g
        game_ids = game_ids or self.game_ids()
        return pl.concat([self.tracking(g) for g in game_ids], how="vertical_relaxed")

    # ── normalisation: make every play run left -> right ───────────────────
    @staticmethod
    def standardize(tr: pl.DataFrame) -> pl.DataFrame:
        """Flip left-moving plays so x always increases toward the offense's target.
        Without this, directional features (closing on the QB, depth of rush) are
        meaningless because half the plays point the other way."""
        left = pl.col("playDirection") == "left"
        return tr.with_columns([
            pl.when(left).then(120 - pl.col("x")).otherwise(pl.col("x")).alias("x"),
            pl.when(left).then(53.3 - pl.col("y")).otherwise(pl.col("y")).alias("y"),
            pl.when(left).then((pl.col("dir") + 180) % 360).otherwise(pl.col("dir")).alias("dir"),
            pl.when(left).then((pl.col("o") + 180) % 360).otherwise(pl.col("o")).alias("o"),
        ])

    # ── convenience slices ─────────────────────────────────────────────────
    def events(self, game_id: int) -> pl.DataFrame:
        """frameId of each tagged event per play (snap, pass_forward, sack...)."""
        tr = self.tracking(game_id)
        return (tr.filter(pl.col("event").is_not_null())
                  .select(["gameId", "playId", "frameId", "event"])
                  .unique()
                  .sort(["playId", "frameId"]))

    def snap_frames(self, game_id: int) -> pl.DataFrame:
        """One row per play: the frameId of ball_snap."""
        return (self.events(game_id)
                .filter(pl.col("event") == "ball_snap")
                .group_by(["gameId", "playId"])
                .agg(pl.col("frameId").min().alias("snapFrame")))

    def at_snap(self, game_id: int) -> pl.DataFrame:
        """Every player's position/orientation at the snap, + PFF role."""
        tr = self.standardize(self.tracking(game_id))
        snaps = self.snap_frames(game_id)
        out = (tr.join(snaps, on=["gameId", "playId"], how="inner")
                 .filter(pl.col("frameId") == pl.col("snapFrame")))
        return out.join(self.pff(), on=["gameId", "playId", "nflId"], how="left")

    def play(self, game_id: int, play_id: int, standardize=True) -> pl.DataFrame:
        tr = self.tracking(game_id).filter(pl.col("playId") == play_id)
        return self.standardize(tr) if standardize else tr

    def describe_play(self, game_id: int, play_id: int) -> str:
        r = self.plays().filter((pl.col("gameId") == game_id) & (pl.col("playId") == play_id))
        if r.is_empty():
            return "(play not found)"
        r = r.row(0, named=True)
        return (f"Q{r['quarter']} {r['gameClock']}  {r['down']}&{r['yardsToGo']}  "
                f"{r['possessionTeam']} vs {r['defensiveTeam']}\n"
                f"  {r['playDescription']}\n"
                f"  coverage={r.get('pff_passCoverage')}  "
                f"dropback={r.get('dropBackType')}  result={r.get('passResult')}")


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")   # polars draws unicode tables
    except Exception:
        pass
    d = BDB()
    print("data dir :", d.dir)
    print("games    :", d.games().height)
    print("plays    :", d.plays().height)
    print("players  :", d.players().height)
    print("pff rows :", d.pff().height)
    print("tracking :", len(d.game_ids()), "game files")
    g = d.game_ids()[0]
    tr = d.tracking(g)
    print(f"\ntracking_{g}: {tr.height:,} rows, {tr.width} cols")
    print("columns:", tr.columns)
    print("\nevents seen:", d.events(g)["event"].unique().to_list())
