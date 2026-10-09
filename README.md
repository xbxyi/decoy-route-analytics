# Decoy Route Analytics & Off-Ball Receiver Impact

Elite wide receivers change games even when they never touch the ball, yet traditional stat sheets treat their off-ball contributions as invisible. We built a tracking metric that measures how effectively "decoy" receivers draw coverage defenders away from primary targets to open up high-probability passing lanes.

By evaluating defender spacing from the moment the ball is snapped until it is thrown, our model calculates the exact amount of open grass created for teammates. This proves that pulling defenders away on deep routes directly drives higher offensive success rates for the entire team. Our project pairs a Python data pipeline with an interactive 2D web application (`app/index.html` and `app_profiles/index.html`) so coaches and fans can replay animated plays and explore league-wide player leaderboards.

**Live demo:** https://temporary-spry-sable-dmgxyf7.vercel.app/#p41282

![Davante Adams, week 2 vs DET, Q4 3:21 — at the throw he has lured two Lions defenders (A.J. Parker, Tracy Walker) inside his 5-yard zone while the ball goes deep to Marquez Valdes-Scantling](docs/media/decoy_play.gif)

*Davante Adams, Week 2 vs Detroit, Q4 3:21 — real tracking, snap to throw. Orange: Adams (the decoy). Yellow: the target. Blue: the defenders he lured, tethered to him at the throw. Decoy score 28.7.*

![Coach view for Davante Adams: what his decoy work buys the target, defenders lured by coverage and by route depth, and decoy score by week](docs/media/coach_view.png)

*Coach view for the same player: target separation, completion and yards when he lures 2+ defenders vs 0–1; defenders lured per route against man and zone and by route depth, each against the league average; decoy score week by week.*

## How it works

For every pass with a named target (7,285 plays, 2021 weeks 1–8), every other route runner is a **decoy**. At the moment of the throw, a coverage defender counts as **lured** when he is within 5 yards of the decoy, the decoy is the route runner he stayed closest to over the previous 1.5 s, the decoy is the offensive player nearest to him, and he is not on the target.

```
decoy score = Σ distance(lured defender, target)  ÷  (1 + teammates inside the decoy's 5-yard zone)
```

A **real decoy play** needs 2+ lured defenders who ended up farther from the target than at the snap. The full derivation, worked example and limits are in **[DECOY_MATH.md](DECOY_MATH.md)**.

![The site: roster and team picker, Madden 22 card front and back, decoy percentiles, match picker and the replay](docs/media/site.png)

## Data

NFL Big Data Bowl 2023 tracking (10 Hz, all 22 players + ball), PFF scouting roles, official play-by-play. Card ratings: EA Madden NFL 22 (week 8 update). Headshots: NFL.com 2021 rosters via nflverse. Place the dataset at `data/` (the folder with `plays.csv`, `players.csv`, `pffScoutingData.csv`, `tracking/`).

## Build and run

```
cd src
python decoy.py                 # lured defenders and decoy score for every route  -> out/
python build_profiles.py        # player profiles + replay frames                   -> app_profiles/
python add_decoy_stats.py       # decoy totals, percentiles, coach view             -> app_profiles/index.json
python fetch_madden.py          # Madden 22 ratings                                 -> app_profiles/madden.json
python fetch_headshots.py       # 2021 headshots                                    -> app_profiles/faces_<TEAM>.json
python make_readme_media.py     # the GIF above                                     -> docs/media/

cd ../app_profiles && python -m http.server 8000      # open http://localhost:8000
```

Deploy: `python src/build_site.py` assembles `site/` (page + data); `npx vercel deploy site --prod`.
