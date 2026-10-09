# awsnfl — NFL Big Data Bowl hackathon

Two web apps built from the 2023 Big Data Bowl tracking data (2021 season, weeks 1–8).

**Player Film Room** (`app_profiles/`) — every player: role stats, percentiles vs players in the same job,
game log, FIFA-style card with headshot and pack-opening reveal, overhead replays of their best plays.

**Decoy Efficiency Index** (`app/`) — which route runners pull the most coverage away from the targeted receiver.

## Build

Put the dataset at `data/` (folder containing `plays.csv`, `players.csv`, `pffScoutingData.csv`, `tracking/`).

```
cd src
python decoy.py && python build_decoy_app.py      # -> app/decoy.json
python build_profiles.py                          # -> app_profiles/index.json, film_<TEAM>.json
python fetch_headshots.py                         # -> app_profiles/faces_<TEAM>.json
python fetch_madden.py                            # -> app_profiles/madden.json (EA Madden NFL 22 ratings)
```

## Run

```
cd app_profiles && python -m http.server 8000     # or cd app
```
