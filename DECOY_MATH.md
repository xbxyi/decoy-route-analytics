# Decoy Route Analysis — the maths

How every number on the site is produced. Code: `src/decoy.py` (scoring), `src/add_decoy_stats.py`
(player totals), `src/build_profiles.py` (replays), `src/fetch_madden.py` (card ratings).

## Data

NFL Big Data Bowl 2023 tracking: 2021 season, weeks 1–8, 122 games, 8,557 pass plays. Every player and
the ball are recorded 10 times a second (x, y in yards). PFF charts each player's job on each play
(`Pass Route`, `Pass Block`, `Coverage`, `Pass Rush`, `Pass`).

Before any geometry, every play is flipped so the offense always moves toward increasing x. Without
this, half the plays point the other way and every distance-to-target would be wrong.

## 1. Who got the ball

The target is read from the official play-by-play text — `pass short left to M.Evans`,
`pass incomplete deep right to A.St. Brown`, `intended for K.Hinton INTERCEPTED` — and matched to a
player PFF charted as running a route on that play (last name, then first initial if two share it).

7,285 of 7,565 thrown passes get a target. The rest are throwaways and spikes with no named receiver,
and are left out.

## 2. Decoys

On a play with a target, **every other player who ran a route is a decoy** for that play.

Everything below is measured at **the throw**: the first `pass_forward` / `autoevent_passforward`
frame. That is the moment the quarterback has chosen, so it is when pulling coverage away matters.

## 3. Baited defenders

A coverage defender `c` counts as **baited** by decoy `D` only if all four hold:

| # | Rule | Why |
|---|------|-----|
| 1 | `dist(c, D) < 5 yd` at the throw | the decoy has to actually be holding him |
| 2 | Over the 1.5 s before the throw (15 frames), `D` is the route runner `c` stayed closest to on average | he is responsible for the decoy, not just passing by |
| 3 | At the throw, `D` is the offensive player nearest to `c` — any teammate except the QB, blockers included | a defender standing on a teammate is never credited to the decoy |
| 4 | `dist(c, target) ≥ 5 yd` at the throw | if he is on the target he was not pulled away |

`n` = number of baited defenders.

## 4. Decoy score (the brief's formula)

The brief: *count the defenders near a non-targeted receiver and multiply that count by the distance
they were pulled away from the main target.*

```
raw   = n × (average distance from the baited defenders to the target)
      = Σ  dist(c, target)          over baited defenders c        (yards)

teammates = offensive players (not the QB, not D) within 5 yd of D at the throw

score = raw / (1 + teammates)
```

The teammate divisor is the "invisible zone": if teammates stand inside the decoy's 5-yard zone, the
defenders there may be there for them, so the decoy gets a share of the credit. One teammate halves
the score, two cut it to a third.

Worked example — Chris Conley, week 3 vs Carolina, Q3 13:39: three baited defenders, on average
18.5 yd from the target, no teammates in his zone → `3 × 18.5 = 55.6`, ÷ 1 → **55.6**.

## 5. Drag (were they really pulled?)

For each baited defender:

```
drag(c) = max(0, dist(c, target) at the throw − dist(c, target) at the snap)
```

Yards he moved *away* from the target between the snap and the throw. Summed over baited defenders.

## 6. Real decoy play

A play counts as a **real decoy play** for a decoy when:

```
n ≥ 2     (the brief: tricking 2 or more defenders)
drag > 0  (those defenders ended up farther from the target than at the snap)
```

This rules out zone defenders who were simply standing in that area from the start.

## 7. Player numbers

Over every decoy route a player ran:

| Shown as | Formula |
|---|---|
| avg score | mean of `score` over his decoy routes |
| best score | max of `score` |
| defenders per route | mean of `n` |
| real decoy plays | count of real decoy plays |
| Drew 2+ defenders | share of decoy routes with `n ≥ 2` |

**Percentiles** compare a player with every decoy who ran at least 40 decoy routes:
`percentile = 100 × (share of that group below him + ½ × share tied with him)`.

**The roster** lists players with at least 20 decoy routes and at least one real decoy play
(274 players). Each team's list is sorted by avg score.

## 8. Replays

For each match the dropdown holds the player's single best **real decoy play** of that game, ranked
by `score`. "Season best" is his top three. The replay runs from the first tracked frame (about 0.5 s
before the snap) and **ends on the throw frame** — the frame every rule above is measured on — so
the highlighted defenders, the 5-yard circle and the tethers are exactly what was scored.
1× playback is real time (10 frames a second); the speed readout is computed from his positions in
neighbouring frames.

Check run on every replay (1,055 plays, 2,157 highlighted defenders): at the throw frame none is
nearer a teammate than the decoy and none is outside 5 yards (one reads 5.06 yd only because replay
positions are rounded to 0.1 yd).

## 9. Card ratings

The overall and the six attributes are EA's official **Madden NFL 22** ratings (week 8 roster update,
launch ratings if he was not on the week 8 roster), matched to our players by name + birth date, then
name + team, then last name + birth date. 1,598 of 1,679 players match; the rest show their real
counting stats instead.

## Choices and limits

- **5 yards** is a modelling choice. A wider, motion-based version (up to 15 yd for defenders
  chasing the decoy) caught more defenders but occasionally credited a far-away one, so the site
  uses the stricter 5-yard version.
- **Tracking stops 0.5 s after the throw** in this dataset, so the catch itself is never shown and
  nothing is extrapolated.
- **Weeks 1–8 only, pass plays only.** Totals are smaller than full-season official numbers.
- The raw formula rewards depth: a deep target is far from everyone, so `raw` rises with throw depth.
  Compare scores between decoys, not between a 3-yard and a 40-yard throw.
