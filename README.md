# Off-Ball Receiver Impact & Decoy Route Analytics

Elite wide receivers change games even when they never touch the ball, yet traditional stat sheets treat their off-ball contributions as invisible. We built a tracking metric that measures how effectively "decoy" receivers draw coverage defenders away from primary targets to open up high-probability passing lanes.

By evaluating defender spacing from the moment the ball is snapped until it is thrown, our model calculates the exact amount of open grass created for teammates. This proves that pulling defenders away on deep routes directly drives higher offensive success rates for the entire team. Our project pairs a Python data pipeline with an interactive 2D web application (`app/index.html` and `app_profiles/index.html`) so coaches and fans can replay animated plays and explore league-wide player leaderboards.

## Deploy (Vercel)

```
cd src && python build_site.py          # -> site/ (page + data, ~25 MB, not committed)
npx vercel deploy ../site --prod        # after `npx vercel login`; or --temporary without an account
```

How the numbers are computed: [DECOY_MATH.md](DECOY_MATH.md).
