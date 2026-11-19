# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lisanye9527/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lisanye9527/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on
Exophase. Given every read of the game's trophy page up to a checkpoint, it
writes, for each of the 55 trophies, five quantiles (5%, 25%, 50%, 75%, 95%)
of its earn rate one week after launch — the percent of tracked players who
will have earned it — scored by pinball loss on the logit scale.

Method: two views per trophy, combined on the logit scale. A trend view fits
the rate's logit against log time since release with a damped, saturating
extrapolation. A decomposition view forecasts earners and tracked players
separately as saturating growth curves — earners can only grow, while a
fast-growing player base dilutes every rate — with earners capped by the
trophy's eventual share inferred from its history. The spread is assembled
from the fit residuals, a horizon random-walk term, a small-count term and a
staleness term for histories whose page never refreshed. Pure Python 3
standard library, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=b79e8z1j&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
