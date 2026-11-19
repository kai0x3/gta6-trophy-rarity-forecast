# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kai0x3/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kai0x3/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's Exophase earn rate (5%, 25%, 50%, 75%, 95% quantiles) at the case's target — week one after launch. A rate is earners / players tracked, and both sides move: Exophase refreshes its page on its own schedule, so the history is a sequence of frozen page states and the answer is the page as refreshed by the target. The program therefore forecasts the two sides separately and divides.

- **Players N(t):** a power-law fit on the last 24 h of page states (weight 0.6) plus a logistic fit (weight 0.4), combined in log space; a launch-jump prior when the history is flat (the page can sit unchanged for hours and then jump several-fold at one refresh, as it did on launch night).
- **Earners E_i(t):** three variants per trophy, all shrunk toward tier-class priors (story chapters earn fast, grind/completion slow): (A) a direct lognormal-CDF saturation fit; (B) a cohort fit that convolves the fitted join curve with the same CDF, separating join-dilution from slow earning; (C) the log-time trend of the logit rate over the last 24 h. The median is the logit-space average of the three, weighted by each variant's in-sample fit quality, with the trend given a 2× prior weight.
- **Spread:** honest quantiles on the logit scale, combining fit residuals, a horizon term, inter-variant disagreement, and a wide flat-history penalty.

Standard library only, deterministic (byte-identical outputs across runs), no network. Validated on the two practice cases and on a synthetic final-like suite (a deliberately different data-generating process: Weibull time-to-earn, logistic player growth, Exophase-style refresh jumps): mean pinball loss 0.16–0.19 across 10 scenarios, beating both house baselines (persistence 0.60–0.85, curve 0.23–0.28) in every scenario.

## Run it

```sh
python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Each output is `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}` for all 55 trophies.

---

<sub>Entered on [make.fun](https://make.fun/?ref=90bhv04h&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
