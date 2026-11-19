"""gta6-trophy-rarity-forecast-01 entry: forecast every GTA VI PS5 trophy's
Exophase earn rate at the case's target.

Model (standard library only, deterministic, no network):

  An earn rate is earners / players_tracked, and both sides move.  Exophase
  refreshes its page on its own schedule, so the history is a sequence of
  frozen page states; the answer is the page as refreshed by the target.
  The program therefore forecasts the two sides separately and divides:

  * players N(t): a power-law fit on the last 24 h of states (weight 0.6)
    and a logistic fit (weight 0.4), combined in log space, with a
    launch-jump prior when the history is flat (the page can sit
    unchanged for hours and then jump several-fold at one refresh, as
    it did on launch night).
  * earners E_i(t) per trophy, in three variants, all shrunk toward
    tier-class priors (story chapters earn fast, grind/completion slow):
      A. direct saturation: E_i(t) = A_i * Phi((ln t - ln tau_i) / s_class),
         fitted per trophy by grid search on tau_i with A_i by least squares.
      B. cohort fit: E_i(t) = max(E_first, p_i * K(t)), where K(t) is the
         earners by t if everyone eventually earned, built by convolving
         the fitted join curve N'(u) with the same lognormal CDF.  Fitting
         (tau_i, p_i) against the kernel separates join-dilution from slow
         earning, which the direct fit confounds.
      C. the log-time trend of the logit rate over the last 24 h (the
         house curve baseline's shape), which extrapolates the observed
         trajectory directly.
    The forecast median is the logit-space average of the three variants,
    weighted by each variant's in-sample fit quality over the last 24 h
    of reads, with the trend variant given a 2x prior weight (it is the
    most accurate variant for fast-rising trophies, where the structural
    fits are identified only from early data).
  * spread: honest quantiles on the logit scale.  sigma combines the
    fit's residual scatter, a horizon term (uncertainty grows like a
    random walk), the weighted disagreement between the three variants,
    and a wide flat-history penalty when the history shows no movement
    at all (a refresh we have not seen can move rates a lot).

  Quantiles are the standard normal's at the case's levels, rounded to
  4 decimals half-up, enforced non-decreasing and inside [0, 100].
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch of Grand Theft Auto VI: the time origin for every time-to-earn fit.
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# --- small numeric helpers -------------------------------------------------

def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(dt):
    return (dt - LAUNCH).total_seconds() / 3600.0


def _lsq_line(xs, ys):
    """Least-squares y = a + b*x; returns (a, b, sse) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return (a, b, sse)


# --- trophy classes --------------------------------------------------------

# Class priors: (typical eventual share of tracked players who will earn the
# trophy, generous cap on that share, typical median time-to-earn in hours).
# These are design judgements about a big open-world launch, used only as
# shrinkage targets where the history does not identify a trophy; the data
# dominates wherever it moves.
CLASS_PRIORS = {
    "platinum":    (0.002, 0.05, 600.0),
    "completion":  (0.015, 0.15, 500.0),
    "story-final": (0.12, 0.40, 120.0),
    "story-mid":   (0.30, 0.70, 48.0),
    "story-early": (0.45, 0.80, 24.0),
    "grind":       (0.04, 0.30, 300.0),
    "activity":    (0.12, 0.50, 72.0),
}

_GRIND_HINTS = (
    "250 Stunts", "gold medal", "Drone Race at every location",
    "20 different races", "mission strands", "Collectible Card sets",
    "100 locations", "50 Challenges", "100 vehicles", "Bank Vaults",
    "laundering", "Spend $100,000", "garage properties",
    "Register 10 vehicles",
)


def classify(trophy):
    """Assign a trophy to an earning-speed class from its description."""
    desc = trophy.get("description", "")
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return "platinum"
    if "100% Completion" in desc:
        return "completion"
    if "Complete Chapter 1" in desc or "Complete Gone Fishing" in desc:
        return "story-early"
    if "Complete Chapter" in desc:
        return "story-mid"
    if "Complete the final mission" in desc:
        return "story-final"
    if desc.startswith("Complete "):
        return "story-mid"
    if any(hint in desc for hint in _GRIND_HINTS):
        return "grind"
    return "activity"


# --- page states -----------------------------------------------------------

def distinct_states(case):
    """Collapse consecutive identical reads into page states.

    Returns a list of (t_hours, players, rates) with t measured from launch,
    one entry per distinct page state, in time order.
    """
    states = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        slot = read.get("slot") or read.get("at")
        t = max(0.0, hours_since_launch(parse_time(slot)))
        states.append((t, int(read["players_tracked"]), list(rates)))
    return states


def earners_from(states, i):
    """(times, earners) for trophy i across states; skips null rates."""
    ts, es = [], []
    for t, n, rates in states:
        r = rates[i] if i < len(rates) else None
        if r is None:
            continue
        ts.append(t)
        es.append(round(r * n / 100.0))
    return ts, es


# --- players curve N(t) ----------------------------------------------------

def _fit_power_law(ts, ns):
    best = None
    for c in (2.0, 6.0, 12.0, 24.0, 48.0, 96.0):
        xs = [math.log(t + c) for t in ts]
        ys = [math.log(n) for n in ns]
        fit = _lsq_line(xs, ys)
        if fit is None:
            continue
        a, b, sse = fit
        if best is None or sse < best[0]:
            best = (sse, a, b, c)
    if best is None:
        return None
    _, a, b, c = best
    return lambda t: math.exp(a + b * math.log(max(t, 0.0) + c))


def _fit_logistic(ts, ns):
    n_last = ns[-1]
    best = None
    t_mid_choices = sorted({
        ts[len(ts) // 4], ts[len(ts) // 2], ts[(3 * len(ts)) // 4]
    }) if len(ts) >= 4 else [ts[len(ts) // 2]]
    for mult in (1.3, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0):
        ninf = mult * n_last
        for k in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3):
            for tm in t_mid_choices:
                def curve(t, ninf=ninf, k=k, tm=tm):
                    z = -k * (t - tm)
                    z = max(-60.0, min(60.0, z))
                    return ninf / (1.0 + math.exp(z))
                sse = 0.0
                for t, n in zip(ts, ns):
                    pred = max(curve(t), 1.0)
                    sse += (math.log(n) - math.log(pred)) ** 2
                if best is None or sse < best[0]:
                    best = (sse, ninf, k, tm)
    if best is None:
        return None
    _, ninf, k, tm = best
    # Two hill-climb passes: nudge each parameter, keep improvements.
    for _ in range(2):
        improved = False
        for dn, dk, dt in ((1.1, 1.0, 1.0), (0.9, 1.0, 1.0),
                           (1.0, 1.1, 1.0), (1.0, 0.9, 1.0),
                           (1.0, 1.0, 1.05), (1.0, 1.0, 0.95)):
            n2, k2, tm2 = ninf * dn, k * dk, tm * dt
            def curve(t, ninf=n2, k=k2, tm=tm2):
                z = max(-60.0, min(60.0, -k * (t - tm)))
                return ninf / (1.0 + math.exp(z))
            sse = 0.0
            for t, n in zip(ts, ns):
                pred = max(curve(t), 1.0)
                sse += (math.log(n) - math.log(pred)) ** 2
            if sse < best[0] - 1e-12:
                best = (sse, n2, k2, tm2)
                ninf, k, tm = n2, k2, tm2
                improved = True
        if not improved:
            break
    _, ninf, k, tm = best
    return lambda t: ninf / (1.0 + math.exp(max(-60.0, min(60.0, -k * (t - tm)))))


def fit_players(states, target_h):
    """Return (N_at_target, N_fn, flat) for the case's history."""
    ts = [s[0] for s in states]
    ns = [s[1] for s in states]
    n_first, n_last = ns[0], ns[-1]
    flat = len(states) < 3 or (max(ns) - min(ns)) < 0.02 * max(ns)
    if flat:
        # No movement to fit: the launch-night prior.  The page went from
        # 638 players at launch to 3,868 at its first refresh ~4.5 h later,
        # a 6x jump; model the early growth as N ~ N_first * (1 + t/2)^1.6.
        n_target = n_first * (1.0 + max(target_h, 0.0) / 2.0) ** 1.6
        n_target = min(max(n_target, n_last), 50.0 * n_last)
        return n_target, (lambda t: n_first * (1.0 + max(t, 0.0) / 2.0) ** 1.6), True
    # Fit the power law on the LAST 24 h of states, not the whole history:
    # the early explosive phase would otherwise dominate the exponent and
    # the extrapolation would overshoot a saturating player base.
    recent = [(t, n) for (t, n) in zip(ts, ns) if t >= ts[-1] - 24.0]
    if len(recent) >= 3:
        power = _fit_power_law([r[0] for r in recent], [r[1] for r in recent])
    else:
        power = _fit_power_law(ts, ns)
    logistic = _fit_logistic(ts, ns)
    curves, weights = [], []
    if power is not None:
        curves.append(power)
        weights.append(0.6)
    if logistic is not None:
        curves.append(logistic)
        weights.append(0.4)
    if not curves:
        return float(n_last), (lambda t: float(n_last)), flat
    n_target = math.exp(sum(w * math.log(max(c(target_h), 1.0))
                            for (w, c) in zip(weights, curves)))
    n_target = min(max(n_target, float(n_last)), 50.0 * float(n_last))
    def n_fn(t):
        vals = [c(t) for c in curves]
        return math.exp(sum(w * math.log(max(v, 1.0)) for (w, v) in zip(weights, vals)))
    return n_target, n_fn, flat


# --- earners curves E_i(t) -------------------------------------------------

# Lognormal-CDF shape widths tried per class (hours are lognormal: the shape
# is the spread of ln(time-to-earn)).
_SHAPES = (0.6, 0.9, 1.3, 1.9)


def _phi(t, mu, s):
    """Lognormal CDF at time t (hours) for ln-median mu and log-spread s."""
    tt = max(t, 1.0)
    return norm_cdf((math.log(tt) - mu) / s)


def _fit_one_trophy(ts, es, s):
    """Best (mu, A, sse) for E(t) = A * Phi((ln t - mu)/s) on log scale."""
    best = None
    mu = math.log(0.5)
    while mu <= math.log(2000.0) + 1e-9:
        phis = [_phi(t, mu, s) for t in ts]
        num = sum(e * p for e, p in zip(es, phis))
        den = sum(p * p for p in phis)
        a = num / den if den > 1e-12 else 0.0
        a = max(a, max(es) if es else 0.0)
        sse = 0.0
        for t, e in zip(ts, es):
            pred = a * _phi(t, mu, s)
            sse += (math.log(e + 0.5) - math.log(pred + 0.5)) ** 2
        if best is None or sse < best[2]:
            best = (mu, a, sse)
        mu += 0.15
    return best


def _cohort_kernel(n_fn, n_first, mu, s, times):
    """K(t) for each t in `times`: earners by t if everyone eventually
    earns, under the cohort model.  Players join over time following the
    fitted N curve; a cohort that joined at u earns with the lognormal
    CDF at age t-u; the launch cohort (N_first players present at t=0)
    is counted with the CDF at t.  Computed in one pass over the join
    grid so the cost is shared across trophies.
    """
    step = 0.5
    max_t = max(times) if times else 0.0
    # Join rate per half-hour bin, from the fitted players curve.
    bins = []
    u = 0.0
    while u < max_t:
        u1 = min(u + step, max_t)
        dn = max(0.0, n_fn(u1) - n_fn(u))
        if dn > 0:
            bins.append((0.5 * (u + u1), dn))
        u = u1
    out = []
    for t in times:
        total = n_first * _phi(t, mu, s)
        for (umid, dn) in bins:
            delta = t - umid
            if delta <= 0:
                continue
            total += dn * _phi(delta, mu, s)
        out.append(total)
    return out


_P_GRID = (0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0)


def _cohort_best(ts, es, e_first, kernels, s, mu_values):
    """Best (mu, p, sse) of the cohort model E(t) = max(E_first, p * K(t)),
    using the shared kernels K(t; mu, s) precomputed per grid mu.
    The direct saturation fit confounds join-dilution with slow earning
    (its mu comes out far too slow); the cohort fit separates them
    because the kernel already accounts for when players joined.
    """
    if len(ts) < 3:
        return None
    best = None
    for mu in mu_values:
        k_map = kernels[(s, mu)]
        for p in _P_GRID:
            sse = 0.0
            for t, e in zip(ts, es):
                pred = max(float(e_first), p * k_map[t])
                sse += (math.log(e + 0.5) - math.log(pred + 0.5)) ** 2
            if best is None or sse < best[2]:
                best = (mu, p, sse)
    return best


def fit_earners(case, states, n_target, n_fn, target_h):
    """Fit every trophy's earners curve.

    Two fits per trophy, both shrunk toward class priors:
      * direct: E(t) = A * Phi((ln t - mu)/s) — variant A.
      * cohort: E(t) = max(E_first, p * K(t)) with K the cohort kernel —
        variant B.  Only where the history moves; a flat trophy cannot
        identify it, so variant B then repeats variant A.
    Returns a dict per trophy with the fitted parameters, the per-trophy
    residual scatter, and a moving flag.
    """
    trophies = case["trophies"]
    n = len(trophies)
    classes = [classify(t) for t in trophies]
    n_first = float(states[0][1])
    state_times = sorted(set(s[0] for s in states))
    fit_times = sorted(set(state_times + [target_h]))
    # Shared cohort kernels K(t; mu, s) for every grid mu and shape.
    mu_values = []
    mu = math.log(0.5)
    while mu <= math.log(2000.0) + 1e-9:
        mu_values.append(mu)
        mu += 0.2
    kernels = {}
    for s in _SHAPES:
        for mu in mu_values:
            kernels[(s, mu)] = dict(zip(
                fit_times, _cohort_kernel(n_fn, n_first, mu, s, fit_times)))
    # Per-class shape choice: the width minimising total cohort-fit SSE
    # of the class's *moving* trophies only.  A flat trophy fits every
    # shape equally well (its fit is unidentified), so it must not vote;
    # a class with no moving trophy takes a moderate default width.
    class_shape = {}
    for cls in sorted(set(classes)):
        idxs = [i for i in range(n) if classes[i] == cls]
        best_s, best_tot = None, None
        for s in _SHAPES:
            tot, counted = 0.0, 0
            for i in idxs:
                ts, es = earners_from(states, i)
                if len(ts) < 3 or max(es) <= 0:
                    continue
                if (max(es) - min(es)) <= 0.10 * max(max(es), 1):
                    continue
                b = _cohort_best(ts, es, es[0], kernels, s, mu_values)
                if b is None:
                    continue
                tot += b[2]
                counted += 1
            if counted == 0:
                continue
            if best_tot is None or tot < best_tot:
                best_tot, best_s = tot, s
        class_shape[cls] = best_s if best_s is not None else 1.2
    # Per-trophy fits with shrinkage toward class priors.
    fitted = []
    for i in range(n):
        cls = classes[i]
        s = class_shape[cls]
        ts, es = earners_from(states, i)
        e_last = es[-1] if es else 0
        e_first = float(es[0]) if es else 0.0
        moving = len(ts) >= 2 and (max(es) - min(es)) > 0.10 * max(max(es), 1)
        n_states = len(ts)
        w = min(0.85, n_states / (n_states + 4.0)) if moving else 0.0
        typ_share, cap_share, typ_tau = CLASS_PRIORS[cls]
        # --- direct saturation fit (variant A).
        if len(ts) >= 2 and max(es) > 0:
            mu_fit, a_fit, _ = _fit_one_trophy(ts, es, s)
            mu_a = w * mu_fit + (1.0 - w) * math.log(typ_tau)
            a_prior = typ_share * n_target
            a_cap = cap_share * n_target
            log_a = w * math.log(max(a_fit, 1.0)) + (1.0 - w) * math.log(max(a_prior, 1.0))
            a = min(math.exp(log_a), a_cap)
            a = max(a, float(e_last))
        else:
            # No movement to fit: anchor the curve through the last
            # observed earners count, with the class prior's time scale.
            # (An unanchored prior A would ignore what the page shows.)
            # Anchor at t >= 2 h: the earners present at t ~= 0 are a
            # pre-launch population, not the CDF's early tail.
            mu_a = math.log(typ_tau)
            t_anchor = max(ts[-1] if ts else 0.0, 2.0)
            denom = _phi(t_anchor, mu_a, s)
            a_cap = cap_share * n_target
            a = float(e_last) / denom if denom > 1e-6 else typ_share * n_target
            a = min(max(a, float(e_last)), a_cap)
        # --- cohort fit (variant B), only where the history moves.
        mu_c, p_c, k_target = None, None, None
        if moving and len(ts) >= 3:
            b = _cohort_best(ts, es, e_first, kernels, s, mu_values)
            if b is not None:
                mu_fit_c, p_fit, _ = b
                mu_c = w * mu_fit_c + (1.0 - w) * math.log(typ_tau)
                log_p = w * math.log(max(p_fit, 1e-4)) + (1.0 - w) * math.log(typ_share)
                p_c = min(math.exp(log_p), cap_share)
                p_c = max(p_c, e_last / max(n_target, 1.0))
                k_map = dict(zip(
                    fit_times, _cohort_kernel(n_fn, n_first, mu_c, s, fit_times)))
                k_target = k_map[target_h]
        # Residual scatter of the best available fit, over states at
        # least 2 h after launch only: the pre-launch earners at t ~= 0
        # are a different (pre-release) population and no saturation
        # curve passes through them.  With too few such states the
        # scatter is unidentified: use a neutral 0.5.
        resid = []
        k_map = None
        if mu_c is not None:
            k_map = dict(zip(
                fit_times, _cohort_kernel(n_fn, n_first, mu_c, s, fit_times)))
        for t, e in zip(ts, es):
            if t < 2.0:
                continue
            if k_map is not None:
                pred_e = max(float(e_first), p_c * k_map[t])
            else:
                pred_e = a * _phi(t, mu_a, s)
            if pred_e > 0 and e >= 0:
                resid.append(abs(math.log((e + 0.5) / (pred_e + 0.5))))
        sigma_fit = (sum(r * r for r in resid) / len(resid)) ** 0.5 if len(resid) >= 3 else 0.5
        fitted.append({
            "class": cls, "shape": s, "mu_a": mu_a, "A": a,
            "mu_c": mu_c, "p_c": p_c, "k_target": k_target,
            "e_first": e_first, "e_last": float(e_last), "moving": moving,
            "sigma_fit": min(max(sigma_fit, 0.15), 0.9),
        })
    return fitted


def forecast_case(case):
    """Forecast one case; returns {"trophies": [{"id", "q": [...]}]}."""
    trophies = case["trophies"]
    states = distinct_states(case)
    if not states:
        raise ValueError("case has no history")
    target_dt = parse_time(case["target"])
    at_dt = parse_time(case["at"])
    target_h = max(hours_since_launch(target_dt), 0.0)
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    n_target, n_fn, flat = fit_players(states, target_h)
    n_first = float(states[0][1])
    fitted = fit_earners(case, states, n_target, n_fn, target_h)
    t_last = states[-1][0]
    state_times = [s[0] for s in states]
    rows = []
    for i, trophy in enumerate(trophies):
        fit = fitted[i]
        cls = fit["class"]
        s = fit["shape"]
        e_first = fit["e_first"]
        e_last = fit["e_last"]
        # --- variant A: direct saturation curve at the target.
        e_a = fit["A"] * _phi(target_h, fit["mu_a"], s)
        e_a = min(max(e_a, float(e_last)), 0.98 * n_target)
        r_a = 100.0 * e_a / n_target
        # --- variant B: cohort fit at the target.  With a flat history
        # there is no data to identify a separate cohort path, so it
        # repeats variant A rather than inventing disagreement.
        if flat or fit["mu_c"] is None:
            r_b = r_a
        else:
            e_b = max(float(e_first), fit["p_c"] * fit["k_target"])
            e_b = min(max(e_b, float(e_last)), 0.98 * n_target)
            r_b = 100.0 * e_b / n_target
        # --- variant C: log-time trend of the logit rate (last 24 h).
        ts_i, es_i = earners_from(states, i)
        r_c = None
        line = None
        aa = bb = 0.0
        pts = []
        for t, e in zip(ts_i, es_i):
            if t >= t_last - 24.0:
                n_at = next((nn for (tt, nn, _) in states if abs(tt - t) < 1e-9), None)
                if n_at:
                    pts.append((math.log1p(t), logit(100.0 * e / n_at)))
        if len(pts) >= 2:
            line = _lsq_line([p[0] for p in pts], [p[1] for p in pts])
            if line is not None:
                aa, bb, _ = line
                r_c = 100.0 * sigmoid(aa + bb * math.log1p(target_h))
        if r_c is None:
            n_last = float(states[-1][1])
            r_c = 100.0 * e_last / n_last if n_last else 0.0
        # --- in-sample fit quality of each variant over the window
        # (states at least 2 h after launch, within the last 24 h):
        # the variant whose trajectory best explains the recent reads
        # gets the most weight in the ensemble.
        w_lo = max(2.0, t_last - 24.0)
        win = [(t, n, rates[i]) for (t, n, rates) in states
               if t >= w_lo and i < len(rates) and rates[i] is not None]
        def rms_resid(pred_fn):
            ss, cnt = 0.0, 0
            for (t, n, r) in win:
                pr = pred_fn(t, n)
                if pr is None or pr <= 0:
                    continue
                ss += (logit(r) - logit(pr)) ** 2
                cnt += 1
            return (ss / cnt) ** 0.5 if cnt >= 2 else None
        sig_a = rms_resid(
            lambda t, n: 100.0 * fit["A"] * _phi(t, fit["mu_a"], s) / n)
        sig_b = None
        if fit["mu_c"] is not None and not flat:
            k_map = dict(zip(
                state_times,
                _cohort_kernel(n_fn, n_first, fit["mu_c"], s, state_times)))
            sig_b = rms_resid(
                lambda t, n: 100.0 * max(float(e_first), fit["p_c"] * k_map[t]) / n)
        sig_c = None
        if line is not None and len(pts) >= 2:
            sig_c = (sum((y - (aa + bb * x)) ** 2 for (x, y) in pts)
                     / len(pts)) ** 0.5
        # --- ensemble median in logit space, weighted by fit quality.
        # The log-time trend gets a 2x prior weight: it extrapolates the
        # observed rate trajectory directly and is the most accurate
        # variant for fast-rising trophies, where the structural fits
        # are identified only from early data and extrapolate badly.
        vals = [(logit(r_a), sig_a), (logit(r_b), sig_b), (logit(r_c), sig_c)]
        wts = []
        for j, (_, sv) in enumerate(vals):
            sv = 0.5 if sv is None else min(max(sv, 0.1), 1.0)
            wts.append((2.0 if j == 2 else 1.0) / (sv * sv + 0.05))
        tot = sum(wts)
        med_logit = sum(w * v for w, (v, _) in zip(wts, vals)) / tot
        # --- spread.
        sigma_dis = math.sqrt(
            sum(w * (v - med_logit) ** 2 for w, (v, _) in zip(wts, vals)) / tot)
        sigma_dis = max(sigma_dis, 0.05)
        sigma_h = 0.12 * math.sqrt(max(horizon_days, 0.0))
        sigma_flat = 0.9 if flat else 0.0
        sigma = math.sqrt(
            fit["sigma_fit"] ** 2 + sigma_h ** 2 + sigma_dis ** 2 + sigma_flat ** 2
        )
        sigma = min(max(sigma, 0.4), 2.5)
        med = 100.0 * sigmoid(med_logit)
        # Saturated trophies: the page says essentially everyone has
        # earned it, and the first mission is earned within hours of
        # joining, so the rate stays near 100% as new players join.  The
        # dilution variants would wrongly send it toward 0; override
        # both the median and the spread for this one case.
        if e_last > 0 and 100.0 * e_last / float(states[-1][1]) >= 99.9:
            med = 94.0
            sigma = 1.0
        if not fit["moving"] and e_last == 0:
            sigma = max(sigma, 1.5)
        qs = []
        for z in Z:
            q = 100.0 * sigmoid(logit(med) + z * sigma)
            qs.append(round4(q))
        # Enforce non-decreasing after rounding, and [0, 100].
        fixed = []
        prev = -1.0
        for q in qs:
            q = min(max(q, prev), 100.0)
            fixed.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": fixed})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
