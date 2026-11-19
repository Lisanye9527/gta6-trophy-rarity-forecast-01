#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates for make.fun (gta6-trophy-rarity-forecast-01).

The program is run once per case in a sandbox. It reads every ``*.in`` case
file in the input directory (``$MAKEFUN_INPUT_DIR`` or ``/input``) and writes
one ``<id>.out`` file per case to the output directory
(``$MAKEFUN_OUTPUT_DIR`` or ``/output``). Each output row gives five
quantiles of one trophy's earn rate (the percent of the players Exophase
tracks for the game who have earned it) at the case's target instant.

Method
------
Two views of the same series are combined per trophy.

* Trend view: a weighted least squares fit of the rate's logit against
  log time since release, x = ln(1 + hours) (a concave quadratic if the
  data bend that way, else a line), extrapolated to the target with its
  increment damped. Log time compresses the long final horizon, and the
  damping keeps a hot recent slope from running away.

* Decomposition view: a rate is earners over players, rate = E / P, and
  the two parts behave differently -- earners only grow, while the
  tracked base P grows on its own curve and dilutes every rate when it
  outruns the earners. Both are fitted as saturating growth curves in
  log space over log time (see GrowthFit). The earners forecast is
  capped by the trophy's eventual share A (inferred from the highest
  rate it has shown, with headroom): E(target) <= A * P(target), so a
  common trophy can approach its ceiling but never blow past it.

The median is the average of the two views on the logit scale, pulled
back toward the last observed rate if it ever strays implausibly far.
Consecutive identical page states (the page only changes when Exophase
refreshes it) are compressed before fitting so flat stretches do not
outvote the moves.

The spread is assembled on the logit scale (the scale the score uses)
from independent parts: a random-walk term growing with the horizon, the
trend fit's residual scatter inflated for extrapolation, a small-count
term for trophies held by only a handful of players, and a staleness
term when the recorded page never changed during the history (its next
refresh is still to come, and it will move every rate at once).

Every trophy always gets a valid row: trophies with no usable history
fall back to a per-tier prior, and each trophy's modelling is guarded so
one bad series can never invalidate the case.

Standard library only. No randomness, no clock, no network: two runs on
the same case write byte-identical output.
"""

import glob
import json
import math
import os
from datetime import datetime, timezone
from statistics import NormalDist

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
_NORM = NormalDist()

# Fallback medians (percent) and logit spread for a trophy with no data.
TIER_PRIOR = {
    "platinum": (0.40, 1.30),
    "gold": (1.50, 1.30),
    "silver": (4.00, 1.25),
    "bronze": (8.00, 1.25),
}
DEFAULT_PRIOR = (3.00, 1.30)

RECENCY_HALF_LIFE_H = 96.0     # fit weights halve this many hours back
EARNER_EPS = 0.25              # continuity correction for log earners
MAX_EARNER_GROWTH = 14.0       # cap on E(target) / E(last), plus a slack
TREND_DAMP = 0.90              # damping on the trend's extrapolation
EARNER_DAMP = 0.92             # damping on the earners extrapolation
LOGIT_SHRINK_START = 2.2       # median may not stray further than this
SIGMA_FLOOR = 0.22
SIGMA_CAP = 1.50


# --------------------------------------------------------------------------
# small numeric helpers

def clamp_rate(x):
    """Clamp a percent rate to the range where its logit is defined."""
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_rate(float(x)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    """Inverse logit back to percent, stable for large |lg|."""
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def quantile_row(median_pct, sigma):
    """Five quantiles around a median on the logit scale, non-decreasing."""
    centre = logit_pct(median_pct)
    row = []
    for tau in LEVELS:
        q = inv_logit_pct(centre + _NORM.inv_cdf(tau) * sigma)
        q = min(100.0, max(0.0, round(q, 4)))
        if row and q < row[-1]:
            q = row[-1]
        row.append(q)
    return row


def parse_time(text):
    """Parse an ISO 8601 date or datetime to epoch seconds (UTC)."""
    if not text:
        return None
    s = str(text).strip()
    try:
        if len(s) == 10:  # a bare date, read as midnight UTC
            dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def solve_linear(a, b):
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def wls_poly(xs, ys, ws, degree):
    """Weighted least squares polynomial; returns coefficients or None."""
    if not xs:
        return None
    size = degree + 1
    ata = [[0.0] * size for _ in range(size)]
    atb = [0.0] * size
    for x, y, w in zip(xs, ys, ws):
        if w <= 0.0:
            continue
        powers = [1.0]
        for _ in range(degree):
            powers.append(powers[-1] * x)
        for i in range(size):
            atb[i] += w * powers[i] * y
            for j in range(size):
                ata[i][j] += w * powers[i] * powers[j]
    return solve_linear(ata, atb)


def poly_eval(coef, x):
    total = 0.0
    for c in reversed(coef):
        total = total * x + c
    return total


def fit_curve(xs, ys, ws):
    """Fit y(x): concave quadratic if it bends that way, else line, else
    constant. Returns (coefficients, weighted residual scatter)."""
    coef = None
    if len(xs) >= 3:
        coef = wls_poly(xs, ys, ws, 2)
        if coef is not None and coef[2] > 0.0:
            coef = wls_poly(xs, ys, ws, 1)
    if coef is None and len(xs) >= 2:
        coef = wls_poly(xs, ys, ws, 1)
    if coef is None:
        wsum = sum(ws) or 1.0
        coef = [sum(w * y for w, y in zip(ws, ys)) / wsum]
    num = 0.0
    den = 0.0
    for x, y, w in zip(xs, ys, ws):
        r = y - poly_eval(coef, x)
        num += w * r * r
        den += w
    resid = math.sqrt(num / den) if den > 0.0 and len(xs) > len(coef) else 0.15
    return coef, resid


class GrowthFit:
    """A saturating growth curve in log space over log time, used for
    quantities that cannot shrink (players, earners): predictions are
    floored at the fitted value at the last observed x."""

    def __init__(self, xs, ys, ws):
        self.x_last = max(xs) if xs else 0.0
        self.coef, self.resid = fit_curve(xs, ys, ws)
        self.y_last = poly_eval(self.coef, self.x_last)

    def predict(self, x, damp=1.0):
        raw = poly_eval(self.coef, x)
        return max(self.y_last + damp * (raw - self.y_last), self.y_last)


# --------------------------------------------------------------------------
# case preparation

def prepare_reads(case):
    """Normalise the history into (hours_since_release, players, rates),
    with consecutive identical page states compressed away."""
    game = case.get("game") or {}
    release_ts = parse_time(game.get("release_date"))
    history = case.get("history") or []
    reads = []
    for read in history:
        ts = parse_time(read.get("at") or read.get("slot"))
        if ts is None:
            continue
        players = read.get("players_tracked")
        try:
            players = float(players) if players is not None else None
        except (TypeError, ValueError):
            players = None
        if players is not None and players <= 0.0:
            players = None
        reads.append([ts, players, read.get("rates") or []])
    reads.sort(key=lambda r: r[0])
    if release_ts is None:
        release_ts = reads[0][0] if reads else parse_time(case.get("at")) or 0.0
    # fill missing player counts from the nearest read that has one
    known = [i for i, r in enumerate(reads) if r[1] is not None]
    if known:
        for i, r in enumerate(reads):
            if r[1] is None:
                nearest = min(known, key=lambda k: abs(k - i))
                r[1] = reads[nearest][1]
    full = [(max(-0.9, (ts - release_ts) / 3600.0), players, rates)
            for ts, players, rates in reads]
    out = []
    prev_key = None
    for hours, players, rates in full:
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append((hours, players, rates))
    # Always keep the final read as well: its timestamp carries the span
    # of a history whose page never changed, and the true last-observation
    # time for the fits.
    if full and out and full[-1][0] != out[-1][0]:
        out.append(full[-1])
    return out, release_ts


def recency_weights(hours_list):
    if not hours_list:
        return []
    t_last = max(hours_list)
    return [0.5 ** ((t_last - h) / RECENCY_HALF_LIFE_H) for h in hours_list]


# --------------------------------------------------------------------------
# forecasting

def forecast_players(reads, x_target, horizon_h):
    """Forecast players_tracked at the target; returns (P_target, stale)."""
    pts = [(math.log1p(h), math.log(p)) for h, p, _ in reads if p]
    if not pts:
        return None, False
    p_last = [p for _, p, _ in reads if p][-1]
    distinct = {p for _, p, _ in reads if p}
    span_h = reads[-1][0] - reads[0][0] if len(reads) > 1 else 0.0
    stale = len(distinct) == 1 and span_h >= 0.5
    if stale:
        # The page never refreshed during the history: the tracked base at
        # the target will be larger than the frozen count suggests.
        growth = 1.0 + 4.0 * (1.0 - math.exp(-horizon_h / 48.0))
        return p_last * growth, True
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    ws = recency_weights([h for h, p, _ in reads if p])
    fit = GrowthFit(xs, ys, ws)
    p_target = math.exp(fit.predict(x_target))
    factor = p_target / p_last if p_last > 0 else 1.0
    factor = min(12.0, max(0.95, factor))
    return p_last * factor, False


def eventual_share_cap(r_max_pct):
    """Upper bound for a trophy's eventual earn share (a fraction), from
    the highest rate it has shown so far, with headroom to grow into."""
    f = clamp_rate(r_max_pct) / 100.0
    if f < 0.5:
        return min(0.995, f * 1.7 + 0.02)
    return min(0.995, f + (1.0 - f) * 0.55)


def forecast_trophy(idx, trophy, reads, x_target, p_target, stale,
                    horizon_days):
    """One trophy's (median percent, logit sigma)."""
    obs = []  # (hours, rate, players)
    for hours, players, rates in reads:
        if idx < len(rates) and rates[idx] is not None:
            try:
                obs.append((hours, float(rates[idx]), players))
            except (TypeError, ValueError):
                continue
    if not obs:
        return TIER_PRIOR.get(trophy.get("tier"), DEFAULT_PRIOR)
    if all(p is None for _, _, p in obs):
        # No player counts anywhere: use a constant base, so earners are
        # proportional to the rates and the ratio math still works.
        obs = [(h, r, 1.0) for h, r, _ in obs]
    else:
        filled = []
        last_p = next(p for _, _, p in obs if p is not None)
        for h, r, p in obs:
            if p is None:
                p = last_p
            else:
                last_p = p
            filled.append((h, r, p))
        obs = filled
    r_last = obs[-1][1]
    e_last = r_last * obs[-1][2] / 100.0
    xs = [math.log1p(h) for h, _, _ in obs]
    base_w = recency_weights([h for h, _, _ in obs])

    # Trend view: damped extrapolation of the rate's logit.
    ys_logit = [logit_pct(r) for _, r, _ in obs]
    t_coef, t_resid = fit_curve(xs, ys_logit, base_w)
    y_at_last = poly_eval(t_coef, xs[-1])
    trend_logit = y_at_last + TREND_DAMP * (poly_eval(t_coef, x_target) - y_at_last)

    if p_target is None:
        median = inv_logit_pct(trend_logit)
        sigma = math.sqrt((0.20 * math.sqrt(horizon_days)) ** 2 +
                          min(0.45, t_resid) ** 2 + 0.35 ** 2)
        return clamp_rate(median), min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    # Decomposition view: earners over players, earners capped by the
    # trophy's eventual share of the target player base.
    ys_e = [math.log(r * p / 100.0 + EARNER_EPS) for _, r, p in obs]
    ws_e = [w * math.sqrt(p / obs[-1][2]) for w, (_, _, p) in zip(base_w, obs)]
    e_fit = GrowthFit(xs, ys_e, ws_e)
    e_target = math.exp(e_fit.predict(x_target, damp=EARNER_DAMP)) - EARNER_EPS
    e_target = max(e_target, e_last * 0.98)
    e_target = min(e_target, e_last * MAX_EARNER_GROWTH + 10.0)
    cap_rate = 100.0 * eventual_share_cap(max(r for _, r, _ in obs))
    # The cap must not contradict the trend view: a trophy whose rate is
    # still climbing steeply is young, and its eventual share sits well
    # above anything it has shown so far.
    cap_rate = max(cap_rate, min(99.5, inv_logit_pct(trend_logit) * 1.25))
    dec_rate = min(100.0 * e_target / p_target, cap_rate)

    # Combine the two views on the logit scale.
    lg = 0.5 * trend_logit + 0.5 * logit_pct(dec_rate)
    # Safety rail: shrink a median that strays too far from the last rate.
    lg_last = logit_pct(r_last)
    if abs(lg - lg_last) > LOGIT_SHRINK_START:
        over = abs(lg - lg_last) - LOGIT_SHRINK_START
        lg = lg_last + math.copysign(LOGIT_SHRINK_START + 0.5 * over, lg - lg_last)
    median = clamp_rate(inv_logit_pct(lg))

    # Spread, on the logit scale, from independent parts.
    span_x = max(0.8, xs[-1] - xs[0]) if len(xs) > 1 else 0.8
    sigma_fit = min(0.45, t_resid * (1.0 + 0.6 * (x_target - xs[-1]) / span_x))
    sigma_rw = min(0.60, 0.20 * math.sqrt(horizon_days))
    sigma_count = min(0.65, 0.85 / math.sqrt(e_last + 1.0))
    sigma_stale = 0.50 if stale else 0.0
    sigma = math.sqrt(sigma_fit ** 2 + sigma_rw ** 2 +
                      sigma_count ** 2 + sigma_stale ** 2)
    return median, min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_case(case):
    trophies = case.get("trophies") or []
    reads, release_ts = prepare_reads(case)
    at_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    if target_ts is None and at_ts is not None:
        target_ts = at_ts + float(case.get("horizon_minutes") or 0) * 60.0
    if target_ts is None:
        target_ts = (reads[-1][0] * 3600.0 + release_ts) if reads else release_ts
    horizon_h = max(0.0, (target_ts - (at_ts or target_ts)) / 3600.0)
    horizon_days = horizon_h / 24.0
    x_target = math.log1p(max(-0.9, (target_ts - release_ts) / 3600.0))
    p_target, stale = forecast_players(reads, x_target, horizon_h)

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id", idx + 1)
        try:
            median, sigma = forecast_trophy(
                idx, trophy, reads, x_target, p_target, stale, horizon_days)
        except Exception:
            median, sigma = TIER_PRIOR.get(trophy.get("tier"), DEFAULT_PRIOR)
        rows.append({"id": tid, "q": quantile_row(median, sigma)})
    return {"trophies": rows}


def fallback_case(case):
    """Last-resort valid output built from tier priors alone."""
    rows = []
    for idx, trophy in enumerate(case.get("trophies") or []):
        median, s = TIER_PRIOR.get(trophy.get("tier"), DEFAULT_PRIOR)
        rows.append({"id": trophy.get("id", idx + 1), "q": quantile_row(median, s)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        case_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            continue
        try:
            result = forecast_case(case)
        except Exception:
            try:
                result = fallback_case(case)
            except Exception:
                continue
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
