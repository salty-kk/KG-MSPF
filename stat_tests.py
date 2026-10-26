"""Statistical comparison of survival models.

* ``paired_bootstrap_cindex``: patient-level paired bootstrap of the difference
  in Harrell's C-index between two risk scores evaluated on the same patients.
  Implemented with multinomial bootstrap weights so that every replicate is
  computed exactly (pairs between copies of the same patient are never
  comparable because their times are equal), but without re-sorting.
* ``corrected_resampled_ttest``: Nadeau & Bengio (2003) corrected repeated
  cross-validation t-test.
* ``holm_adjust``: Holm step-down adjustment of a family of p-values.
* ``bootstrap_ci``: percentile bootstrap interval of a single C-index.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats

from survival_utils import concordance_index_fast


def _pair_matrices(times, events, risk):
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk, dtype=float)
    comparable = (t[None, :] > t[:, None]) & (e[:, None] == 1)
    diff = r[:, None] - r[None, :]
    score = (diff > 0).astype(np.float32) + 0.5 * (diff == 0).astype(np.float32)
    return comparable.astype(np.float32), score * comparable


def _weighted_c(counts: np.ndarray, comp: np.ndarray, conc: np.ndarray) -> np.ndarray:
    """C-index for each row of bootstrap count vectors (rows x n)."""
    den = np.einsum("bi,bi->b", counts @ comp, counts)
    num = np.einsum("bi,bi->b", counts @ conc, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def paired_bootstrap_cindex(
    times,
    events,
    risk_a,
    risk_b,
    n_boot: int = 2000,
    seed: int = 0,
    chunk: int = 200,
) -> Dict[str, float]:
    """Paired bootstrap of C(a) - C(b) on the same patients."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    n = t.size
    comp, conc_a = _pair_matrices(t, e, risk_a)
    _, conc_b = _pair_matrices(t, e, risk_b)
    point_a = concordance_index_fast(t, risk_a, e)
    point_b = concordance_index_fast(t, risk_b, e)
    rng = np.random.default_rng(seed)
    ca: List[np.ndarray] = []
    cb: List[np.ndarray] = []
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.zeros((b, n), dtype=np.float32)
        np.add.at(counts, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
        ca.append(_weighted_c(counts, comp, conc_a))
        cb.append(_weighted_c(counts, comp, conc_b))
        done += b
    ca_arr = np.concatenate(ca)
    cb_arr = np.concatenate(cb)
    delta = ca_arr - cb_arr
    delta = delta[np.isfinite(delta)]
    # Finite Monte Carlo resolution. Never print an empirical p-value of zero.
    p_two = 2.0 * min((np.sum(delta <= 0) + 1) / (delta.size + 1),
                     (np.sum(delta >= 0) + 1) / (delta.size + 1))
    return {
        "c_a": float(point_a),
        "c_b": float(point_b),
        "delta": float(point_a - point_b),
        "delta_lo": float(np.percentile(delta, 2.5)),
        "delta_hi": float(np.percentile(delta, 97.5)),
        "p_bootstrap": float(min(1.0, p_two)),
        "c_a_lo": float(np.nanpercentile(ca_arr, 2.5)),
        "c_a_hi": float(np.nanpercentile(ca_arr, 97.5)),
        "c_b_lo": float(np.nanpercentile(cb_arr, 2.5)),
        "c_b_hi": float(np.nanpercentile(cb_arr, 97.5)),
        "n_boot": int(delta.size),
    }


def bootstrap_ci(T, risk, E, metric_fn=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of Harrell's C (metric_fn kept for compatibility)."""
    t = np.asarray(T, dtype=float)
    e = np.asarray(E, dtype=int)
    comp, conc = _pair_matrices(t, e, risk)
    rng = np.random.default_rng(seed)
    n = t.size
    vals = []
    done = 0
    while done < n_boot:
        b = min(200, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.zeros((b, n), dtype=np.float32)
        np.add.at(counts, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
        vals.append(_weighted_c(counts, comp, conc))
        done += b
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return float(np.mean(v)), float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def paired_ttest(scores_a, scores_b):
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0
    t_stat, p_value = stats.ttest_rel(a, b)
    return float(t_stat), float(p_value)


def corrected_resampled_ttest(diffs: Sequence[float], n_train: float, n_test: float) -> Dict[str, float]:
    """Nadeau & Bengio corrected t-test for repeated k-fold CV differences.

    var_corrected = (1/J + n_test/n_train) * s^2, with J = number of splits.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    J = d.size
    if J < 2:
        return {"mean": float(np.mean(d)) if J else float("nan"), "t": float("nan"), "p": float("nan"), "df": J - 1}
    var = np.var(d, ddof=1)
    denom = np.sqrt((1.0 / J + n_test / n_train) * var)
    t = float(np.mean(d) / denom) if denom > 0 else (0.0 if np.mean(d) == 0 else float("inf") * np.sign(np.mean(d)))
    p = float(2 * stats.t.sf(abs(t), df=J - 1))
    half = stats.t.ppf(0.975, df=J - 1) * denom
    return {"mean": float(np.mean(d)), "t": t, "p": p, "df": J - 1,
            "ci_lo": float(np.mean(d) - half), "ci_hi": float(np.mean(d) + half),
            "wins": int(np.sum(d > 0)), "n_splits": J}


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def sig_mark(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
