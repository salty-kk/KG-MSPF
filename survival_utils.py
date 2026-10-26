"""Survival analysis utilities (numpy / scipy only, no lifelines or sksurv).

Contents
--------
* Kaplan-Meier estimator and censoring distribution (for IPCW weights).
* Harrell's C-index (vectorised) and Uno's IPCW C-index.
* Cumulative/dynamic time-dependent AUC with IPCW weights (Uno et al. 2007).
* IPCW Brier score (Graf et al. 1999) and integrated Brier score.
* Cox proportional-hazards regression with Breslow ties:
    - ridge / unpenalised fit (L-BFGS),
    - elastic-net / LASSO fit (FISTA proximal gradient) with an inner-CV path,
    - Newton-Raphson fit with standard errors for low-dimensional models.
* Breslow baseline hazard and survival prediction.
* Calibration: decile calibration table, ICI / E50 / E90 (Austin et al. 2020),
  calibration slope.
* k-group log-rank test and hazard ratios of risk groups.

Every likelihood used here is normalised by the number of samples, so the
penalty strength ``alpha`` is comparable across data sets of different size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.model_selection import StratifiedKFold


# =====================================================================
# Step functions and Kaplan-Meier
# =====================================================================


class StepFunction:
    """Right-continuous step function f(t) = y[k] for x[k] <= t < x[k+1]."""

    def __init__(self, x: np.ndarray, y: np.ndarray, y0: float = 1.0) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.y0 = float(y0)

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if self.x.size == 0:
            return np.full(t.shape, self.y0)
        idx = np.searchsorted(self.x, t, side="right") - 1
        out = np.where(idx < 0, self.y0, self.y[np.clip(idx, 0, None)])
        return out

    def left_limit(self, t) -> np.ndarray:
        """Value just before t, i.e. f(t-)."""
        t = np.asarray(t, dtype=float)
        if self.x.size == 0:
            return np.full(t.shape, self.y0)
        idx = np.searchsorted(self.x, t, side="left") - 1
        return np.where(idx < 0, self.y0, self.y[np.clip(idx, 0, None)])


def kaplan_meier(times, events) -> StepFunction:
    """Kaplan-Meier survival estimate as a right-continuous step function."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    uniq = np.unique(t[e == 1])
    if uniq.size == 0:
        return StepFunction(np.array([]), np.array([]))
    t_sorted = np.sort(t)
    n_at_risk = t.size - np.searchsorted(t_sorted, uniq, side="left")
    te = np.sort(t[e == 1])
    deaths = np.searchsorted(te, uniq, side="right") - np.searchsorted(te, uniq, side="left")
    surv = np.cumprod(1.0 - deaths / np.maximum(n_at_risk, 1))
    return StepFunction(uniq, surv)


def censoring_distribution(times, events) -> StepFunction:
    """Kaplan-Meier estimate G(t) of the censoring survival function."""
    e = np.asarray(events, dtype=int)
    return kaplan_meier(times, 1 - e)


# Backwards-compatible class (used by older plotting code)
class KaplanMeierEstimator:
    def __init__(self) -> None:
        self.timeline = None
        self.survival_function = None
        self._fn: Optional[StepFunction] = None

    def fit(self, event_times, event_observed):
        self._fn = kaplan_meier(event_times, event_observed)
        self.timeline = self._fn.x
        self.survival_function = self._fn.y
        return self

    def predict_survival(self, time_point):
        return float(self._fn(np.asarray([time_point]))[0])


# =====================================================================
# Concordance
# =====================================================================


def concordance_index_fast(event_times, predicted_risk, event_observed) -> float:
    """Harrell's C-index (higher risk = earlier event).

    Comparable pairs: i has an event and T_j > T_i (strict).  Risk ties count 0.5.
    """
    t = np.asarray(event_times, dtype=float)
    r = np.asarray(predicted_risk, dtype=float)
    e = np.asarray(event_observed, dtype=int)
    idx = np.where(e == 1)[0]
    if idx.size == 0:
        return 0.5
    concordant = 0.0
    total = 0.0
    n_chunks = max(1, int(np.ceil(idx.size * t.size / 4_000_000)))
    for chunk in np.array_split(idx, n_chunks):
        comparable = t[None, :] > t[chunk, None]
        diff = r[chunk, None] - r[None, :]
        concordant += float(((diff > 0) & comparable).sum()) + 0.5 * float(((diff == 0) & comparable).sum())
        total += float(comparable.sum())
    return concordant / total if total > 0 else 0.5


concordance_index = concordance_index_fast


def concordance_index_ipcw(
    train_times,
    train_events,
    test_times,
    test_events,
    risk,
    tau: Optional[float] = None,
) -> float:
    """Uno's C-index with IPCW weights 1/G(T_i-)^2, truncated at tau."""
    G = censoring_distribution(train_times, train_events)
    t = np.asarray(test_times, dtype=float)
    e = np.asarray(test_events, dtype=int)
    r = np.asarray(risk, dtype=float)
    if tau is None:
        tau = float(np.max(t))
    idx = np.where((e == 1) & (t < tau))[0]
    if idx.size == 0:
        return 0.5
    g = G.left_limit(t[idx])
    keep = g > 0
    idx = idx[keep]
    w = 1.0 / np.square(g[keep])
    num = 0.0
    den = 0.0
    n_chunks = max(1, int(np.ceil(idx.size * t.size / 4_000_000)))
    for chunk, wc in zip(np.array_split(idx, n_chunks), np.array_split(w, n_chunks)):
        comparable = t[None, :] > t[chunk, None]
        diff = r[chunk, None] - r[None, :]
        score = (diff > 0).astype(float) + 0.5 * (diff == 0)
        num += float((wc[:, None] * score * comparable).sum())
        den += float((wc[:, None] * comparable).sum())
    return num / den if den > 0 else 0.5


# =====================================================================
# Time-dependent AUC and Brier score (IPCW)
# =====================================================================


def time_dependent_auc(
    test_times,
    test_events,
    risk,
    eval_time: float,
    train_times=None,
    train_events=None,
) -> float:
    """Cumulative/dynamic AUC(t) with IPCW weights (Uno et al. 2007).

    Cases: T_i <= t with an event, weighted by 1/G(T_i-).
    Controls: T_j > t (weight 1/G(t), constant, cancels).
    If no training data are supplied the censoring distribution is estimated
    on the evaluation set itself.
    """
    t = np.asarray(test_times, dtype=float)
    e = np.asarray(test_events, dtype=int)
    r = np.asarray(risk, dtype=float)
    if train_times is None:
        train_times, train_events = t, e
    G = censoring_distribution(train_times, train_events)
    cases = (t <= eval_time) & (e == 1)
    controls = t > eval_time
    if cases.sum() == 0 or controls.sum() == 0:
        return float("nan")
    g = G.left_limit(t[cases])
    valid = g > 0
    w = np.zeros_like(g)
    w[valid] = 1.0 / g[valid]
    rc = r[cases]
    rn = r[controls]
    # rank-based computation
    rn_sorted = np.sort(rn)
    less = np.searchsorted(rn_sorted, rc, side="left")
    equal = np.searchsorted(rn_sorted, rc, side="right") - less
    score = (less + 0.5 * equal) / rn.size
    return float(np.sum(w * score) / np.sum(w)) if np.sum(w) > 0 else float("nan")


def brier_score_ipcw(
    test_times,
    test_events,
    surv_prob_at_t,
    eval_time: float,
    train_times=None,
    train_events=None,
) -> float:
    """Graf et al. IPCW Brier score at time t."""
    t = np.asarray(test_times, dtype=float)
    e = np.asarray(test_events, dtype=int)
    s = np.asarray(surv_prob_at_t, dtype=float)
    if train_times is None:
        train_times, train_events = t, e
    G = censoring_distribution(train_times, train_events)
    died = (t <= eval_time) & (e == 1)
    alive = t > eval_time
    g_died = G.left_limit(t[died])
    g_t = float(G(np.asarray([eval_time]))[0])
    contrib = np.zeros(t.size)
    gd = np.where(g_died > 0, g_died, np.inf)
    contrib[died] = np.square(s[died]) / gd
    if g_t > 0:
        contrib[alive] = np.square(1.0 - s[alive]) / g_t
    return float(np.mean(contrib))


def integrated_brier_score(
    test_times,
    test_events,
    surv_matrix: np.ndarray,
    time_grid: Sequence[float],
    train_times=None,
    train_events=None,
) -> float:
    """Integrated Brier score over the time grid (trapezoid rule)."""
    grid = np.asarray(time_grid, dtype=float)
    scores = np.asarray([
        brier_score_ipcw(test_times, test_events, surv_matrix[:, k], grid[k], train_times, train_events)
        for k in range(grid.size)
    ])
    if grid.size < 2:
        return float(scores[0])
    return float(np.trapezoid(scores, grid) / (grid[-1] - grid[0])) if hasattr(np, "trapezoid") else float(
        np.trapz(scores, grid) / (grid[-1] - grid[0])
    )


def brier_score(event_times, event_observed, predicted_survival_prob, eval_time):
    """IPCW Brier score with the censoring distribution of the evaluation set."""
    return brier_score_ipcw(event_times, event_observed, predicted_survival_prob, eval_time)


# =====================================================================
# Cox proportional hazards
# =====================================================================


def _risk_set_layout(times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order by descending time and the index of the last tied member (Breslow)."""
    order = np.argsort(-times, kind="mergesort")
    ts = times[order]
    end_idx = np.searchsorted(-ts, -ts, side="right") - 1
    return order, end_idx


def cox_negative_log_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    events_sorted: np.ndarray,
    order: np.ndarray,
    end_idx: np.ndarray,
    with_gradient: bool = True,
):
    """Normalised negative Cox partial log-likelihood (Breslow ties) and gradient."""
    n = X.shape[0]
    eta = X @ beta
    eta_s = eta[order]
    m = float(np.max(eta_s)) if eta_s.size else 0.0
    w = np.exp(eta_s - m)
    cw = np.cumsum(w)
    risk_sum = cw[end_idx]
    ev = events_sorted == 1
    ll = float(np.sum(eta_s[ev] - m - np.log(risk_sum[ev])))
    if not with_gradient:
        return -ll / n
    Xs = X[order]
    cwx = np.cumsum(w[:, None] * Xs, axis=0)
    risk_x = cwx[end_idx]
    grad_ll = np.sum(Xs[ev] - risk_x[ev] / risk_sum[ev, None], axis=0)
    return -ll / n, -grad_ll / n


class CoxPHRegression:
    """Cox PH regression with optional ridge / elastic-net penalty.

    Objective: (1/n) * NLL + alpha * ((1 - l1_ratio)/2 * ||b||^2 + l1_ratio * ||b||_1).
    Features are standardised internally (training statistics).  When
    ``l1_ratio > 0`` the model is fitted with FISTA (proximal gradient), which
    produces exact zeros.
    """

    def __init__(
        self,
        alpha: float = 1e-6,
        l1_ratio: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        standardize: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.standardize = standardize
        self.coef_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    # -- preprocessing -------------------------------------------------
    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
            self.scale_ = sd
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        return (X - self.mean_) / self.scale_

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    # -- fitting -------------------------------------------------------
    def fit(self, X, T, E, beta0: Optional[np.ndarray] = None) -> "CoxPHRegression":
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        E = np.asarray(E, dtype=int)
        Xs = self._fit_scaler(X)
        self.coef_ = self._fit_standardized(Xs, T, E, beta0)
        return self

    def _fit_standardized(self, Xs, T, E, beta0=None) -> np.ndarray:
        order, end_idx = _risk_set_layout(T)
        es = E[order]
        p = Xs.shape[1]
        b0 = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
        if self.l1_ratio <= 0:
            ridge = self.alpha

            def fun(b):
                f, g = cox_negative_log_likelihood(b, Xs, es, order, end_idx)
                return f + 0.5 * ridge * float(b @ b), g + ridge * b

            res = optimize.minimize(fun, b0, jac=True, method="L-BFGS-B",
                                    options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-7})
            self.n_iter_ = int(res.nit)
            return res.x
        return self._fista(Xs, es, order, end_idx, b0)

    def _fista(self, Xs, es, order, end_idx, b0) -> np.ndarray:
        lam1 = self.alpha * self.l1_ratio
        lam2 = self.alpha * (1.0 - self.l1_ratio)

        def smooth(b):
            f, g = cox_negative_log_likelihood(b, Xs, es, order, end_idx)
            return f + 0.5 * lam2 * float(b @ b), g + lam2 * b

        def prox(v, step):
            return np.sign(v) * np.maximum(np.abs(v) - step * lam1, 0.0)

        b = b0.copy()
        y = b.copy()
        t_k = 1.0
        L = 1.0
        f_prev = np.inf
        for it in range(self.max_iter):
            fy, gy = smooth(y)
            while True:
                b_new = prox(y - gy / L, 1.0 / L)
                diff = b_new - y
                f_new = smooth(b_new)[0]
                if f_new <= fy + float(gy @ diff) + 0.5 * L * float(diff @ diff) + 1e-15:
                    break
                L *= 2.0
                if L > 1e12:
                    break
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            y = b_new + ((t_k - 1.0) / t_next) * (b_new - b)
            obj = f_new + lam1 * float(np.abs(b_new).sum())
            # restart if objective increased (adaptive restart)
            if obj > f_prev:
                y = b_new.copy()
                t_next = 1.0
            converged = np.max(np.abs(b_new - b)) < self.tol * max(1.0, np.max(np.abs(b)))
            b = b_new
            t_k = t_next
            if abs(f_prev - obj) < 1e-10 and converged:
                break
            f_prev = obj
            L = max(L * 0.9, 1e-4)
        self.n_iter_ = it + 1
        return b

    # -- prediction ----------------------------------------------------
    def predict_risk(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("CoxPHRegression is not fitted")
        return self._transform(X) @ self.coef_

    @property
    def n_nonzero(self) -> int:
        return int(np.sum(np.abs(self.coef_) > 0)) if self.coef_ is not None else 0


def lasso_lambda_max(X, T, E, l1_ratio: float = 1.0) -> float:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mean) / sd
    order, end_idx = _risk_set_layout(np.asarray(T, dtype=float))
    _, g = cox_negative_log_likelihood(np.zeros(X.shape[1]), Xs, np.asarray(E, int)[order], order, end_idx)
    return float(np.max(np.abs(g)) / max(l1_ratio, 1e-3))


def fit_penalized_cox_cv(
    X,
    T,
    E,
    l1_ratio: float = 1.0,
    n_lambda: int = 20,
    lambda_min_ratio: float = 0.02,
    n_folds: int = 3,
    seed: int = 0,
    max_iter: int = 600,
    fold_data=None,
) -> Tuple[CoxPHRegression, Dict[str, object]]:
    """Elastic-net Cox with the penalty chosen by inner K-fold CV (C-index)."""
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    lam_max = max(lasso_lambda_max(X, T, E, l1_ratio), 1e-12)
    ratios = np.geomspace(1.0, lambda_min_ratio, n_lambda)
    lambdas = lam_max * ratios
    scores = np.zeros((n_folds, n_lambda))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (tr, va) in enumerate(splitter.split(X, E)):
        if fold_data is None:
            Xtr, Xva = X[tr], X[va]
        else:
            Xtr, Xva = fold_data[f]
        inner_max = max(lasso_lambda_max(Xtr, T[tr], E[tr], l1_ratio), 1e-12)
        inner_lambdas = inner_max * ratios
        beta = None
        model = CoxPHRegression(alpha=inner_lambdas[0], l1_ratio=l1_ratio, max_iter=max_iter)
        Xs_tr = model._fit_scaler(Xtr)
        for k, lam in enumerate(inner_lambdas):
            model.alpha = lam
            beta = model._fit_standardized(Xs_tr, T[tr], E[tr], beta)
            model.coef_ = beta
            scores[f, k] = concordance_index_fast(T[va], model.predict_risk(Xva), E[va])
    mean_scores = scores.mean(axis=0)
    best = int(np.argmax(mean_scores))
    final = CoxPHRegression(alpha=lambdas[0], l1_ratio=l1_ratio, max_iter=max_iter)
    Xs = final._fit_scaler(X)
    beta = None
    for k in range(best + 1):  # warm-started path down to the selected lambda
        final.alpha = lambdas[k]
        beta = final._fit_standardized(Xs, T, E, beta)
    final.coef_ = beta
    info = {
        "lambdas": lambdas.tolist(),
        "cv_cindex": mean_scores.tolist(),
        "best_lambda": float(lambdas[best]),
        "n_nonzero": final.n_nonzero,
    }
    return final, info


def cox_fit_newton(X, T, E, max_iter: int = 50, tol: float = 1e-9) -> Dict[str, np.ndarray]:
    """Unpenalised Cox fit with standard errors (Breslow ties), for small p."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    n, p = X.shape
    order, end_idx = _risk_set_layout(T)
    Xs = X[order]
    es = E[order]
    ev = es == 1
    beta = np.zeros(p)
    hess = np.eye(p)
    for _ in range(max_iter):
        eta = Xs @ beta
        m = eta.max()
        w = np.exp(eta - m)
        S0 = np.cumsum(w)[end_idx]
        S1 = np.cumsum(w[:, None] * Xs, axis=0)[end_idx]
        S2 = np.cumsum(w[:, None, None] * Xs[:, :, None] * Xs[:, None, :], axis=0)[end_idx]
        xbar = S1[ev] / S0[ev, None]
        grad = np.sum(Xs[ev] - xbar, axis=0)
        hess = np.sum(S2[ev] / S0[ev, None, None], axis=0) - xbar.T @ xbar
        try:
            step = np.linalg.solve(hess + 1e-10 * np.eye(p), grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    try:
        cov = np.linalg.inv(hess + 1e-10 * np.eye(p))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    z = beta / np.where(se > 0, se, np.nan)
    pval = 2 * stats.norm.sf(np.abs(z))
    return {"coef": beta, "se": se, "z": z, "p": pval}


# =====================================================================
# Breslow baseline hazard
# =====================================================================


class BreslowSurvival:
    """Breslow estimator of the baseline cumulative hazard for a risk score."""

    def __init__(self) -> None:
        self.event_times: Optional[np.ndarray] = None
        self.cum_hazard: Optional[np.ndarray] = None
        self.center: float = 0.0

    def fit(self, risk, T, E) -> "BreslowSurvival":
        r = np.asarray(risk, dtype=float)
        T = np.asarray(T, dtype=float)
        E = np.asarray(E, dtype=int)
        self.center = float(np.mean(r))
        w = np.exp(np.clip(r - self.center, -50, 50))
        uniq = np.unique(T[E == 1])
        order = np.argsort(T)
        T_sorted = T[order]
        w_sorted = w[order]
        rev_cum = np.cumsum(w_sorted[::-1])[::-1]
        first = np.searchsorted(T_sorted, uniq, side="left")
        denom = rev_cum[first]
        te = np.sort(T[E == 1])
        deaths = np.searchsorted(te, uniq, side="right") - np.searchsorted(te, uniq, side="left")
        self.event_times = uniq
        self.cum_hazard = np.cumsum(deaths / np.maximum(denom, 1e-300))
        return self

    def baseline_cumulative_hazard(self, times) -> np.ndarray:
        fn = StepFunction(self.event_times, self.cum_hazard, y0=0.0)
        return fn(np.asarray(times, dtype=float))

    def survival(self, risk, times) -> np.ndarray:
        """Survival matrix [n_patients, n_times]."""
        H0 = self.baseline_cumulative_hazard(times)
        w = np.exp(np.clip(np.asarray(risk, dtype=float) - self.center, -50, 50))
        return np.exp(-np.outer(w, H0))


def step_survival_on_grid(curve_times: np.ndarray, surv: np.ndarray, grid: Sequence[float]) -> np.ndarray:
    """Evaluate right-continuous survival curves (rows) given at curve_times on a grid."""
    curve_times = np.asarray(curve_times, dtype=float)
    grid = np.asarray(grid, dtype=float)
    idx = np.searchsorted(curve_times, grid, side="right") - 1
    out = np.ones((surv.shape[0], grid.size))
    valid = idx >= 0
    out[:, valid] = surv[:, idx[valid]]
    return out


# =====================================================================
# Calibration
# =====================================================================


def calibration_table(T, E, surv_at_t, t: float, n_groups: int = 10) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier observed event probability by risk decile."""
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    pred_fail = 1.0 - np.asarray(surv_at_t, dtype=float)
    ranks = pd.qcut(pd.Series(pred_fail).rank(method="first"), n_groups, labels=False).to_numpy()
    rows = []
    for g in range(n_groups):
        m = ranks == g
        if m.sum() < 5:
            continue
        km = kaplan_meier(T[m], E[m])
        obs = 1.0 - float(km(np.asarray([t]))[0])
        # Greenwood-type binomial approximation for the error bar
        se = np.sqrt(max(obs * (1 - obs), 1e-12) / m.sum())
        rows.append({"group": g + 1, "n": int(m.sum()), "predicted": float(pred_fail[m].mean()),
                     "observed": obs, "observed_lo": max(0.0, obs - 1.96 * se),
                     "observed_hi": min(1.0, obs + 1.96 * se)})
    return pd.DataFrame(rows)


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell) with the linear term first."""
    k = np.asarray(knots, dtype=float)
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    def pos3(v):
        return np.clip(v, 0, None) ** 3
    for j in range(k.size - 2):
        term = (pos3(x - k[j])
                - pos3(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                + pos3(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / norm)
    return np.column_stack(cols)


def calibration_ici(T, E, surv_at_t, t: float) -> Dict[str, float]:
    """ICI, E50, E90 following Austin, Harrell & van Klaveren (2020).

    Observed probabilities are obtained from a Cox model of the outcome on a
    restricted cubic spline (3 knots) of the complementary log-log transform of
    the predicted event probability, fitted on the evaluation set.
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    s = np.clip(np.asarray(surv_at_t, dtype=float), 1e-6, 1 - 1e-6)
    pred_fail = 1.0 - s
    x = np.log(-np.log(s))
    knots = np.quantile(x, [0.1, 0.5, 0.9])
    if np.unique(knots).size < 3:
        basis = x[:, None]
    else:
        basis = _rcs_basis(x, knots)
    fit = cox_fit_newton(basis, T, E)
    lp = basis @ fit["coef"]
    br = BreslowSurvival().fit(lp, T, E)
    obs_fail = 1.0 - br.survival(lp, [t])[:, 0]
    err = np.abs(obs_fail - pred_fail)
    return {
        "ici": float(np.mean(err)),
        "e50": float(np.median(err)),
        "e90": float(np.quantile(err, 0.9)),
        "emax": float(np.max(err)),
        "mean_predicted": float(np.mean(pred_fail)),
        "km_observed": float(1.0 - kaplan_meier(T, E)(np.asarray([t]))[0]),
    }


def calibration_slope(T, E, linear_predictor) -> Dict[str, float]:
    """Slope of the Cox model of the outcome on the model linear predictor."""
    fit = cox_fit_newton(np.asarray(linear_predictor, dtype=float)[:, None], T, E)
    b = float(fit["coef"][0])
    se = float(fit["se"][0])
    return {"slope": b, "slope_lo": b - 1.96 * se, "slope_hi": b + 1.96 * se}


# =====================================================================
# Log-rank tests and risk-group hazard ratios
# =====================================================================


def logrank_test(time1, event1, time2, event2):
    """Two-group log-rank test; returns (chi2, p)."""
    t = np.concatenate([np.asarray(time1, float), np.asarray(time2, float)])
    e = np.concatenate([np.asarray(event1, int), np.asarray(event2, int)])
    g = np.concatenate([np.zeros(len(time1), int), np.ones(len(time2), int)])
    chi2, p, _ = logrank_multigroup(t, e, g)
    return chi2, p


def logrank_multigroup(times, events, groups) -> Tuple[float, float, int]:
    """k-group log-rank test; returns (chi2, p, df)."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    labels, g = np.unique(np.asarray(groups), return_inverse=True)
    k = labels.size
    if k < 2:
        return 0.0, 1.0, 0
    uniq = np.unique(t[e == 1])
    O = np.zeros(k)
    Ex = np.zeros(k)
    V = np.zeros((k, k))
    order = np.argsort(t)
    ts = t[order]
    gs = g[order]
    es = e[order]
    onehot = np.eye(k)[gs]
    rev_n = np.cumsum(onehot[::-1], axis=0)[::-1]
    first = np.searchsorted(ts, uniq, side="left")
    n_g = rev_n[first]  # at risk per group
    death_rows = onehot * es[:, None]
    tsd = ts
    left = np.searchsorted(tsd, uniq, side="left")
    right = np.searchsorted(tsd, uniq, side="right")
    cum_d = np.vstack([np.zeros((1, k)), np.cumsum(death_rows, axis=0)])
    d_g = cum_d[right] - cum_d[left]
    n = n_g.sum(axis=1)
    d = d_g.sum(axis=1)
    valid = n > 1
    O = d_g[valid].sum(axis=0)
    Ex = (n_g[valid] * (d[valid] / n[valid])[:, None]).sum(axis=0)
    for row_n, row_d, nn, dd in zip(n_g[valid], d_g[valid], n[valid], d[valid]):
        frac = row_n / nn
        factor = dd * (nn - dd) / max(nn - 1, 1)
        V += factor * (np.diag(frac) - np.outer(frac, frac))
    diff = (O - Ex)[:-1]
    Vr = V[:-1, :-1]
    try:
        chi2 = float(diff @ np.linalg.solve(Vr, diff))
    except np.linalg.LinAlgError:
        chi2 = float("nan")
    df = k - 1
    p = float(stats.chi2.sf(chi2, df)) if np.isfinite(chi2) else float("nan")
    return chi2, p, df


def risk_group_hazard_ratios(times, events, groups, reference=0) -> pd.DataFrame:
    """Cox hazard ratios of each risk group versus the reference group."""
    g = np.asarray(groups)
    levels = [lv for lv in np.unique(g) if lv != reference]
    X = np.column_stack([(g == lv).astype(float) for lv in levels])
    fit = cox_fit_newton(X, times, events)
    rows = []
    for j, lv in enumerate(levels):
        b, se = fit["coef"][j], fit["se"][j]
        rows.append({"group": lv, "hr": float(np.exp(b)), "hr_lo": float(np.exp(b - 1.96 * se)),
                     "hr_hi": float(np.exp(b + 1.96 * se)), "p": float(fit["p"][j])})
    return pd.DataFrame(rows)


# =====================================================================
# Convenience: evaluate a risk score and survival matrix
# =====================================================================


def evaluate_predictions(
    train_times,
    train_events,
    test_times,
    test_events,
    risk,
    surv_matrix: Optional[np.ndarray],
    time_grid: Sequence[float],
    auc_times: Sequence[float] = (12, 36, 60, 120),
    brier_times: Sequence[float] = (36, 60, 120),
    tau: float = 180.0,
) -> Dict[str, float]:
    """Discrimination and calibration metrics for one test set."""
    grid = np.asarray(time_grid, dtype=float)
    out: Dict[str, float] = {
        "c_index": concordance_index_fast(test_times, risk, test_events),
        "c_uno": concordance_index_ipcw(train_times, train_events, test_times, test_events, risk, tau=tau),
    }
    max_t = float(np.max(test_times))
    for t in auc_times:
        out[f"auc_{int(t)}m"] = (
            time_dependent_auc(test_times, test_events, risk, t, train_times, train_events)
            if t < max_t else float("nan")
        )
    if surv_matrix is not None:
        for t in brier_times:
            k = int(np.argmin(np.abs(grid - t)))
            out[f"brier_{int(t)}m"] = brier_score_ipcw(test_times, test_events, surv_matrix[:, k], grid[k],
                                                       train_times, train_events)
        keep = grid <= min(tau, max_t)
        if keep.sum() >= 2:
            out["ibs"] = integrated_brier_score(test_times, test_events, surv_matrix[:, keep], grid[keep],
                                                train_times, train_events)
        else:
            out["ibs"] = float("nan")
    return out
