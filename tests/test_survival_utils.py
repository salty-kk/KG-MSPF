import numpy as np

import survival_utils as su


def _sim(n=400, p=4, seed=0, censor=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.5, 0.3, 0.0])[:p]
    T = rng.exponential(scale=np.exp(-(X @ beta)) * 50)
    if censor:
        C = rng.exponential(scale=80, size=n)
        E = (T <= C).astype(int)
        T = np.minimum(T, C)
    else:
        E = np.ones(n, dtype=int)
    T = np.round(T, 1) + 0.1  # create ties
    return X, T, E, beta


def _slow_harrell(t, r, e):
    conc = tot = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        for j in range(len(t)):
            if t[j] > t[i]:
                tot += 1
                conc += 1.0 if r[i] > r[j] else (0.5 if r[i] == r[j] else 0.0)
    return conc / tot


def test_harrell_matches_bruteforce():
    X, T, E, _ = _sim(n=300)
    r = np.round(X[:, 0], 1)
    assert abs(su.concordance_index_fast(T, r, E) - _slow_harrell(T, r, E)) < 1e-12


def test_uno_equals_harrell_without_censoring():
    X, T, E, _ = _sim(censor=False)
    r = X[:, 0]
    c1 = su.concordance_index_fast(T, r, E)
    c2 = su.concordance_index_ipcw(T, E, T, E, r, tau=T.max() + 1)
    assert abs(c1 - c2) < 1e-10


def test_km_simple():
    km = su.kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    # S(1)=4/5, S(2)=4/5*3/4=0.6, S(3)=0.6*1/2=0.3
    np.testing.assert_allclose(km([0.5, 1, 2, 3, 10]), [1, 0.8, 0.6, 0.3, 0.3])
    np.testing.assert_allclose(km.left_limit([1, 2, 3]), [1, 0.8, 0.6])


def test_auc_and_brier_without_censoring_match_naive():
    X, T, E, _ = _sim(censor=False)
    r = X[:, 0]
    t = np.median(T)
    cases = T <= t
    controls = T > t
    naive = np.mean([(rc > r[controls]).mean() + 0.5 * (rc == r[controls]).mean() for rc in r[cases]])
    assert abs(su.time_dependent_auc(T, E, r, t) - naive) < 1e-10
    s = np.clip(1 / (1 + np.exp(r)), 0, 1)
    naive_bs = np.mean(np.where(T <= t, s ** 2, (1 - s) ** 2))
    assert abs(su.brier_score_ipcw(T, E, s, t) - naive_bs) < 1e-10


def test_cox_gradient_and_newton_agree():
    X, T, E, beta_true = _sim(n=600)
    order, end_idx = su._risk_set_layout(T)
    b = np.array([0.1, -0.2, 0.05, 0.3])
    f, g = su.cox_negative_log_likelihood(b, X, E[order], order, end_idx)
    eps = 1e-6
    num = np.array([
        (su.cox_negative_log_likelihood(b + eps * np.eye(4)[k], X, E[order], order, end_idx, False)
         - su.cox_negative_log_likelihood(b - eps * np.eye(4)[k], X, E[order], order, end_idx, False)) / (2 * eps)
        for k in range(4)
    ])
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-8)
    newton = su.cox_fit_newton(X, T, E)
    model = su.CoxPHRegression(alpha=0.0, standardize=False).fit(X, T, E)
    np.testing.assert_allclose(model.coef_, newton["coef"], atol=2e-4)
    assert np.all(np.sign(newton["coef"][:3]) == np.sign(beta_true[:3]))


def test_lasso_produces_sparsity_and_cv_runs():
    rng = np.random.default_rng(1)
    X, T, E, _ = _sim(n=500, p=4)
    noise = rng.normal(size=(500, 40))
    Xb = np.hstack([X, noise])
    m = su.CoxPHRegression(alpha=0.05, l1_ratio=1.0).fit(Xb, T, E)
    assert m.n_nonzero < Xb.shape[1]
    assert abs(m.coef_[0]) > 0
    model, info = su.fit_penalized_cox_cv(Xb, T, E, l1_ratio=1.0, n_lambda=8, n_folds=3)
    assert 0 < info["n_nonzero"] <= Xb.shape[1]
    assert su.concordance_index_fast(T, model.predict_risk(Xb), E) > 0.6


def test_breslow_survival_matches_km_on_average():
    X, T, E, _ = _sim(n=800)
    m = su.CoxPHRegression(alpha=1e-6).fit(X, T, E)
    r = m.predict_risk(X)
    br = su.BreslowSurvival().fit(r, T, E)
    grid = np.array([10, 30, 60])
    S = br.survival(r, grid)
    assert np.all(np.diff(S, axis=1) <= 1e-12)
    km = su.kaplan_meier(T, E)(grid)
    np.testing.assert_allclose(S.mean(axis=0), km, atol=0.03)
    cal = su.calibration_ici(T, E, S[:, 1], 30)
    assert cal["ici"] < 0.05
    slope = su.calibration_slope(T, E, r)
    assert 0.8 < slope["slope"] < 1.2
    tab = su.calibration_table(T, E, S[:, 1], 30)
    assert len(tab) == 10


def test_logrank_multigroup_two_groups_matches_reference():
    X, T, E, _ = _sim(n=400)
    g = (X[:, 0] > 0).astype(int)
    chi2, p, df = su.logrank_multigroup(T, E, g)
    # reference implementation (original code)
    from scipy import stats
    t1, e1, t2, e2 = T[g == 0], E[g == 0], T[g == 1], E[g == 1]
    all_t = np.sort(np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]])))
    O1 = E1 = V = 0.0
    for t in all_t:
        n1, n2 = np.sum(t1 >= t), np.sum(t2 >= t)
        d1, d2 = np.sum((t1 == t) & (e1 == 1)), np.sum((t2 == t) & (e2 == 1))
        n, d = n1 + n2, d1 + d2
        if n < 2:
            continue
        O1 += d1
        E1 += n1 * d / n
        V += n1 * n2 * d * (n - d) / (n ** 2 * (n - 1))
    ref = (O1 - E1) ** 2 / V
    assert abs(chi2 - ref) / ref < 1e-8
    assert df == 1
    hr = su.risk_group_hazard_ratios(T, E, g, reference=0)
    assert hr["hr"].iloc[0] > 1


def test_step_survival_on_grid():
    times = np.array([1.0, 2.0, 3.0])
    surv = np.array([[0.9, 0.8, 0.7]])
    out = su.step_survival_on_grid(times, surv, [0.5, 1.0, 2.5, 5])
    np.testing.assert_allclose(out, [[1.0, 0.9, 0.8, 0.7]])


def test_evaluate_predictions_keys():
    X, T, E, _ = _sim(n=400)
    m = su.CoxPHRegression().fit(X, T, E)
    r = m.predict_risk(X)
    grid = np.arange(6, 61, 6)
    S = su.BreslowSurvival().fit(r, T, E).survival(r, grid)
    out = su.evaluate_predictions(T, E, T, E, r, S, grid, auc_times=(12, 36), brier_times=(36,), tau=60)
    for key in ("c_index", "c_uno", "auc_12m", "auc_36m", "brier_36m", "ibs"):
        assert key in out and np.isfinite(out[key]), key
