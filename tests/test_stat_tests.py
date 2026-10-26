import numpy as np

import stat_tests as st
from survival_utils import concordance_index_fast


def _sim(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    T = np.round(rng.exponential(np.exp(-0.7 * x) * 40), 0) + 1
    C = rng.exponential(60, size=n)
    E = (T <= C).astype(int)
    return x, np.minimum(T, np.round(C) + 1), E


def test_weighted_bootstrap_equals_resampled_cindex():
    x, T, E = _sim(n=120)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(T), len(T))
    counts = np.bincount(idx, minlength=len(T)).astype(np.float32)[None, :]
    comp, conc = st._pair_matrices(T, E, x)
    fast = st._weighted_c(counts, comp, conc)[0]
    direct = concordance_index_fast(T[idx], x[idx], E[idx])
    assert abs(fast - direct) < 1e-6


def test_paired_bootstrap_detects_better_model():
    x, T, E = _sim(n=400)
    noise = np.random.default_rng(1).normal(size=len(x))
    res = st.paired_bootstrap_cindex(T, E, x, x + 2.0 * noise, n_boot=300)
    assert res["delta"] > 0 and res["delta_lo"] > 0 and res["p_bootstrap"] < 0.05
    same = st.paired_bootstrap_cindex(T, E, x, x, n_boot=100)
    assert abs(same["delta"]) < 1e-12


def test_corrected_ttest_and_holm():
    d = np.array([0.01, 0.005, 0.012, -0.002, 0.008] * 5)
    res = st.corrected_resampled_ttest(d, n_train=1522, n_test=381)
    naive = st.paired_ttest(d, np.zeros_like(d))
    assert res["p"] > naive[1]  # correction is conservative
    adj = st.holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
