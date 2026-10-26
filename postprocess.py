"""Generate aggregate tables and figures from locally reproduced results.

python postprocess.py --root outputs/reproduction
"""

from __future__ import annotations

import argparse
import json
import traceback
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURE_DPI
from experiment import (
    DERIVED_BLENDS,
    MAIN_MODELS,
    REVISION_DIR,
    TIME_GRID,
    ResultStore,
    ablation_variants,
    pam50_variants,
    prior_variants,
)
from hetero_data import FoldPreprocessor, PAM50_GENES, graph_statistics, load_metabric, pathway_gene_coverage
from stat_tests import bootstrap_ci, corrected_resampled_ttest, holm_adjust, paired_bootstrap_cindex, sig_mark
from survival_utils import (
    calibration_ici,
    calibration_slope,
    calibration_table,
    concordance_index_fast,
    kaplan_meier,
    logrank_multigroup,
    risk_group_hazard_ratios,
)

PROPOSED = "KG-MSPF"
N_BOOT = 2000
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})
MODEL_COLORS = {
    "KG-MSPF": "#C0392B", "Clinical-Cox": "#2E86C1", "Clinical+PAM50-Cox": "#5DADE2", "FullConcat-Cox": "#1ABC9C",
    "FullConcat-LASSO-Cox": "#48C9B0", "RSF": "#7D3C98", "DeepSurv": "#E67E22", "DeepHit": "#D4AC0D",
    "SurvTRACE": "#A04000", "Cox-PASNet": "#566573", "HomoGAT": "#AF7AC5",
}
BLEND_LABELS = {
    "neural_only": "Neural score only (no blend)",
    "anchors_only": "Linear anchors only (no neural score)",
    "no_clinical_cox": "Blend w/o Clinical-Cox",
    "no_subtype_cox": "Blend w/o subtype-Cox",
    "no_full_cox": "Blend w/o FullConcat-Cox",
}


# =====================================================================
# Loading helpers
# =====================================================================


class Results:
    def __init__(self, root: Path, n_patients: int) -> None:
        self.store = ResultStore(root)
        self.root = Path(root)
        self.n = n_patients
        self._cache: Dict[str, List[Dict]] = {}

    def records(self, exp: str) -> List[Dict]:
        if exp not in self._cache:
            self._cache[exp] = self.store.records(exp)
        return self._cache[exp]

    def has(self, exp: str) -> bool:
        return len(self.records(exp)) > 0

    def configs(self, exp: str) -> List[str]:
        seen = []
        for r in self.records(exp):
            if r["config"] not in seen:
                seen.append(r["config"])
        return seen

    def metrics_frame(self, exp: str) -> pd.DataFrame:
        rows = []
        for r in self.records(exp):
            row = {"config": r["config"], "split": r["split"], "repeat": r["repeat"], "fold": r["fold"],
                   "n_train": r.get("n_train"), "n_test": r.get("n_test")}
            row.update({k: (np.nan if v is None else v) for k, v in r["metrics"].items()})
            rows.append(row)
            for blend, m in (r.get("derived_blends") or {}).items():
                row_b = dict(row)
                row_b["config"] = f"{r['config']}[{blend}]"
                row_b.update({k: (np.nan if v is None else v) for k, v in m.items()})
                rows.append(row_b)
        return pd.DataFrame(rows)

    def arrays(self, rec: Dict) -> Dict[str, np.ndarray]:
        return dict(np.load(rec["_npz"], allow_pickle=False))

    def pooled(self, exp: str, config: str, repeat: int = 0, key: str = "risk", surv_key: str = "surv"):
        risk = np.full(self.n, np.nan)
        surv = None
        groups = np.full(self.n, -1)
        for rec in self.records(exp):
            if rec["config"] != config or rec["repeat"] != repeat:
                continue
            a = self.arrays(rec)
            if key not in a:
                continue
            idx = a["test_index"]
            risk[idx] = a[key]
            if surv_key in a:
                if surv is None:
                    surv = np.full((self.n, a[surv_key].shape[1]), np.nan)
                surv[idx] = a[surv_key]
            q_key = "train_risk_quantiles" + (key[len("risk"):] if key.startswith("risk") else "")
            if q_key in a:
                q = a[q_key]
                groups[idx] = np.digitize(a[key], q)
        return risk, surv, groups

    def single(self, exp: str, config: str) -> Optional[Dict[str, np.ndarray]]:
        for rec in self.records(exp):
            if rec["config"] == config:
                return self.arrays(rec)
        return None


def mean_sd(values) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "-"
    return f"{v.mean():.3f} ({v.std(ddof=1) if v.size > 1 else 0.0:.3f})"


def fmt_ci(point, lo, hi, digits=3) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =====================================================================
# Sections
# =====================================================================


def section_main(R: Results, table, out: Path, notes: List[str]) -> Optional[pd.DataFrame]:
    if not R.has("main"):
        return None
    mf = R.metrics_frame("main")
    base = mf[~mf["config"].str.contains(r"\[")]
    models = [m for m in MAIN_MODELS if m in set(base["config"])] + \
             [m for m in base["config"].unique() if m not in MAIN_MODELS]
    rows = []
    repeats = sorted(base["repeat"].unique())
    for m in models:
        g = base[base["config"] == m]
        pooled_c = []
        for r in repeats:
            risk, _, _ = R.pooled("main", m, r)
            ok = np.isfinite(risk)
            if ok.sum() > 0.95 * len(risk):
                pooled_c.append(concordance_index_fast(table.times[ok], risk[ok], table.events[ok]))
        row = {"model": m, "n_splits": len(g), "c_index": mean_sd(g["c_index"]), "c_uno": mean_sd(g["c_uno"])}
        for col in ("auc_12m", "auc_36m", "auc_60m", "auc_120m", "brier_60m", "ibs"):
            if col in g:
                row[col] = mean_sd(g[col])
        row["pooled_oof_c"] = f"{np.mean(pooled_c):.3f}" if pooled_c else "-"
        row["_c_mean"] = g["c_index"].mean()
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.drop(columns="_c_mean").to_csv(out / "tables" / "table_main_comparison.csv", index=False)
    notes.append(f"Main comparison: {len(repeats)} repeat(s) x {base['fold'].nunique()} folds.")
    _figure_main(base, models, out)
    return base


def _figure_main(base: pd.DataFrame, models: List[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    order = sorted(models, key=lambda m: base[base["config"] == m]["c_index"].mean())
    for ax, metric, label in ((axes[0], "c_index", "Harrell's C-index"), (axes[1], "ibs", "Integrated Brier score")):
        if metric not in base:
            continue
        for i, m in enumerate(order):
            v = base[base["config"] == m][metric].dropna()
            ax.errorbar(v.mean(), i, xerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="o",
                        color=MODEL_COLORS.get(m, "#555555"), ms=5, capsize=2)
            ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)),
                       s=4, color=MODEL_COLORS.get(m, "#555555"), alpha=0.35)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([("KG-MSPF (ours)" if m == PROPOSED else m) for m in order])
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_yticklabels([])
    times = [t for t in (12, 36, 60, 120) if f"auc_{t}m" in base]
    for m in models:
        g = base[base["config"] == m]
        means = [g[f"auc_{t}m"].mean() for t in times]
        axes[2].plot(times, means, marker="o", ms=3, lw=1.6 if m == PROPOSED else 0.9,
                     color=MODEL_COLORS.get(m, "#555555"), label="KG-MSPF (ours)" if m == PROPOSED else m)
    axes[2].set_xlabel("Time (months)")
    axes[2].set_ylabel("Time-dependent AUC (IPCW)")
    axes[2].set_xticks(times)
    axes[2].legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_model_comparison.png")


def section_significance(R: Results, table, base: pd.DataFrame, out: Path, notes: List[str]) -> None:
    if base is None or PROPOSED not in set(base["config"]):
        return
    piv_c = base.pivot_table(index="split", columns="config", values="c_index")
    piv_ibs = base.pivot_table(index="split", columns="config", values="ibs") if "ibs" in base else None
    n_train = float(base["n_train"].mean())
    n_test = float(base["n_test"].mean())
    ours_risk, _, _ = R.pooled("main", PROPOSED, 0)
    rows = []
    for m in piv_c.columns:
        if m == PROPOSED:
            continue
        d = (piv_c[PROPOSED] - piv_c[m]).dropna()
        res = corrected_resampled_ttest(d.to_numpy(), n_train, n_test)
        row = {"comparison": f"KG-MSPF vs {m}", "baseline": m, "mean_delta_c": res["mean"],
               "corrected_ci_lo": res.get("ci_lo"), "corrected_ci_hi": res.get("ci_hi"), "t": res["t"],
               "p_corrected": res["p"], "wins": f"{res.get('wins', 0)}/{res.get('n_splits', 0)}"}
        if piv_ibs is not None and m in piv_ibs:
            di = (piv_ibs[PROPOSED] - piv_ibs[m]).dropna()
            ri = corrected_resampled_ttest(di.to_numpy(), n_train, n_test)
            row.update({"mean_delta_ibs": ri["mean"], "p_ibs_corrected": ri["p"]})
        other, _, _ = R.pooled("main", m, 0)
        ok = np.isfinite(ours_risk) & np.isfinite(other)
        if ok.sum() > 0.95 * len(ok):
            b = paired_bootstrap_cindex(table.times[ok], table.events[ok], ours_risk[ok], other[ok], n_boot=N_BOOT)
            row.update({"oof_delta_c_r0": b["delta"], "boot_ci_lo": b["delta_lo"], "boot_ci_hi": b["delta_hi"],
                        "p_bootstrap": b["p_bootstrap"]})
        rows.append(row)
    sig = pd.DataFrame(rows)
    sig["p_holm"] = holm_adjust(sig["p_corrected"].fillna(1.0))
    sig["significance"] = sig["p_holm"].map(sig_mark)
    margin = 0.01
    sig["exploratory_ci_above_minus_0.01"] = sig["corrected_ci_lo"] > -margin
    sig.to_csv(out / "tables" / "table_significance.csv", index=False)
    notes.append("Significance: Nadeau-Bengio corrected repeated-CV t-test (Holm-adjusted across baselines) "
                 "and patient-level paired bootstrap on pooled OOF predictions of repeat 0 "
                 f"({N_BOOT} resamples, conditional on the fitted OOF models; training variability is omitted). "
                 "The 0.01 threshold is exploratory, not a prespecified non-inferiority or equivalence claim.")
    # forest plot
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(sig) + 1.0))
    s = sig.sort_values("mean_delta_c")
    y = np.arange(len(s))
    ax.errorbar(s["mean_delta_c"], y, xerr=[s["mean_delta_c"] - s["corrected_ci_lo"], s["corrected_ci_hi"] - s["mean_delta_c"]],
                fmt="o", color="#C0392B", capsize=2, ms=4, label="repeated-CV (corrected 95% CI)")
    if "boot_ci_lo" in s:
        ax.errorbar(s["oof_delta_c_r0"], y + 0.25, xerr=[s["oof_delta_c_r0"] - s["boot_ci_lo"], s["boot_ci_hi"] - s["oof_delta_c_r0"]],
                    fmt="s", color="#2E86C1", capsize=2, ms=3, label="pooled OOF bootstrap 95% CI")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-margin, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y + 0.12)
    ax.set_yticklabels(s["baseline"])
    ax.set_xlabel("Delta C-index (KG-MSPF minus baseline)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_delta_c_forest.png")


def _variant_table(R: Results, table, exp: str, ref: str, labels: Dict[str, str], out_name: str, out: Path,
                   group_fn: Optional[Callable[[str], str]] = None) -> Optional[pd.DataFrame]:
    if not R.has(exp):
        return None
    mf = R.metrics_frame(exp)
    if ref not in set(mf["config"]):
        return None
    repeats = sorted(mf["repeat"].unique())
    rows = []
    for cfg in R.configs(exp):
        g = mf[mf["config"] == cfg]
        merged = g.merge(mf[mf["config"] == ref][["split", "c_index"]], on="split", suffixes=("", "_ref"))
        delta = merged["c_index"] - merged["c_index_ref"]
        row = {"config": cfg, "description": labels.get(cfg, cfg), "n_splits": len(g), "c_index": mean_sd(g["c_index"]),
               "c_mean": g["c_index"].mean(), "delta_c_vs_ref": delta.mean(),
               "ibs": mean_sd(g["ibs"]) if "ibs" in g else "-", "folds_better_than_ref": int((delta > 0).sum())}
        # Primary comparison accounts for overlapping training folds. A pooled
        # patient bootstrap alone omits variation from fitting the models.
        train_n = pd.to_numeric(g.get("n_train", pd.Series(dtype=float)), errors="coerce").mean()
        test_n = pd.to_numeric(g.get("n_test", pd.Series(dtype=float)), errors="coerce").mean()
        if cfg != ref and np.isfinite(train_n) and train_n > 0 and np.isfinite(test_n):
            corrected = corrected_resampled_ttest(delta, train_n, test_n)
            row.update({"p_corrected": corrected["p"], "corrected_ci_lo": corrected.get("ci_lo"),
                        "corrected_ci_hi": corrected.get("ci_hi")})
        if cfg != ref:
            a, _, _ = R.pooled(exp, cfg, repeats[0])
            b, _, _ = R.pooled(exp, ref, repeats[0])
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 0.95 * len(ok):
                bs = paired_bootstrap_cindex(table.times[ok], table.events[ok], a[ok], b[ok], n_boot=N_BOOT)
                row.update({"oof_delta": bs["delta"], "oof_delta_lo": bs["delta_lo"], "oof_delta_hi": bs["delta_hi"],
                            "p_bootstrap": bs["p_bootstrap"]})
        if group_fn is not None:
            row["family"] = group_fn(cfg)
        rows.append(row)
    tab = pd.DataFrame(rows)
    if "p_corrected" in tab:
        mask = tab["config"] != ref
        tab.loc[mask, "p_holm"] = holm_adjust(tab.loc[mask, "p_corrected"].fillna(1.0))
    tab.to_csv(out / "tables" / out_name, index=False)
    return tab


def section_ablation(R: Results, table, out: Path, notes: List[str]) -> None:
    labels = {k: v.description for k, v in ablation_variants().items()}
    tab = _variant_table(R, table, "ablation", "full", labels, "table_ablation_retrained.csv", out)
    # derived blends from the main experiment
    rows = []
    if R.has("main"):
        mf = R.metrics_frame("main")
        ref = mf[mf["config"] == PROPOSED][["split", "c_index", "ibs"]]
        for blend in DERIVED_BLENDS:
            g = mf[mf["config"] == f"{PROPOSED}[{blend}]"]
            if g.empty:
                continue
            merged = g.merge(ref, on="split", suffixes=("", "_ref"))
            a, _, _ = R.pooled("main", PROPOSED, 0, key=f"risk__{blend}", surv_key=f"surv__{blend}")
            b, _, _ = R.pooled("main", PROPOSED, 0)
            row = {"config": blend, "description": BLEND_LABELS.get(blend, blend), "n_splits": len(g),
                   "c_index": mean_sd(g["c_index"]), "c_mean": g["c_index"].mean(),
                   "delta_c_vs_ref": (merged["c_index"] - merged["c_index_ref"]).mean(),
                   "ibs": mean_sd(g["ibs"]), "folds_better_than_ref": int(((merged["c_index"] - merged["c_index_ref"]) > 0).sum())}
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 0.95 * len(ok):
                bs = paired_bootstrap_cindex(table.times[ok], table.events[ok], a[ok], b[ok], n_boot=N_BOOT)
                row.update({"oof_delta": bs["delta"], "oof_delta_lo": bs["delta_lo"], "oof_delta_hi": bs["delta_hi"],
                            "p_bootstrap": bs["p_bootstrap"]})
            rows.append(row)
    blend_tab = pd.DataFrame(rows)
    if not blend_tab.empty:
        blend_tab.to_csv(out / "tables" / "table_ablation_blend_components.csv", index=False)
    if tab is None and blend_tab.empty:
        return
    parts = [t for t in (tab, blend_tab) if t is not None and not t.empty]
    allt = pd.concat(parts, ignore_index=True)
    allt = allt[allt["config"] != "full"]
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(allt) + 1.0))
    allt = allt.sort_values("delta_c_vs_ref")
    y = np.arange(len(allt))
    has_ci = "oof_delta_lo" in allt
    ax.barh(y, allt["delta_c_vs_ref"], color=["#C0392B" if d < 0 else "#2E86C1" for d in allt["delta_c_vs_ref"]],
            alpha=0.75, height=0.6, label="mean fold-wise delta")
    if has_ci:
        ok = allt["oof_delta_lo"].notna()
        ax.errorbar(allt.loc[ok, "oof_delta"], y[ok.to_numpy()],
                    xerr=[allt.loc[ok, "oof_delta"] - allt.loc[ok, "oof_delta_lo"],
                          allt.loc[ok, "oof_delta_hi"] - allt.loc[ok, "oof_delta"]],
                    fmt="o", color="black", ms=3, capsize=2, label="pooled OOF bootstrap 95% CI")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(allt["description"])
    ax.set_xlabel("Delta C-index versus full KG-MSPF")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_ablation.png")
    notes.append("Ablations: re-trained variants on repeat-0 folds plus blend-component removals evaluated on all "
                 "main-experiment splits (derived from the same fitted models).")


def _prior_family(cfg: str) -> str:
    for fam in ("perturb", "permute", "rewired"):
        if cfg.startswith(fam):
            return fam
    return cfg


def section_prior(R: Results, table, out: Path, notes: List[str]) -> None:
    labels = {k: v.description for k, v in prior_variants(10).items()}
    tab = _variant_table(R, table, "prior", "default", labels, "table_prior_sensitivity_by_draw.csv", out, _prior_family)
    if tab is None:
        return
    fam = tab.groupby("family").agg(n_configs=("config", "count"), c_mean=("c_mean", "mean"),
                                    c_sd_across_draws=("c_mean", "std"), delta_mean=("delta_c_vs_ref", "mean"),
                                    delta_min=("delta_c_vs_ref", "min"), delta_max=("delta_c_vs_ref", "max")).reset_index()
    fam.to_csv(out / "tables" / "table_prior_sensitivity.csv", index=False)
    order = ["default", "uniform", "inverted", "perturb", "permute", "rewired"]
    fam_labels = {"default": "Default tiers", "uniform": "Uniform (no prior term)", "inverted": "Inverted tiers",
                  "perturb": "Perturbed (log-normal, sd 0.3)", "permute": "Permuted values",
                  "rewired": "Rewired knowledge edges"}
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for i, f in enumerate([o for o in order if o in set(tab["family"])]):
        v = tab[tab["family"] == f]["c_mean"]
        ax.scatter(np.full(len(v), i), v, color="#2E86C1" if f != "default" else "#C0392B", s=22, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1)
    ref = tab[tab["config"] == "default"]["c_mean"]
    if len(ref):
        ax.axhline(ref.iloc[0], color="#C0392B", lw=0.8, ls="--")
    fams = [o for o in order if o in set(tab["family"])]
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([fam_labels[f] for f in fams], rotation=25, ha="right")
    ax.set_ylabel("Mean C-index (5 folds)")
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_prior_sensitivity.png")
    notes.append("Prior sensitivity: each random setting repeated with independent draws; family-level mean and range reported.")


def section_pam50(R: Results, table, out: Path, notes: List[str]) -> None:
    labels = {k: v.description for k, v in pam50_variants().items()}
    tab = _variant_table(R, table, "pam50", "pam50_full", labels, "table_pam50.csv", out)
    # overlap of PAM50 genes with the expression panel and with selected genes
    rows = []
    for rec in R.records("main"):
        if rec["config"] == PROPOSED and rec.get("preprocessor"):
            sel = rec["preprocessor"].get("pam50_genes_selected", [])
            rows.append({"split": rec["split"], "n_pam50_selected": len(sel), "genes": ", ".join(sel)})
    if rows:
        pd.DataFrame(rows).to_csv(out / "tables" / "table_pam50_gene_overlap_by_split.csv", index=False)
    panel = set(table.expression.columns)
    in_panel = [g for g in PAM50_GENES if g in panel]
    ct = pd.crosstab(pd.Series(table.pam50, name="PAM50"), pd.Series(table.ihc_subtype, name="IHC surrogate"))
    ct.to_csv(out / "tables" / "table_pam50_vs_ihc_crosstab.csv")
    chi = ct.to_numpy(float)
    expected = chi.sum(1, keepdims=True) * chi.sum(0, keepdims=True) / chi.sum()
    chi2 = float(((chi - expected) ** 2 / np.where(expected > 0, expected, 1)).sum())
    cramer = float(np.sqrt(chi2 / (chi.sum() * (min(chi.shape) - 1))))
    notes.append(f"PAM50 genes present in the METABRIC expression panel: {len(in_panel)}/50 ({', '.join(in_panel)}). "
                 f"Cramer's V between PAM50 and IHC surrogate subtype: {cramer:.2f}.")


def _km_figure(ax, times, events, groups, title, labels=("Low", "Intermediate", "High")):
    colors = ["#27AE60", "#F39C12", "#C0392B"]
    for g in range(3):
        m = groups == g
        if m.sum() == 0:
            continue
        km = kaplan_meier(times[m], events[m])
        x = np.concatenate([[0], km.x])
        y = np.concatenate([[1], km.y])
        ax.step(x, y, where="post", color=colors[g], lw=1.4,
                label=f"{labels[g]} (n={int(m.sum())}, events={int(events[m].sum())})")
    chi2, p, _ = logrank_multigroup(times[groups >= 0], events[groups >= 0], groups[groups >= 0])
    ax.text(0.97, 0.97, f"log-rank p = {p:.1e}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    ax.set_title(title)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    return p


def _km_rows(name, times, events, groups, rule):
    hr = risk_group_hazard_ratios(times[groups >= 0], events[groups >= 0], groups[groups >= 0], reference=0)
    chi2, p, _ = logrank_multigroup(times[groups >= 0], events[groups >= 0], groups[groups >= 0])
    rows = []
    for g in range(3):
        m = groups == g
        row = {"analysis": name, "cut_points": rule, "group": ["low", "intermediate", "high"][g], "n": int(m.sum()),
               "events": int(events[m].sum()), "logrank_p": p}
        h = hr[hr["group"] == g]
        if len(h):
            row.update({"hr_vs_low": h["hr"].iloc[0], "hr_lo": h["hr_lo"].iloc[0], "hr_hi": h["hr_hi"].iloc[0]})
        rows.append(row)
    return rows


def section_holdout(R: Results, table, out: Path, notes: List[str]) -> None:
    if not R.has("holdout"):
        return
    recs = {r["config"]: r for r in R.records("holdout")}
    ours = R.arrays(recs[PROPOSED]) if PROPOSED in recs else None
    rows = []
    for m in [x for x in MAIN_MODELS if x in recs] + [x for x in recs if x not in MAIN_MODELS]:
        a = R.arrays(recs[m])
        idx = a["test_index"]
        t, e = table.times[idx], table.events[idx]
        met = recs[m]["metrics"]
        _, lo, hi = bootstrap_ci(t, a["risk"], e, n_boot=N_BOOT)
        row = {"model": m, "n": len(idx), "events": int(e.sum()), "c_index": fmt_ci(met["c_index"], lo, hi),
               "c_uno": met.get("c_uno"), "auc_36m": met.get("auc_36m"), "auc_60m": met.get("auc_60m"),
               "auc_120m": met.get("auc_120m"), "brier_60m": met.get("brier_60m"), "ibs": met.get("ibs")}
        if ours is not None and m != PROPOSED:
            bs = paired_bootstrap_cindex(t, e, ours["risk"], a["risk"], n_boot=N_BOOT)
            row.update({"delta_c_kgmspf_minus_model": bs["delta"], "delta_lo": bs["delta_lo"],
                        "delta_hi": bs["delta_hi"], "p_bootstrap": bs["p_bootstrap"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "tables" / "table_holdout.csv", index=False)
    if ours is None:
        return
    idx = ours["test_index"]
    t, e = table.times[idx], table.events[idx]
    fixed = np.digitize(ours["risk"], ours["train_risk_quantiles"])
    within = np.digitize(ours["risk"], np.percentile(ours["risk"], [100 / 3, 200 / 3]))
    km_rows = _km_rows("holdout", t, e, fixed, "training-cohort tertiles (fixed)")
    km_rows += _km_rows("holdout", t, e, within, "within-test tertiles (descriptive)")
    # OOF repeat 0 with fold-specific training cut points
    risk, _, groups = R.pooled("main", PROPOSED, 0)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    if np.all(groups >= 0):
        km_rows += _km_rows("5-fold OOF (repeat 0)", table.times, table.events, groups, "fold-specific training tertiles")
        _km_figure(axes[0], table.times, table.events, groups, "Five-fold OOF (training-fold cut-points)")
    _km_figure(axes[1], t, e, fixed, "Cohort hold-out (cut-points from cohorts 1-3)")
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_km_risk_groups.png")
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    _km_figure(ax, t, e, within, "Cohort hold-out (within-test tertiles)")
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_km_holdout_within_test.png")
    pd.DataFrame(km_rows).to_csv(out / "tables" / "table_km_risk_groups.csv", index=False)
    notes.append("Hold-out KM: tertile cut-points are taken from the training-cohort risk distribution and applied "
                 "unchanged to cohorts 4-5; within-test tertiles are reported only as a supplementary view.")


def _calibration_rows(name, times, events, surv, grid, horizons):
    rows = []
    curves = {}
    for h in horizons:
        k = int(np.argmin(np.abs(grid - h)))
        s = surv[:, k]
        ok = np.isfinite(s)
        ici = calibration_ici(times[ok], events[ok], s[ok], grid[k])
        lp = np.log(-np.log(np.clip(s[ok], 1e-6, 1 - 1e-6)))
        sl = calibration_slope(times[ok], events[ok], lp)
        rows.append({"analysis": name, "horizon_months": h, **ici, **sl})
        curves[h] = calibration_table(times[ok], events[ok], s[ok], grid[k])
    return rows, curves


def section_calibration(R: Results, table, out: Path, notes: List[str]) -> None:
    rows = []
    curve_store = {}
    horizons = (36, 60, 120)
    models = [m for m in MAIN_MODELS if m in R.configs("main")] if R.has("main") else []
    for m in models:
        _, surv, _ = R.pooled("main", m, 0)
        if surv is None:
            continue
        r, c = _calibration_rows(f"OOF r0 | {m}", table.times, table.events, surv, TIME_GRID, horizons)
        for x in r:
            x["model"] = m
        rows += r
        curve_store[("OOF", m)] = c
    if R.has("holdout"):
        for rec in R.records("holdout"):
            a = R.arrays(rec)
            idx = a["test_index"]
            r, c = _calibration_rows(f"holdout | {rec['config']}", table.times[idx], table.events[idx], a["surv"],
                                     TIME_GRID, horizons)
            for x in r:
                x["model"] = rec["config"]
            rows += r
            curve_store[("holdout", rec["config"])] = c
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out / "tables" / "table_calibration.csv", index=False)
    show = [m for m in (PROPOSED, "Clinical-Cox", "FullConcat-Cox", "RSF", "DeepSurv") if ("OOF", m) in curve_store]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.6))
    for row_i, setting in enumerate(("OOF", "holdout")):
        for col_i, h in enumerate(horizons):
            ax = axes[row_i, col_i]
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
            for m in show:
                c = curve_store.get((setting, m), {}).get(h)
                if c is None or c.empty:
                    continue
                ax.errorbar(c["predicted"], c["observed"], yerr=[c["observed"] - c["observed_lo"], c["observed_hi"] - c["observed"]],
                            fmt="o-", ms=3, lw=1.4 if m == PROPOSED else 0.8, capsize=1.5,
                            color=MODEL_COLORS.get(m, "#555555"), label="KG-MSPF (ours)" if m == PROPOSED else m)
            ax.set_title(f"{'Five-fold OOF' if setting == 'OOF' else 'Cohort hold-out'}: {h} months")
            ax.set_xlabel("Predicted probability of death")
            ax.set_ylabel("Observed (Kaplan-Meier)")
            vals = [0.1]
            for m in show:
                c = curve_store.get((setting, m), {}).get(h)
                if c is not None and not c.empty:
                    vals += list(c["predicted"]) + list(c["observed_hi"])
            lim = min(1.0, 1.1 * max(vals))
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_calibration.png")
    notes.append("Calibration: decile curves with Kaplan-Meier observed risk; ICI/E50/E90 (Austin et al. 2020) and "
                 "calibration slope of the complementary log-log predicted risk; IBS over 6-180 months (IPCW).")


def section_interpretability(R: Results, out: Path, notes: List[str]) -> None:
    recs = [r for r in R.records("main") if r["config"] == PROPOSED and r.get("interpretability")]
    if not recs:
        return
    ko, att, top, branch, gates, pw, clin, blends, scales = [], [], [], [], [], [], [], [], []
    for r in recs:
        it = r["interpretability"]
        for row in it.get("knockout", []) or []:
            ko.append({"split": r["split"], **row})
        for rel, st in (it.get("attention", {}) or {}).get("relations", {}).items():
            att.append({"split": r["split"], "relation": rel, **st})
        for row in (it.get("attention", {}) or {}).get("top_edges", []):
            top.append({"split": r["split"], **row})
        if it.get("branch_contributions"):
            branch.append({"split": r["split"], **it["branch_contributions"]})
        s = r.get("summary", {})
        for layer, d in (s.get("relation_gates") or {}).items():
            for rel, v in d.items():
                gates.append({"split": r["split"], "layer": layer, "relation": rel, "value": v})
        for p_name, v in (s.get("pathway_coefficients") or {}).items():
            pw.append({"split": r["split"], "pathway": p_name, "coef": v})
        for c_name, v in (s.get("clinical_coefficients") or {}).items():
            clin.append({"split": r["split"], "feature": c_name, "coef": v})
        w = (s.get("blends") or {}).get("main", {}).get("weights", {})
        if w:
            blends.append({"split": r["split"], **w})
        if s.get("branch_scales"):
            scales.append({"split": r["split"], **s["branch_scales"]})
    T = out / "tables"
    if ko:
        kdf = pd.DataFrame(ko)
        agg = kdf.groupby(["kind", "name"]).agg(delta_c_nn_mean=("delta_c_nn", "mean"), delta_c_nn_sd=("delta_c_nn", "std"),
                                                delta_c_kg_mean=("delta_c_kg", "mean"), delta_c_kg_sd=("delta_c_kg", "std"),
                                                frac_splits_worse=("delta_c_nn", lambda v: float(np.mean(np.asarray(v) < 0))),
                                                mean_abs_risk_change=("mean_abs_change_nn", "mean")).reset_index()
        agg.to_csv(T / "table_knockout.csv", index=False)
    if att:
        adf = pd.DataFrame(att)
        adf.groupby("relation").agg(informative=("informative", "first"), mean_in_degree=("mean_in_degree", "mean"),
                                    kl_from_uniform=("kl_from_uniform", "mean"), kl_sd=("kl_from_uniform", "std"),
                                    normalized_entropy=("normalized_entropy", "mean")).reset_index().to_csv(
            T / "table_attention_concentration.csv", index=False)
    if top:
        tdf = pd.DataFrame(top)
        tdf.groupby(["relation", "source", "destination"]).agg(n_splits_in_top10=("split", "count"),
                                                               mean_ratio_to_uniform=("ratio_to_uniform", "mean")).reset_index().sort_values(
            ["relation", "n_splits_in_top10"], ascending=[True, False]).to_csv(T / "table_top_attention_edges.csv", index=False)
    if branch:
        pd.DataFrame(branch).describe().T.to_csv(T / "table_branch_contributions.csv")
    if gates:
        gdf = pd.DataFrame(gates)
        gdf.groupby(["layer", "relation"]).agg(mean=("value", "mean"), sd=("value", "std")).reset_index().to_csv(
            T / "table_relation_gates.csv", index=False)
    if pw:
        pdf = pd.DataFrame(pw)
        pdf.groupby("pathway").agg(mean=("coef", "mean"), sd=("coef", "std"),
                                   frac_positive=("coef", lambda v: float(np.mean(np.asarray(v) > 0)))).reset_index().to_csv(
            T / "table_pathway_coefficients.csv", index=False)
    if clin:
        cdf = pd.DataFrame(clin)
        cdf.groupby("feature").agg(mean=("coef", "mean"), sd=("coef", "std")).reset_index().to_csv(
            T / "table_clinical_head_coefficients.csv", index=False)
    if blends:
        bdf = pd.DataFrame(blends)
        bdf.to_csv(T / "table_blend_weights_by_split.csv", index=False)
        bdf.drop(columns="split").describe().T.to_csv(T / "table_blend_weights_summary.csv")
    if scales:
        pd.DataFrame(scales).describe().T.to_csv(T / "table_branch_gates.csv")

    # Figure: relation analysis
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    if ko:
        rel = agg[agg["kind"] == "relation"].sort_values("delta_c_nn_mean")
        axes[0].barh(rel["name"], rel["delta_c_nn_mean"], xerr=rel["delta_c_nn_sd"], color="#2E86C1", alpha=0.8, capsize=2)
        axes[0].axvline(0, color="black", lw=0.8)
        axes[0].set_xlabel("Delta C of neural score when relation removed")
        axes[0].set_title("(a) Relation knockout (held-out folds)")
    if att:
        a2 = adf[adf["informative"] == True].groupby("relation")["kl_from_uniform"].agg(["mean", "std"]).sort_values("mean")  # noqa: E712
        axes[1].barh(a2.index, a2["mean"], xerr=a2["std"], color="#7D3C98", alpha=0.8, capsize=2)
        axes[1].set_xlabel("KL divergence from uniform attention (nats)")
        axes[1].set_title("(b) Attention concentration")
    if ko:
        nt = agg[agg["kind"].isin(["node_type", "pathway"])].copy()
        nt["label"] = np.where(nt["kind"] == "node_type", "[type] " + nt["name"], nt["name"])
        nt = nt.sort_values("delta_c_nn_mean")
        axes[2].barh(nt["label"], nt["delta_c_nn_mean"], xerr=nt["delta_c_nn_sd"],
                     color=np.where(nt["kind"] == "node_type", "#C0392B", "#16A085"), alpha=0.8, capsize=2)
        axes[2].axvline(0, color="black", lw=0.8)
        axes[2].set_xlabel("Delta C of neural score when nodes zeroed")
        axes[2].set_title("(c) Node-type and pathway knockout")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_interpretability.png")
    if blends:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        cols = [c for c in ("clinical_cox", "subtype_cox", "full_cox", "heterogat") if c in bdf]
        names = {"clinical_cox": "Clinical-Cox", "subtype_cox": "Subtype-Cox", "full_cox": "FullConcat-Cox",
                 "heterogat": "Neural"}
        ax.boxplot([bdf[c].to_numpy() for c in cols])
        ax.set_xticks(range(1, len(cols) + 1))
        ax.set_xticklabels([names[c] for c in cols])
        for i, c in enumerate(cols, start=1):
            ax.scatter(np.full(len(bdf), i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(bdf)), bdf[c], s=6, alpha=0.5)
        ax.set_ylabel("Blend weight")
        ax.set_title(f"Validation-selected blend weights ({len(bdf)} splits)")
        fig.tight_layout()
        savefig(fig, out / "figures" / "fig_blend_weights.png")
    notes.append("Interpretability: mean attention per relation is not reported because it equals 1/mean in-degree "
                 "by construction; instead relation gates, KL divergence of attention from uniform (destinations "
                 "with in-degree > 1), relation/node-type/pathway knockouts on held-out folds, branch variance shares "
                 "and pathway coefficients are summarised across all splits.")


def section_subgroups(R: Results, table, out: Path, notes: List[str]) -> None:
    if not R.has("main"):
        return
    ours, _, _ = R.pooled("main", PROPOSED, 0)
    clin, _, _ = R.pooled("main", "Clinical-Cox", 0)
    if not np.isfinite(ours).all():
        return
    rows = []
    for s in ("Luminal_A", "Luminal_B", "HER2_enriched", "Basal_like", "Other"):
        m = table.pam50 == s
        if m.sum() < 30:
            continue
        row = {"subtype": s, "n": int(m.sum()), "events": int(table.events[m].sum())}
        _, lo, hi = bootstrap_ci(table.times[m], ours[m], table.events[m], n_boot=1000)
        row["kgmspf_c"] = fmt_ci(concordance_index_fast(table.times[m], ours[m], table.events[m]), lo, hi)
        if np.isfinite(clin).all():
            bs = paired_bootstrap_cindex(table.times[m], table.events[m], ours[m], clin[m], n_boot=1000)
            row.update({"clinical_cox_c": bs["c_b"], "delta": bs["delta"], "delta_lo": bs["delta_lo"],
                        "delta_hi": bs["delta_hi"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "table_subgroups_pam50.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    y = np.arange(len(df))
    kc = [float(x.split(" ")[0]) for x in df["kgmspf_c"]]
    ax.scatter(kc, y, color="#C0392B", label="KG-MSPF", zorder=3, s=df["n"] / 8)
    if "clinical_cox_c" in df:
        ax.scatter(df["clinical_cox_c"], y, color="#2E86C1", marker="s", label="Clinical-Cox", zorder=3, s=18)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s.replace('_', ' ')} (n={n})" for s, n in zip(df["subtype"], df["n"])])
    ax.set_xlabel("Within-stratum C-index (pooled OOF, repeat 0)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    savefig(fig, out / "figures" / "fig_subgroup.png")


def section_external(R: Results, out: Path, notes: List[str]) -> None:
    exps = sorted(p.name for p in R.root.glob("external_*") if p.is_dir())
    rows = []
    for exp in exps:
        recs = {r["config"]: r for r in R.records(exp)}
        if not recs:
            continue
        plan_path = R.root / exp / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        harmonized = None
        ours = R.arrays(recs[PROPOSED]) if PROPOSED in recs else None
        # event data are stored implicitly: reload from the harmonised file is not needed, use metrics + arrays
        for m, rec in recs.items():
            a = R.arrays(rec)
            met = rec["metrics"]
            row = {"cohort": exp.replace("external_", ""), "model": m, "n": rec.get("n_test"),
                   "c_index": met.get("c_index"), "c_uno": met.get("c_uno"), "ibs": met.get("ibs")}
            for k, v in met.items():
                if k.startswith("auc_") or k.startswith("brier_"):
                    row[k] = v
            if "times" in a and "events" in a:
                t, e = a["times"], a["events"]
                _, lo, hi = bootstrap_ci(t, a["risk"], e, n_boot=N_BOOT)
                row["c_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                if ours is not None and m != PROPOSED:
                    bs = paired_bootstrap_cindex(t, e, ours["risk"], a["risk"], n_boot=N_BOOT)
                    row.update({"delta_c_kgmspf_minus_model": bs["delta"], "delta_lo": bs["delta_lo"],
                                "delta_hi": bs["delta_hi"], "p_bootstrap": bs["p_bootstrap"]})
            row.update({"tau": plan.get("tau"), "clinical_features": ";".join(plan.get("clinical_features", []))})
            rows.append(row)
        if ours is not None and "times" in ours:
            g = np.digitize(ours["risk"], ours["train_risk_quantiles"])
            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            _km_figure(ax, ours["times"], ours["events"], g, f"{exp.replace('external_', '').upper()}: METABRIC cut-points")
            fig.tight_layout()
            savefig(fig, out / "figures" / f"fig_km_{exp}.png")
    if rows:
        pd.DataFrame(rows).to_csv(out / "tables" / "table_external_validation.csv", index=False)
        notes.append("External validation: models trained on all METABRIC patients with the transportable feature set "
                     "and applied unchanged (expression re-standardised within the external cohort).")


def section_data_tables(table, out: Path) -> None:
    pathway_gene_coverage(table.expression.columns).to_csv(out / "tables" / "table1_pathway_gene_coverage.csv", index=False)
    pre = FoldPreprocessor().fit(table, np.arange(len(table)))
    graph_statistics(pre.schema).to_csv(out / "tables" / "table2_graph_statistics.csv", index=False)
    meta = pre.metadata()
    (out / "tables" / "full_cohort_preprocessor.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mut = table.mutation
    if mut is not None:
        freq = mut.mean().sort_values(ascending=False)
        freq.head(30).to_csv(out / "tables" / "mutation_frequency_top30.csv", header=["frequency"])


def export_predictions(R: Results, table, out: Path) -> None:
    if R.has("main"):
        df = pd.DataFrame({"patient_id": table.patient_ids, "time": table.times, "event": table.events,
                           "pam50": table.pam50, "cohort": table.cohorts})
        for m in R.configs("main"):
            risk, surv, _ = R.pooled("main", m, 0)
            df[f"risk|{m}"] = risk
            if surv is not None:
                k = int(np.argmin(np.abs(TIME_GRID - 60)))
                df[f"S60|{m}"] = surv[:, k]
        df.to_csv(out / "predictions_oof_repeat0.csv", index=False)
    if R.has("holdout"):
        recs = R.records("holdout")
        first = R.arrays(recs[0])
        idx = first["test_index"]
        df = pd.DataFrame({"patient_id": table.patient_ids[idx], "time": table.times[idx], "event": table.events[idx]})
        for rec in recs:
            df[f"risk|{rec['config']}"] = R.arrays(rec)["risk"]
        df.to_csv(out / "predictions_holdout.csv", index=False)


def run_all(root: Path = REVISION_DIR, smoke: bool = False) -> Path:
    root = Path(root)
    table = load_metabric()
    R = Results(root, len(table))
    out = root / "manuscript"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    errors: List[str] = []
    base = None

    def guarded(name, fn, *args):
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001
            errors.append(f"--- {name}\n{traceback.format_exc()}")
            print(f"  section {name} failed (see postprocess_errors.log)")
            return None

    guarded("data_tables", section_data_tables, table, out)
    base = guarded("main", section_main, R, table, out, notes)
    guarded("significance", section_significance, R, table, base, out, notes)
    guarded("ablation", section_ablation, R, table, out, notes)
    guarded("prior", section_prior, R, table, out, notes)
    guarded("pam50", section_pam50, R, table, out, notes)
    guarded("holdout", section_holdout, R, table, out, notes)
    guarded("calibration", section_calibration, R, table, out, notes)
    guarded("interpretability", section_interpretability, R, out, notes)
    guarded("subgroups", section_subgroups, R, table, out, notes)
    guarded("external", section_external, R, out, notes)

    def _fig5():
        import plot_prediction_graph
        plot_prediction_graph.plot_prediction_graph(out / "figures" / "fig_prediction_graph.png")

    guarded("figure5", _fig5)

    status_text = 'Software smoke check only' if smoke else 'Aggregate results from the supplied output directory.'
    notes.append('SurvTRACE denotes a SurvTRACE-style single-event reimplementation; Cox-PASNet denotes a pathway-mask reimplementation without sparse coding. See model_implementations.json.')
    summary = ["# KG-MSPF results", "",status_text,''] + [f"- {n}" for n in notes] + [""]
    for csv in sorted((out / "tables").glob("*.csv")):
        try:
            df = pd.read_csv(csv)
            summary += [f"## {csv.stem}", "", df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(), ""]
        except Exception:  # noqa: BLE001
            try:
                summary += [f"## {csv.stem}", "", "```", pd.read_csv(csv).to_string(), "```", ""]
            except Exception:
                pass
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    if errors:
        (root / "postprocess_errors.log").write_text("\n".join(errors), encoding="utf-8")
    print(f'Post-processing finished: {out}')
    if errors and not smoke:
        print(f"{len(errors)} section(s) failed; see {root / 'postprocess_errors.log'}")
    if errors:
        raise RuntimeError("post-processing sections failed:\n" + "\n".join(errors))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(REVISION_DIR))
    args = p.parse_args()
    run_all(Path(args.root))


if __name__ == "__main__":
    main()
