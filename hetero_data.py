"""Cohort tables, training-fold preprocessing and patient-specific heterogeneous graphs.

Mutation strings are parsed as mutated, wild type or missing. Overall-survival events are derived from death_from_cancer. Feature selection and scaling use training patients only."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "METABRIC_RNA_Mutation.csv"

# "os": overall survival (any death = event); "dss": disease-specific survival
SURVIVAL_ENDPOINT = "os"

PAM50_TO_CANONICAL: Mapping[str, str] = {
    "LUMA": "Luminal_A",
    "LUMB": "Luminal_B",
    "HER2": "HER2_enriched",
    "BASAL": "Basal_like",
    "NORMAL": "Other",
    "CLAUDIN-LOW": "Other",
    "NC": "Other",
    "LUMINAL_A": "Luminal_A",
    "LUMINAL_B": "Luminal_B",
    "HER2_ENRICHED": "HER2_enriched",
    "BASAL_LIKE": "Basal_like",
    "OTHER": "Other",
}

SUBTYPE_NAMES: Tuple[str, ...] = (
    "Luminal_A",
    "Luminal_B",
    "HER2_enriched",
    "Basal_like",
    "Other",
)

PAM50_GENES: Tuple[str, ...] = (
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
)

# Pathway-gene map (exact gene symbols only).  Only genes present in the
# expression panel of a cohort enter the graph; see ``pathway_gene_coverage``.
PATHWAY_GENE_MAP: Mapping[str, Tuple[str, ...]] = {
    "ER_signaling": ("ESR1", "PGR", "FOXA1", "GATA3", "XBP1", "TBX3", "BCL2"),
    "HER2_ERBB": ("ERBB2", "ERBB3", "GRB7", "EGFR", "SHC1", "GRB2"),
    "PI3K_AKT_mTOR": ("PIK3CA", "PIK3R1", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR", "TSC1", "TSC2"),
    "p53_DNA_repair": ("TP53", "BRCA1", "BRCA2", "ATM", "CHEK2", "MDM2", "PALB2", "RAD50", "RAD51C", "RAD51D"),
    "Cell_cycle": ("CCND1", "CCNE1", "CDK1", "CDK2", "CDK4", "CDK6", "RB1", "CDKN2A", "CDKN1A", "E2F1", "E2F3"),
    "RAS_MAPK": ("KRAS", "NRAS", "BRAF", "MAP2K1", "MAP2K4", "MAP3K1", "MAPK1", "MAPK3"),
    "WNT": ("CTNNB1", "APC", "AXIN1", "WNT1", "WNT3A"),
    "Chromatin": ("KMT2C", "KMT2D", "ARID1A", "KDM6A", "SETD2", "NCOR1"),
    "Notch": ("NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4"),
    "Apoptosis": ("BCL2", "BAX", "CASP8", "BIRC3", "BCL2L1"),
    "Growth_factor": ("EGFR", "FGFR1", "FGFR2", "IGF1R", "MET", "VEGFA", "ERBB2"),
    "Immune": ("CD274", "PDCD1", "CTLA4", "JAK1", "JAK2", "STAT1", "STAT3", "CD8A", "CD4"),
}

CLINICAL_FEATURES: Tuple[str, ...] = (
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_positive",
    "grade",
    "stage",
    "er_status",
    "pr_status",
    "her2_status",
    "surgery_type",
    "chemotherapy",
    "radiotherapy",
    "hormone_therapy",
    "menopausal_status",
    "histology_type",
    "mutation_count_log1p",
    "npi",
)

# node_group: 0 = node-negative, 1 = 1-3 positive nodes, 2 = >= 4 (harmonisable
# with pN stage and with the SCAN-B lymph-node group).
ALL_CLINICAL_FEATURES: Tuple[str, ...] = CLINICAL_FEATURES + ("node_group",)

# Clinical variables that can be harmonised with external cohorts.  The
# mutation count is excluded (targeted panel vs whole-exome counts are not
# comparable) and the lymph-node count is replaced by the node group.
TRANSPORTABLE_CLINICAL: Tuple[str, ...] = (
    "age_at_diagnosis", "tumor_size", "node_group", "grade", "stage", "er_status",
    "pr_status", "her2_status", "chemotherapy", "radiotherapy", "hormone_therapy",
    "menopausal_status", "histology_type", "npi",
)

BINARY_CLINICAL = {
    "er_status", "pr_status", "her2_status", "surgery_type", "chemotherapy",
    "radiotherapy", "hormone_therapy", "menopausal_status", "histology_type",
}

# Clinical-to-pathway relations (biological / clinical priors, not outcomes).
CLINICAL_PATHWAY_MAP: Mapping[str, Tuple[str, ...]] = {
    "er_status": ("ER_signaling",),
    "pr_status": ("ER_signaling",),
    "her2_status": ("HER2_ERBB", "PI3K_AKT_mTOR"),
    "grade": ("Cell_cycle", "p53_DNA_repair"),
    "stage": ("Growth_factor", "RAS_MAPK"),
    "tumor_size": ("Growth_factor",),
    "lymph_nodes_positive": ("Immune", "Growth_factor"),
    "node_group": ("Immune", "Growth_factor"),
    "chemotherapy": ("Cell_cycle", "p53_DNA_repair"),
    "radiotherapy": ("p53_DNA_repair",),
    "hormone_therapy": ("ER_signaling",),
    "mutation_count_log1p": ("p53_DNA_repair", "Chromatin"),
    "npi": ("Cell_cycle", "Growth_factor"),
}

SUBTYPE_PATHWAY_MAP: Mapping[str, Tuple[str, ...]] = {
    "Luminal_A": ("ER_signaling", "Apoptosis"),
    "Luminal_B": ("ER_signaling", "Cell_cycle", "PI3K_AKT_mTOR"),
    "HER2_enriched": ("HER2_ERBB", "PI3K_AKT_mTOR", "Growth_factor"),
    "Basal_like": ("p53_DNA_repair", "Cell_cycle", "Immune"),
    "Other": ("Chromatin", "Immune"),
}

# Ordinal confidence tiers of the soft edge priors (see manuscript Sec. 3.2.3).
DEFAULT_PRIORS: Mapping[str, float] = {
    "membership": 1.0,            # curated gene-pathway membership (definitional)
    "subtype_pathway": 0.9,       # established subtype biology
    "clinical_pathway": 0.8,      # indirect clinical-pathway association
    "subtype_patient": 0.9,       # aggregation edges into the patient node
    "pathway_patient": 0.85,
    "gene_patient_curated": 0.75,
    "clinical_patient": 0.7,
    "gene_patient_other": 0.45,   # data-driven genes without curated membership
}

KNOWN_METABRIC_CLINICAL_COLUMNS = {
    "patient_id", "age_at_diagnosis", "type_of_breast_surgery", "cancer_type",
    "cancer_type_detailed", "cellularity", "chemotherapy",
    "pam50_+_claudin-low_subtype", "cohort", "er_status_measured_by_ihc",
    "er_status", "neoplasm_histologic_grade", "her2_status_measured_by_snp6",
    "her2_status", "tumor_other_histologic_subtype", "hormone_therapy",
    "inferred_menopausal_state", "integrative_cluster", "primary_tumor_laterality",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "oncotree_code", "overall_survival_months", "overall_survival", "pr_status",
    "radio_therapy", "3-gene_classifier_subtype", "tumor_size", "tumor_stage",
    "death_from_cancer",
}


# =====================================================================
# Parsing helpers
# =====================================================================


def canonical_subtype_label(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Other"
    s = str(value).strip()
    if s.upper().startswith("BRCA_"):
        s = s[5:]
    return PAM50_TO_CANONICAL.get(s.upper(), "Other")


def parse_mutation_series(series: pd.Series) -> np.ndarray:
    """Return 1 (mutated), 0 (wild type) or NaN (not profiled)."""
    out = np.full(len(series), np.nan)
    values = series.to_numpy(dtype=object)
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, (int, np.integer)):
            out[i] = 1.0 if v != 0 else 0.0
            continue
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                continue
            out[i] = 1.0 if v != 0 else 0.0
            continue
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "na", "none"}:
            continue
        out[i] = 0.0 if s in {"0", "0.0"} else 1.0
    return out


def _binary_status(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "positive": 1.0, "positve": 1.0, "pos": 1.0, "yes": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0,
        "negative": 0.0, "neg": 0.0, "no": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0,
    }
    mapped = normalized.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.where(mapped.notna(), numeric)


def node_group_from_count(count: pd.Series) -> pd.Series:
    c = pd.to_numeric(count, errors="coerce")
    out = pd.Series(np.nan, index=c.index)
    out[c <= 0] = 0.0
    out[(c >= 1) & (c <= 3)] = 1.0
    out[c >= 4] = 2.0
    return out


def npi_formula(tumor_size_mm: pd.Series, grade: pd.Series, node_group: pd.Series) -> pd.Series:
    """NPI = 0.2 * size(cm) + grade + nodal score (1, 2, 3)."""
    return 0.2 * (tumor_size_mm / 10.0) + grade + (node_group + 1.0)


def ihc_surrogate_subtype(clinical: pd.DataFrame) -> np.ndarray:
    """St Gallen-style surrogate subtype from ER, PR, HER2 and grade.

    Grade is used as the proliferation proxy because Ki-67 is not available.
    HR+/HER2-/PR+/grade 1-2 -> Luminal A-like; other HR+/HER2- and HR+/HER2+
    -> Luminal B-like; HR-/HER2+ -> HER2-enriched-like; HR-/HER2- -> Basal-like
    (triple negative); incomplete receptor data -> Other.
    """
    er = clinical["er_status"].to_numpy(float)
    pr = clinical["pr_status"].to_numpy(float)
    her2 = clinical["her2_status"].to_numpy(float)
    grade = clinical["grade"].to_numpy(float)
    out = np.full(len(clinical), "Other", dtype=object)
    known = np.isfinite(er) & np.isfinite(her2)
    hr = (er == 1) | (pr == 1)
    lum_a = known & hr & (her2 == 0) & (pr == 1) & np.isfinite(grade) & (grade <= 2)
    lum_b = known & hr & ~lum_a
    her2_e = known & ~hr & (her2 == 1)
    basal = known & ~hr & (her2 == 0)
    out[lum_a] = "Luminal_A"
    out[lum_b] = "Luminal_B"
    out[her2_e] = "HER2_enriched"
    out[basal] = "Basal_like"
    return out.astype(str)


# =====================================================================
# Cohort table
# =====================================================================


@dataclass
class CohortTable:
    name: str
    patient_ids: np.ndarray
    clinical: pd.DataFrame          # columns = ALL_CLINICAL_FEATURES (NaN allowed)
    expression: pd.DataFrame        # columns = upper-case gene symbols
    mutation: Optional[pd.DataFrame]  # columns = upper-case gene symbols, 0/1/NaN
    times: np.ndarray
    events: np.ndarray
    pam50: np.ndarray               # canonical labels (SUBTYPE_NAMES)
    cohorts: np.ndarray

    def __len__(self) -> int:
        return len(self.times)

    @property
    def ihc_subtype(self) -> np.ndarray:
        return ihc_surrogate_subtype(self.clinical)

    def subset(self, indices: Sequence[int]) -> "CohortTable":
        idx = np.asarray(indices, dtype=int)
        return CohortTable(
            name=self.name,
            patient_ids=self.patient_ids[idx],
            clinical=self.clinical.iloc[idx].reset_index(drop=True),
            expression=self.expression.iloc[idx].reset_index(drop=True),
            mutation=None if self.mutation is None else self.mutation.iloc[idx].reset_index(drop=True),
            times=self.times[idx],
            events=self.events[idx],
            pam50=self.pam50[idx],
            cohorts=self.cohorts[idx],
        )

    # ---------------------------------------------------------------
    @classmethod
    def from_metabric(cls, path: Path | str = DATA_PATH, endpoint: str = SURVIVAL_ENDPOINT) -> "CohortTable":
        raw = load_metabric_raw(path)
        clinical = build_clinical_frame(raw)
        expr_cols, mut_cols = identify_expression_mutation_columns(raw)
        expression = raw[expr_cols].apply(pd.to_numeric, errors="coerce")
        expression.columns = [c.upper() for c in expr_cols]
        mutation = pd.DataFrame(
            {c[:-4].upper(): parse_mutation_series(raw[c]) for c in mut_cols},
            index=raw.index,
        )
        return cls(
            name="METABRIC",
            patient_ids=raw["patient_id"].astype(str).to_numpy(),
            clinical=clinical.reset_index(drop=True),
            expression=expression.reset_index(drop=True),
            mutation=mutation.reset_index(drop=True),
            times=overall_survival_times(raw).astype(float),
            events=survival_event_indicator(raw, endpoint=endpoint),
            pam50=canonical_subtypes(raw).to_numpy(str),
            cohorts=pd.to_numeric(raw["cohort"], errors="coerce").to_numpy(float),
        )

    @classmethod
    def from_harmonized_csv(cls, path: Path | str, name: Optional[str] = None,
                            require_outcomes: bool = True) -> "CohortTable":
        """Load a cohort written in the harmonised schema (see external_data.py)."""
        df = pd.read_csv(path, low_memory=False)
        required = {"patient_id", "time_months", "event"} if require_outcomes else {"patient_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Harmonised file {path} lacks columns {sorted(missing)}")
        if not require_outcomes:
            # Prediction-only placeholders. These are never used as observed
            # endpoints or passed to fitting/evaluation by the inference CLI.
            df['time_months'] = 1.0
            df['event'] = 0
        times = pd.to_numeric(df["time_months"], errors="coerce")
        events = pd.to_numeric(df["event"], errors="coerce")
        if not ((times > 0) & np.isfinite(times) & events.isin([0, 1])).all():
            raise ValueError("Invalid/missing survival endpoint; missing events must not be converted to censoring")
        if df['patient_id'].isna().any() or df['patient_id'].duplicated().any():
            raise ValueError("Patient IDs must be present and unique")
        df = df.reset_index(drop=True)
        clinical = pd.DataFrame(index=df.index)
        for col in ALL_CLINICAL_FEATURES:
            clinical[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        expr_cols = [c for c in df.columns if c.startswith("expr:")]
        mut_cols = [c for c in df.columns if c.startswith("mut:")]
        expression = df[expr_cols].apply(pd.to_numeric, errors="coerce")
        expression.columns = [c[5:].upper() for c in expr_cols]
        mutation = None
        if mut_cols:
            mutation = df[mut_cols].apply(pd.to_numeric, errors="coerce")
            mutation.columns = [c[4:].upper() for c in mut_cols]
        pam50 = (df["pam50"].map(canonical_subtype_label).to_numpy(str)
                 if "pam50" in df.columns else np.full(len(df), "Other"))
        cohorts = (pd.to_numeric(df["cohort"], errors="coerce").to_numpy(float)
                   if "cohort" in df.columns else np.full(len(df), np.nan))
        return cls(
            name=name or Path(path).stem,
            patient_ids=df["patient_id"].astype(str).to_numpy(),
            clinical=clinical,
            expression=expression,
            mutation=mutation,
            times=pd.to_numeric(df["time_months"], errors="coerce").to_numpy(float),
            events=pd.to_numeric(df["event"], errors="raise").astype(int).to_numpy(),
            pam50=pam50,
            cohorts=cohorts,
        )

    def to_harmonized_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({
            "patient_id": self.patient_ids,
            "time_months": self.times,
            "event": self.events,
            "pam50": self.pam50,
            "cohort": self.cohorts,
        })
        for col in ALL_CLINICAL_FEATURES:
            out[col] = self.clinical[col].to_numpy(float)
        expr = self.expression.copy()
        expr.columns = [f"expr:{g}" for g in expr.columns]
        parts = [out, expr.reset_index(drop=True)]
        if self.mutation is not None:
            mut = self.mutation.copy()
            mut.columns = [f"mut:{g}" for g in mut.columns]
            parts.append(mut.reset_index(drop=True))
        return pd.concat(parts, axis=1)


def clinical_coverage(table: CohortTable) -> pd.Series:
    """Fraction of non-missing values per clinical feature."""
    return table.clinical.notna().mean()


def choose_transportable_features(table: CohortTable, min_coverage: float = 0.6) -> List[str]:
    cov = clinical_coverage(table)
    return [c for c in TRANSPORTABLE_CLINICAL if cov.get(c, 0.0) >= min_coverage]


def pathway_gene_coverage(gene_symbols: Iterable[str]) -> pd.DataFrame:
    """Curated vs matched pathway genes for an expression panel (manuscript Table 1)."""
    available = {g.upper() for g in gene_symbols}
    rows = []
    for pathway, genes in PATHWAY_GENE_MAP.items():
        matched = [g for g in genes if g in available]
        rows.append({
            "pathway": pathway,
            "n_curated": len(genes),
            "n_matched": len(matched),
            "matched_genes": ", ".join(matched),
            "unmatched_genes": ", ".join(g for g in genes if g not in available),
        })
    return pd.DataFrame(rows)


# =====================================================================
# METABRIC specific loading
# =====================================================================


def load_metabric_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"METABRIC file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    times = pd.to_numeric(df["overall_survival_months"], errors="coerce")
    valid = times.notna() & (times > 0)
    return df.loc[valid].reset_index(drop=True)


def overall_survival_times(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["overall_survival_months"], errors="coerce").to_numpy(float)


def survival_event_indicator(df: pd.DataFrame, endpoint: str = SURVIVAL_ENDPOINT) -> np.ndarray:
    """Death indicator (1 = event).  ``overall_survival`` is a living flag in this dump."""
    endpoint = str(endpoint).lower().strip()
    if endpoint not in {"os", "dss"}:
        raise ValueError("endpoint must be 'os' or 'dss'")
    if "death_from_cancer" in df.columns:
        status = df["death_from_cancer"].astype(str).str.strip().str.lower()
        died_disease = status.str.contains("died of disease", na=False)
        died_other = status.str.contains("died of other", na=False)
        living = status.str.contains("living", na=False)
        if endpoint == "dss":
            events = died_disease.to_numpy(bool)
        else:
            events = (died_disease | died_other).to_numpy(bool)
            missing = ~(died_disease | died_other | living).to_numpy(bool)
            if missing.any() and "overall_survival" in df.columns:
                os_living = pd.to_numeric(df["overall_survival"], errors="coerce").to_numpy(float)
                events = np.where(missing, os_living == 0.0, events)
        return events.astype(np.int64)
    if "overall_survival" not in df.columns:
        return np.zeros(len(df), dtype=np.int64)
    os_living = pd.to_numeric(df["overall_survival"], errors="coerce").fillna(1.0).to_numpy(float)
    return (os_living == 0.0).astype(np.int64)


def identify_expression_mutation_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    mut_cols = [c for c in df.columns if c.lower().endswith("_mut")]
    mut_set = set(mut_cols)
    expr_cols = [
        c for c in df.columns
        if c not in KNOWN_METABRIC_CLINICAL_COLUMNS
        and c not in mut_set
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return expr_cols, mut_cols


def canonical_subtypes(df: pd.DataFrame) -> pd.Series:
    raw = df["pam50_+_claudin-low_subtype"]
    return raw.map(canonical_subtype_label).fillna("Other")


def build_clinical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age_at_diagnosis"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
    out["tumor_size"] = pd.to_numeric(df["tumor_size"], errors="coerce")
    out["lymph_nodes_positive"] = pd.to_numeric(df["lymph_nodes_examined_positive"], errors="coerce")
    out["grade"] = pd.to_numeric(df["neoplasm_histologic_grade"], errors="coerce")
    out["stage"] = pd.to_numeric(df["tumor_stage"], errors="coerce")
    out["er_status"] = _binary_status(df["er_status"])
    out["pr_status"] = _binary_status(df["pr_status"])
    out["her2_status"] = _binary_status(df["her2_status"])
    out["surgery_type"] = df["type_of_breast_surgery"].astype(str).str.upper().str.contains(
        "MASTECTOMY", na=False
    ).astype(float)
    out["chemotherapy"] = pd.to_numeric(df["chemotherapy"], errors="coerce")
    out["radiotherapy"] = pd.to_numeric(df["radio_therapy"], errors="coerce")
    out["hormone_therapy"] = pd.to_numeric(df["hormone_therapy"], errors="coerce")
    out["menopausal_status"] = (
        df["inferred_menopausal_state"].astype(str).str.lower().map({"pre": 0.0, "post": 1.0})
    )
    out["histology_type"] = df["cancer_type_detailed"].astype(str).str.lower().str.contains(
        "lobular", na=False
    ).astype(float)
    mutation_count = pd.to_numeric(df["mutation_count"], errors="coerce").clip(lower=0)
    out["mutation_count_log1p"] = np.log1p(mutation_count)
    out["node_group"] = node_group_from_count(out["lymph_nodes_positive"])
    npi_original = pd.to_numeric(df["nottingham_prognostic_index"], errors="coerce")
    fallback = npi_formula(out["tumor_size"], out["grade"], out["node_group"].fillna(0.0))
    out["npi"] = npi_original.where(npi_original.notna(), fallback)
    return out.loc[:, list(ALL_CLINICAL_FEATURES)]


def load_metabric(endpoint: str = SURVIVAL_ENDPOINT, path: Path | str = DATA_PATH) -> CohortTable:
    return CohortTable.from_metabric(path, endpoint=endpoint)


# =====================================================================
# Gene screening (vectorised log-rank)
# =====================================================================


def _logrank_score(times: np.ndarray, events: np.ndarray, values: np.ndarray) -> float:
    """Reference (scalar) implementation kept for tests."""
    return float(logrank_scores_matrix(times, events, np.asarray(values, float)[:, None])[0])


def logrank_scores_matrix(times: np.ndarray, events: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Median-split log-rank chi-square for every column of X (NaN-aware).

    Columns with < 100 valid samples or < 30 patients in either group score 0.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    X = np.asarray(X, dtype=float)
    valid_t = np.isfinite(times) & (times > 0)
    V = np.isfinite(X) & valid_t[:, None]
    n_valid = V.sum(axis=0)
    masked = np.where(V, X, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(masked, axis=0)
    G = (masked > med[None, :]) & V
    ok = (n_valid >= 100) & (G.sum(axis=0) >= 30) & ((V & ~G).sum(axis=0) >= 30)

    order = np.argsort(np.where(valid_t, times, np.inf), kind="mergesort")
    t = times[order]
    e = events[order]
    Vs = V[order].astype(np.int64)
    Gs = G[order].astype(np.int64)
    finite = np.isfinite(t) & valid_t[order]
    uniq, first = np.unique(t[finite], return_index=True)
    rev_V = np.cumsum(Vs[::-1], axis=0)[::-1]
    rev_G = np.cumsum(Gs[::-1], axis=0)[::-1]
    n_at = rev_V[first].astype(float)
    n1_at = rev_G[first].astype(float)
    ev = (e == 1)[:, None]
    D = (Vs * ev)[: finite.sum()]
    D1 = (Gs * ev)[: finite.sum()]
    d = np.add.reduceat(D, first, axis=0).astype(float)
    d1 = np.add.reduceat(D1, first, axis=0).astype(float)
    use = (d > 0) & (n_at > 1)
    with np.errstate(all="ignore"):
        expected = np.where(use, d * n1_at / n_at, 0.0)
        var = np.where(use, n1_at * (n_at - n1_at) * d * (n_at - d) / (n_at * n_at * np.maximum(n_at - 1, 1)), 0.0)
    ome = np.sum(np.where(use, d1 - expected, 0.0), axis=0)
    vs = np.sum(var, axis=0)
    score = np.where((vs > 1e-12) & ok, ome ** 2 / np.where(vs > 1e-12, vs, 1.0), 0.0)
    return score


def _safe_standard_deviation(values: np.ndarray) -> np.ndarray:
    std = np.nanstd(values, axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return std


# =====================================================================
# Graph schema
# =====================================================================


@dataclass
class GraphConfig:
    """Knowledge-graph variant used to build the fold schema."""

    priors: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIORS))
    prior_mode: str = "default"         # default | uniform | inverted | perturb | permute
    perturb_sigma: float = 0.3          # log-normal noise for prior_mode="perturb"
    rewire_knowledge: bool = False      # degree-preserving rewiring of curated edges
    relation_aware: bool = True         # False -> all edges share one relation type
    include_subtype_nodes: bool = True
    seed: int = 0

    def describe(self) -> str:
        parts = [f"prior={self.prior_mode}"]
        if self.prior_mode == "perturb":
            parts.append(f"sigma={self.perturb_sigma}")
        if self.rewire_knowledge:
            parts.append("rewired")
        if not self.relation_aware:
            parts.append("relation-agnostic")
        if not self.include_subtype_nodes:
            parts.append("no-subtype-nodes")
        if self.prior_mode in {"perturb", "permute"} or self.rewire_knowledge:
            parts.append(f"seed={self.seed}")
        return ",".join(parts)


@dataclass(frozen=True)
class GraphSchema:
    node_names: Tuple[str, ...]
    node_types: Tuple[str, ...]
    node_type_ids: np.ndarray
    relation_edges: Dict[str, np.ndarray]
    relation_priors: Dict[str, np.ndarray]
    patient_node_index: int
    subtype_node_indices: Dict[str, int]
    gene_node_indices: Dict[str, int]
    pathway_node_indices: Dict[str, int]
    clinical_node_indices: Dict[str, int]

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def relation_names(self) -> Tuple[str, ...]:
        return tuple(self.relation_edges.keys())

    @property
    def n_edges(self) -> int:
        return int(sum(e.shape[1] for e in self.relation_edges.values()))

    def in_degree(self, relation: str) -> np.ndarray:
        edges = self.relation_edges[relation]
        return np.bincount(edges[1], minlength=self.n_nodes)

    def to_json_dict(self) -> dict:
        return {
            "node_names": list(self.node_names),
            "node_types": list(self.node_types),
            "patient_node_index": self.patient_node_index,
            "subtype_node_indices": self.subtype_node_indices,
            "gene_node_indices": self.gene_node_indices,
            "pathway_node_indices": self.pathway_node_indices,
            "clinical_node_indices": self.clinical_node_indices,
            "relations": {
                name: {
                    "n_edges": int(edges.shape[1]),
                    "edges": edges.T.tolist(),
                    "priors": self.relation_priors[name].tolist(),
                }
                for name, edges in self.relation_edges.items()
            },
        }


def _rewire_bipartite(edges: List[Tuple[str, str]], rng: np.random.Generator, factor: int = 20) -> List[Tuple[str, str]]:
    """Degree-preserving double-edge swaps on a bipartite edge list."""
    edges = list(edges)
    m = len(edges)
    if m < 2:
        return edges
    present = set(edges)
    for _ in range(factor * m):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        (a1, b1), (a2, b2) = edges[i], edges[j]
        if a1 == a2 or b1 == b2:
            continue
        if (a1, b2) in present or (a2, b1) in present:
            continue
        present.discard((a1, b1))
        present.discard((a2, b2))
        present.add((a1, b2))
        present.add((a2, b1))
        edges[i] = (a1, b2)
        edges[j] = (a2, b1)
    return edges


@dataclass
class GraphDataset:
    node_features: np.ndarray  # [patients, nodes, 3]
    subtype_ids: np.ndarray
    n_subtypes: int
    times: np.ndarray
    events: np.ndarray
    flat_features: np.ndarray
    flat_blocks: Dict[str, Tuple[int, int]]
    flat_names: List[str]
    patient_ids: np.ndarray
    cohorts: np.ndarray
    subtypes: np.ndarray
    schema: GraphSchema
    clinical_features: Tuple[str, ...]
    selected_genes: Tuple[str, ...]
    pathway_members: Dict[str, Tuple[str, ...]]
    gene_expression: np.ndarray  # scaled expression [patients, genes]
    gene_mutation: np.ndarray    # mutation indicators [patients, genes]
    gene_has_mutation: np.ndarray  # bool [genes]

    def __len__(self) -> int:
        return len(self.times)

    @property
    def clinical_dim(self) -> int:
        return len(self.clinical_features)

    def block(self, *names: str) -> np.ndarray:
        cols = []
        for name in names:
            if name not in self.flat_blocks:
                continue
            start, stop = self.flat_blocks[name]
            cols.append(self.flat_features[:, start:stop])
        if not cols:
            return np.zeros((len(self.times), 0), dtype=np.float32)
        return np.concatenate(cols, axis=1)

    def subset(self, indices: Sequence[int]) -> "GraphDataset":
        idx = np.asarray(indices, dtype=int)
        return GraphDataset(
            node_features=self.node_features[idx],
            subtype_ids=self.subtype_ids[idx],
            n_subtypes=self.n_subtypes,
            times=self.times[idx],
            events=self.events[idx],
            flat_features=self.flat_features[idx],
            flat_blocks=self.flat_blocks,
            flat_names=self.flat_names,
            patient_ids=self.patient_ids[idx],
            cohorts=self.cohorts[idx],
            subtypes=self.subtypes[idx],
            schema=self.schema,
            clinical_features=self.clinical_features,
            selected_genes=self.selected_genes,
            pathway_members=self.pathway_members,
            gene_expression=self.gene_expression[idx],
            gene_mutation=self.gene_mutation[idx],
            gene_has_mutation=self.gene_has_mutation,
        )


# =====================================================================
# Fold preprocessor
# =====================================================================


class FoldPreprocessor:
    """Fit all data-dependent preprocessing on training patients only."""

    def __init__(
        self,
        data_driven_top_k: int = 30,
        max_genes: int = 90,
        clinical_features: Optional[Sequence[str]] = None,
        subtype_source: str = "pam50",
        use_mutations: bool = True,
        graph_config: Optional[GraphConfig] = None,
    ) -> None:
        if data_driven_top_k < 0:
            raise ValueError("data_driven_top_k must be non-negative")
        if subtype_source not in {"pam50", "ihc", "none"}:
            raise ValueError("subtype_source must be 'pam50', 'ihc' or 'none'")
        self.data_driven_top_k = int(data_driven_top_k)
        self.max_genes = int(max_genes)
        self.clinical_features: Tuple[str, ...] = tuple(clinical_features or CLINICAL_FEATURES)
        unknown = set(self.clinical_features) - set(ALL_CLINICAL_FEATURES)
        if unknown:
            raise ValueError(f"Unknown clinical features: {sorted(unknown)}")
        self.subtype_source = subtype_source
        self.use_mutations = bool(use_mutations)
        self.graph_config = graph_config or GraphConfig()
        if subtype_source == "none":
            self.graph_config = copy.deepcopy(self.graph_config)
            self.graph_config.include_subtype_nodes = False
        self.selected_genes: List[str] = []
        self.gene_scores: Dict[str, float] = {}
        self.gene_has_mutation: np.ndarray = np.zeros(0, dtype=bool)
        self.clinical_medians: Optional[np.ndarray] = None
        self.clinical_means: Optional[np.ndarray] = None
        self.clinical_stds: Optional[np.ndarray] = None
        self.expression_medians: Optional[np.ndarray] = None
        self.expression_means: Optional[np.ndarray] = None
        self.expression_stds: Optional[np.ndarray] = None
        self.schema: Optional[GraphSchema] = None
        self.pathway_members: Dict[str, Tuple[str, ...]] = {}
        self._fitted = False

    # -----------------------------------------------------------------
    @property
    def subtype_names(self) -> Tuple[str, ...]:
        return ("All",) if self.subtype_source == "none" else SUBTYPE_NAMES

    def subtype_labels(self, table: CohortTable, idx: np.ndarray) -> np.ndarray:
        if self.subtype_source == "pam50":
            return table.pam50[idx]
        if self.subtype_source == "ihc":
            return table.ihc_subtype[idx]
        return np.full(len(idx), "All")

    # -----------------------------------------------------------------
    def fit(self, table: CohortTable, train_indices: Sequence[int]) -> "FoldPreprocessor":
        train_idx = np.asarray(train_indices, dtype=int)
        genes_available = list(table.expression.columns)
        available_set = set(genes_available)
        curated_order: List[str] = []
        for genes in PATHWAY_GENE_MAP.values():
            for gene in genes:
                if gene in available_set and gene not in curated_order:
                    curated_order.append(gene)

        X_train = table.expression.iloc[train_idx].to_numpy(float)
        scores_arr = logrank_scores_matrix(table.times[train_idx], table.events[train_idx], X_train)
        scores = {g: float(s) for g, s in zip(genes_available, scores_arr)}
        ranked = sorted(scores, key=lambda g: (-scores[g], g))
        limit = min(self.max_genes, len(curated_order) + self.data_driven_top_k)
        selected = list(curated_order)
        for gene in ranked:
            if len(selected) >= limit:
                break
            if gene not in selected:
                selected.append(gene)
        if not selected:
            raise RuntimeError("No expression genes were identified")
        self.selected_genes = selected[: self.max_genes]
        self.gene_scores = {g: float(scores.get(g, 0.0)) for g in self.selected_genes}

        mut_cols = set(table.mutation.columns) if (table.mutation is not None and self.use_mutations) else set()
        self.gene_has_mutation = np.asarray([g in mut_cols for g in self.selected_genes], dtype=bool)

        clinical = table.clinical.iloc[train_idx][list(self.clinical_features)].to_numpy(float)
        import warnings

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.clinical_medians = np.nanmedian(clinical, axis=0)
        self.clinical_medians[~np.isfinite(self.clinical_medians)] = 0.0
        clinical_imputed = np.where(np.isfinite(clinical), clinical, self.clinical_medians)
        self.clinical_means = np.mean(clinical_imputed, axis=0)
        self.clinical_stds = _safe_standard_deviation(clinical_imputed)

        expression = table.expression.iloc[train_idx][self.selected_genes].to_numpy(float)
        self.expression_medians, self.expression_means, self.expression_stds = self._expression_stats(expression)

        self._build_schema()
        self._fitted = True
        return self

    @staticmethod
    def _expression_stats(expression: np.ndarray):
        import warnings

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(expression, axis=0)
        med[~np.isfinite(med)] = 0.0
        imputed = np.where(np.isfinite(expression), expression, med)
        return med, np.mean(imputed, axis=0), _safe_standard_deviation(imputed)

    def with_graph_config(self, graph_config: GraphConfig) -> "FoldPreprocessor":
        """Copy with the same fitted statistics but a different knowledge graph."""
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        other = copy.copy(self)
        other.graph_config = copy.deepcopy(graph_config)
        if self.subtype_source == "none":
            other.graph_config.include_subtype_nodes = False
        other._build_schema()
        return other

    # -----------------------------------------------------------------
    def _build_schema(self) -> None:
        cfg = self.graph_config
        rng = np.random.default_rng(cfg.seed)
        node_names: List[str] = []
        node_types: List[str] = []

        def add_nodes(prefix: str, names: Iterable[str], node_type: str) -> Dict[str, int]:
            result: Dict[str, int] = {}
            for name in names:
                result[name] = len(node_names)
                node_names.append(f"{prefix}:{name}")
                node_types.append(node_type)
            return result

        include_subtypes = cfg.include_subtype_nodes and self.subtype_source != "none"
        gene_nodes = add_nodes("gene", self.selected_genes, "gene")
        pathway_nodes = add_nodes("pathway", PATHWAY_GENE_MAP.keys(), "pathway")
        clinical_nodes = add_nodes("clinical", self.clinical_features, "clinical")
        subtype_nodes = add_nodes("subtype", SUBTYPE_NAMES, "subtype") if include_subtypes else {}
        patient_index = len(node_names)
        node_names.append("patient:index")
        node_types.append("patient")
        type_order = ("gene", "pathway", "clinical", "subtype", "patient")
        type_to_id = {name: i for i, name in enumerate(type_order)}
        type_ids = np.asarray([type_to_id[t] for t in node_types], dtype=np.int64)

        # Curated knowledge edges (possibly rewired)
        membership = [(g, p) for p, genes in PATHWAY_GENE_MAP.items() for g in genes if g in gene_nodes]
        membership = list(dict.fromkeys(membership))
        clin_path = [(c, p) for c, ps in CLINICAL_PATHWAY_MAP.items() if c in clinical_nodes for p in ps]
        sub_path = [(s, p) for s, ps in SUBTYPE_PATHWAY_MAP.items() if s in subtype_nodes for p in ps]
        if cfg.rewire_knowledge:
            membership = _rewire_bipartite(membership, rng)
            clin_path = _rewire_bipartite(clin_path, rng)
            sub_path = _rewire_bipartite(sub_path, rng)
        members: Dict[str, List[str]] = {p: [] for p in PATHWAY_GENE_MAP}
        for g, p in membership:
            members[p].append(g)
        self.pathway_members = {p: tuple(gs) for p, gs in members.items()}

        priors = dict(DEFAULT_PRIORS)
        priors.update(cfg.priors or {})
        edge_store: Dict[str, Dict[Tuple[int, int], float]] = {}
        tier_of_edge: Dict[Tuple[str, int, int], str] = {}

        def add_edge(relation: str, src: int, dst: int, tier: str) -> None:
            rel = edge_store.setdefault(relation, {})
            value = float(priors[tier])
            if value >= rel.get((src, dst), -1.0):
                rel[(src, dst)] = value
                tier_of_edge[(relation, src, dst)] = tier

        membership_count = {g: 0 for g in self.selected_genes}
        for g, p in membership:
            membership_count[g] += 1
            add_edge("gene_to_pathway", gene_nodes[g], pathway_nodes[p], "membership")
            add_edge("pathway_to_gene", pathway_nodes[p], gene_nodes[g], "membership")
        for gene, g_idx in gene_nodes.items():
            tier = "gene_patient_curated" if membership_count[gene] > 0 else "gene_patient_other"
            add_edge("gene_to_patient", g_idx, patient_index, tier)
            add_edge("patient_to_gene", patient_index, g_idx, tier)
        for c, p in clin_path:
            add_edge("clinical_to_pathway", clinical_nodes[c], pathway_nodes[p], "clinical_pathway")
            add_edge("pathway_to_clinical", pathway_nodes[p], clinical_nodes[c], "clinical_pathway")
        for s, p in sub_path:
            add_edge("subtype_to_pathway", subtype_nodes[s], pathway_nodes[p], "subtype_pathway")
            add_edge("pathway_to_subtype", pathway_nodes[p], subtype_nodes[s], "subtype_pathway")
        for c_idx in clinical_nodes.values():
            add_edge("clinical_to_patient", c_idx, patient_index, "clinical_patient")
            add_edge("patient_to_clinical", patient_index, c_idx, "clinical_patient")
        for p_idx in pathway_nodes.values():
            add_edge("pathway_to_patient", p_idx, patient_index, "pathway_patient")
            add_edge("patient_to_pathway", patient_index, p_idx, "pathway_patient")
        for s_idx in subtype_nodes.values():
            add_edge("subtype_to_patient", s_idx, patient_index, "subtype_patient")
            add_edge("patient_to_subtype", patient_index, s_idx, "subtype_patient")

        relation_edges: Dict[str, np.ndarray] = {}
        relation_priors: Dict[str, np.ndarray] = {}
        for relation, mapping in edge_store.items():
            pairs = sorted(mapping)
            relation_edges[relation] = np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
            relation_priors[relation] = np.asarray([mapping[p] for p in pairs], dtype=np.float32)

        relation_priors = self._transform_priors(relation_priors, priors, rng)

        if not cfg.relation_aware:
            names = list(relation_edges)
            relation_edges = {"shared": np.concatenate([relation_edges[r] for r in names], axis=1)}
            relation_priors = {"shared": np.concatenate([relation_priors[r] for r in names])}

        self.schema = GraphSchema(
            node_names=tuple(node_names),
            node_types=tuple(node_types),
            node_type_ids=type_ids,
            relation_edges=relation_edges,
            relation_priors=relation_priors,
            patient_node_index=patient_index,
            subtype_node_indices=subtype_nodes,
            gene_node_indices=gene_nodes,
            pathway_node_indices=pathway_nodes,
            clinical_node_indices=clinical_nodes,
        )

    def _transform_priors(self, relation_priors: Dict[str, np.ndarray], priors: Mapping[str, float],
                          rng: np.random.Generator) -> Dict[str, np.ndarray]:
        mode = self.graph_config.prior_mode
        if mode == "default":
            return relation_priors
        if mode == "uniform":
            return {r: np.ones_like(v) for r, v in relation_priors.items()}
        if mode == "inverted":
            values = np.asarray(list(priors.values()), dtype=float)
            hi, lo = values.max(), values.min()
            return {r: (hi + lo - v.astype(np.float64)).astype(np.float32) for r, v in relation_priors.items()}
        if mode == "perturb":
            sigma = float(self.graph_config.perturb_sigma)
            return {
                r: np.clip(v * np.exp(sigma * rng.standard_normal(v.shape)), 0.05, 1.0).astype(np.float32)
                for r, v in relation_priors.items()
            }
        if mode == "permute":
            names = list(relation_priors)
            sizes = [relation_priors[r].size for r in names]
            pooled = np.concatenate([relation_priors[r] for r in names])
            pooled = pooled[rng.permutation(pooled.size)]
            out: Dict[str, np.ndarray] = {}
            start = 0
            for r, s in zip(names, sizes):
                out[r] = pooled[start:start + s].astype(np.float32)
                start += s
            return out
        raise ValueError(f"Unknown prior_mode: {mode}")

    # -----------------------------------------------------------------
    def transform(self, table: CohortTable, indices: Sequence[int], expression_scaling: str = "train") -> GraphDataset:
        """Build node features and flat features for the given patients.

        expression_scaling: "train" uses training-fold medians/means/SDs (internal
        evaluation); "within" re-standardises each gene within the transformed
        set (used for external cohorts measured on another platform).
        """
        if not self._fitted or self.schema is None:
            raise RuntimeError("FoldPreprocessor.fit must be called before transform")
        idx = np.asarray(indices, dtype=int)
        n = idx.size
        schema = self.schema

        clinical_raw = table.clinical.iloc[idx][list(self.clinical_features)].to_numpy(float)
        clinical_observed = np.isfinite(clinical_raw).astype(np.float32)
        clinical_imputed = np.where(np.isfinite(clinical_raw), clinical_raw, self.clinical_medians)
        clinical_scaled = (clinical_imputed - self.clinical_means) / self.clinical_stds

        missing_genes = [g for g in self.selected_genes if g not in table.expression.columns]
        expr_df = table.expression.iloc[idx].reindex(columns=self.selected_genes)
        expression_raw = expr_df.to_numpy(float)
        expression_observed = np.isfinite(expression_raw).astype(np.float32)
        if expression_scaling == "within":
            med, mean, std = self._expression_stats(expression_raw)
            imputed = np.where(np.isfinite(expression_raw), expression_raw, med)
            expression_scaled = (imputed - mean) / std
            for j, g in enumerate(self.selected_genes):
                if g in missing_genes:
                    expression_scaled[:, j] = 0.0
        elif expression_scaling == "train":
            imputed = np.where(np.isfinite(expression_raw), expression_raw, self.expression_medians)
            expression_scaled = (imputed - self.expression_means) / self.expression_stds
        else:
            raise ValueError("expression_scaling must be 'train' or 'within'")

        n_genes = len(self.selected_genes)
        mutation_values = np.zeros((n, n_genes), dtype=float)
        mutation_observed = np.zeros((n, n_genes), dtype=float)
        if self.use_mutations and table.mutation is not None:
            for j, gene in enumerate(self.selected_genes):
                if not self.gene_has_mutation[j] or gene not in table.mutation.columns:
                    continue
                raw = table.mutation[gene].to_numpy(float)[idx]
                finite = np.isfinite(raw)
                mutation_values[:, j] = np.where(finite, np.clip(raw, 0, 1), 0.0)
                mutation_observed[:, j] = finite.astype(float)

        subtype_labels = self.subtype_labels(table, idx)
        names = self.subtype_names
        subtype_to_id = {s: i for i, s in enumerate(names)}
        default_id = subtype_to_id.get("Other", 0)
        subtype_ids = np.asarray([subtype_to_id.get(s, default_id) for s in subtype_labels], dtype=np.int64)
        subtype_onehot = np.eye(len(names), dtype=np.float32)[subtype_ids]

        pathways = list(PATHWAY_GENE_MAP)
        gene_to_col = {g: j for j, g in enumerate(self.selected_genes)}
        pathway_expr = np.zeros((n, len(pathways)), dtype=np.float32)
        pathway_mut = np.zeros_like(pathway_expr)
        pathway_observed = np.zeros_like(pathway_expr)
        for p, pathway in enumerate(pathways):
            cols = [gene_to_col[g] for g in self.pathway_members.get(pathway, ()) if g in gene_to_col]
            if not cols:
                continue
            pathway_expr[:, p] = np.mean(expression_scaled[:, cols], axis=1)
            pathway_mut[:, p] = np.mean(mutation_values[:, cols], axis=1)
            pathway_observed[:, p] = (np.mean(expression_observed[:, cols], axis=1) > 0).astype(np.float32)

        node_features = np.zeros((n, schema.n_nodes, 3), dtype=np.float32)
        for gene, node_idx in schema.gene_node_indices.items():
            j = gene_to_col[gene]
            node_features[:, node_idx, 0] = expression_scaled[:, j]
            node_features[:, node_idx, 1] = mutation_values[:, j]
            node_features[:, node_idx, 2] = np.maximum(expression_observed[:, j], mutation_observed[:, j])
        for p, pathway in enumerate(pathways):
            node_idx = schema.pathway_node_indices[pathway]
            node_features[:, node_idx, 0] = pathway_expr[:, p]
            node_features[:, node_idx, 1] = pathway_mut[:, p]
            node_features[:, node_idx, 2] = pathway_observed[:, p]
        for j, clinical in enumerate(self.clinical_features):
            node_idx = schema.clinical_node_indices[clinical]
            node_features[:, node_idx, 0] = clinical_scaled[:, j]
            if clinical in BINARY_CLINICAL:
                node_features[:, node_idx, 1] = clinical_imputed[:, j]
            node_features[:, node_idx, 2] = clinical_observed[:, j]
        for subtype, node_idx in schema.subtype_node_indices.items():
            s = SUBTYPE_NAMES.index(subtype)
            node_features[:, node_idx, 0] = (subtype_ids == s).astype(np.float32)
            node_features[:, node_idx, 2] = 1.0
        node_features[:, schema.patient_node_index, :] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

        blocks: List[Tuple[str, np.ndarray, List[str]]] = [
            ("clinical", clinical_scaled, list(self.clinical_features)),
            ("expression", expression_scaled, [f"expr:{g}" for g in self.selected_genes]),
        ]
        if self.use_mutations and self.gene_has_mutation.any():
            cols = np.where(self.gene_has_mutation)[0]
            blocks.append(("mutation", mutation_values[:, cols], [f"mut:{self.selected_genes[j]}" for j in cols]))
        blocks.append(("pathway_expr", pathway_expr, [f"pw_expr:{p}" for p in pathways]))
        if self.use_mutations and self.gene_has_mutation.any():
            blocks.append(("pathway_mut", pathway_mut, [f"pw_mut:{p}" for p in pathways]))
        if self.subtype_source != "none":
            blocks.append(("subtype", subtype_onehot, [f"subtype:{s}" for s in names]))
        flat_blocks: Dict[str, Tuple[int, int]] = {}
        flat_names: List[str] = []
        start = 0
        arrays = []
        for name, arr, cols in blocks:
            arrays.append(np.asarray(arr, dtype=np.float32))
            flat_blocks[name] = (start, start + arr.shape[1])
            flat_names.extend(cols)
            start += arr.shape[1]
        flat_features = np.concatenate(arrays, axis=1).astype(np.float32)

        return GraphDataset(
            node_features=node_features,
            subtype_ids=subtype_ids,
            n_subtypes=len(names),
            times=table.times[idx].astype(np.float32),
            events=table.events[idx].astype(np.int64),
            flat_features=flat_features,
            flat_blocks=flat_blocks,
            flat_names=flat_names,
            patient_ids=table.patient_ids[idx],
            cohorts=table.cohorts[idx],
            subtypes=np.asarray(subtype_labels, dtype=str),
            schema=schema,
            clinical_features=self.clinical_features,
            selected_genes=tuple(self.selected_genes),
            pathway_members=dict(self.pathway_members),
            gene_expression=expression_scaled.astype(np.float32),
            gene_mutation=mutation_values.astype(np.float32),
            gene_has_mutation=self.gene_has_mutation.copy(),
        )

    def fit_transform(self, table: CohortTable, train_indices: Sequence[int]) -> GraphDataset:
        self.fit(table, train_indices)
        return self.transform(table, train_indices)

    def metadata(self) -> dict:
        if not self._fitted or self.schema is None:
            raise RuntimeError("Preprocessor is not fitted")
        return {
            "selected_genes": self.selected_genes,
            "gene_scores": self.gene_scores,
            "clinical_features": list(self.clinical_features),
            "subtype_source": self.subtype_source,
            "use_mutations": self.use_mutations,
            "graph_config": self.graph_config.describe(),
            "pathway_members": {k: list(v) for k, v in self.pathway_members.items()},
            "pam50_genes_selected": [g for g in self.selected_genes if g in PAM50_GENES],
            "n_nodes": self.schema.n_nodes,
            "n_edges": self.schema.n_edges,
            "relations": {r: int(e.shape[1]) for r, e in self.schema.relation_edges.items()},
        }

    def save_metadata(self, output_path: Path | str) -> None:
        Path(output_path).write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")


def graph_statistics(schema: GraphSchema) -> pd.DataFrame:
    """Per-relation edge counts and in-degree statistics (manuscript Table 2)."""
    rows = []
    for relation, edges in schema.relation_edges.items():
        indeg = np.bincount(edges[1], minlength=schema.n_nodes)
        dest = indeg[indeg > 0]
        rows.append({
            "relation": relation,
            "n_edges": int(edges.shape[1]),
            "n_destinations": int(dest.size),
            "mean_in_degree": float(dest.mean()) if dest.size else 0.0,
            "max_in_degree": int(dest.max()) if dest.size else 0,
            "prior_min": float(schema.relation_priors[relation].min()),
            "prior_max": float(schema.relation_priors[relation].max()),
        })
    return pd.DataFrame(rows)
