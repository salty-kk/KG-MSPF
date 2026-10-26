"""Prepare external validation cohorts in the harmonised KG-MSPF schema.

Harmonised CSV columns
----------------------
patient_id, time_months, event (1 = death), pam50 (LumA/LumB/Her2/Basal/Normal or
canonical), cohort, clinical features named as in ``hetero_data.ALL_CLINICAL_FEATURES``
(missing ones may be absent), ``expr:<GENE>`` (log-scale expression; re-standardised
within the cohort at evaluation time) and optional ``mut:<GENE>`` (0/1).

Usage
-----
TCGA-BRCA (cBioPortal PanCancer Atlas, optional legacy study for receptor status)::

    python external_data.py tcga --pancan-dir external_data/brca_tcga_pan_can_atlas_2018 \
        --legacy-dir external_data/brca_tcga --out external_data/tcga_brca_harmonized.csv

SCAN-B (GEO GSE96058)::

    python external_data.py scanb \
        --series-matrix external_data/GSE96058-GPL11154_series_matrix.txt.gz \
                        external_data/GSE96058-GPL18573_series_matrix.txt.gz \
        --expression external_data/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz \
        --out external_data/scanb_harmonized.csv

Both commands print and save a coverage report (``<out>.report.txt``).  The
parsers match column names tolerantly; if a field is reported as missing,
please check the report and adjust the candidate lists below.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from hetero_data import ALL_CLINICAL_FEATURES, load_metabric, node_group_from_count, npi_formula

NONSYNONYMOUS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del",
    "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
}


# =====================================================================
# Generic helpers
# =====================================================================


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else open(
        path, "r", encoding="utf-8", errors="replace")


def read_cbioportal_table(path: Path) -> pd.DataFrame:
    """cBioPortal clinical / data files (comment lines start with '#')."""
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def find_column(df: pd.DataFrame, candidates: Sequence[str], patterns: Sequence[str] = ()) -> Optional[str]:
    upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if rx.search(c):
                return c
    return None


def to_binary(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index)
    pos = s.str.match(r"^(positive|pos|yes|1|1\.0|true|\+)$")
    neg = s.str.match(r"^(negative|neg|no|0|0\.0|false|-)$")
    out[pos] = 1.0
    out[neg] = 0.0
    return out


def parse_ajcc_stage(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.replace("STAGE", "", regex=False).str.strip()
    out = pd.Series(np.nan, index=series.index)
    out[s.str.match(r"^IV")] = 4.0
    out[s.str.match(r"^III")] = 3.0
    out[s.str.match(r"^II(?!I)")] = 2.0
    out[s.str.match(r"^I(?!I|V)")] = 1.0
    return out


def parse_pn(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    out = pd.Series(np.nan, index=series.index)
    out[s.str.match(r"^N0")] = 0.0
    out[s.str.match(r"^N1")] = 1.0
    out[s.str.match(r"^N[23]")] = 2.0
    return out


def coverage_report(df: pd.DataFrame, name: str) -> str:
    lines = [f"Harmonised cohort: {name}", f"patients: {len(df)}",
             f"deaths: {int(df['event'].sum())}  median follow-up (months, reverse KM not applied): "
             f"{df['time_months'].median():.1f}",
             f"expression genes: {sum(c.startswith('expr:') for c in df.columns)}",
             f"mutation genes: {sum(c.startswith('mut:') for c in df.columns)}", "", "clinical coverage:"]
    for col in ALL_CLINICAL_FEATURES:
        cov = df[col].notna().mean() if col in df.columns else 0.0
        lines.append(f"  {col:<22s} {cov:6.1%}")
    if "pam50" in df.columns:
        lines.append("")
        lines.append("PAM50: " + ", ".join(f"{k}={v}" for k, v in df["pam50"].value_counts().items()))
    return "\n".join(lines)


def _finish(df: pd.DataFrame, out: Path, name: str) -> pd.DataFrame:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    time = pd.to_numeric(df["time_months"], errors="coerce")
    event = pd.to_numeric(df["event"], errors="coerce")
    keep = (time > 0) & np.isfinite(time) & event.isin([0, 1])
    print(f"Endpoint QC: {len(df)} input, {int((~keep).sum())} missing/invalid endpoint rows excluded")
    df = df.loc[keep].reset_index(drop=True)
    if df['patient_id'].duplicated().any():
        raise ValueError('Duplicate patient IDs after harmonisation')
    df.to_csv(out, index=False)
    report = coverage_report(df, name)
    Path(str(out) + ".report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved {out}")
    return df


def metabric_gene_sets():
    table = load_metabric()
    return set(table.expression.columns), set(table.mutation.columns) if table.mutation is not None else set()


# =====================================================================
# TCGA-BRCA (cBioPortal)
# =====================================================================


def _first_existing(folder: Path, names: Sequence[str]) -> Optional[Path]:
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None


def prepare_tcga(pancan_dir: Path, out: Path, legacy_dir: Optional[Path] = None) -> pd.DataFrame:
    pancan_dir = Path(pancan_dir)
    expr_genes, mut_genes = metabric_gene_sets()
    patient = read_cbioportal_table(pancan_dir / "data_clinical_patient.txt")
    sample = read_cbioportal_table(pancan_dir / "data_clinical_sample.txt")
    clin = patient.copy()
    pid = find_column(clin, ["PATIENT_ID"])
    clin = clin.rename(columns={pid: "PATIENT_ID"})
    s_pid = find_column(sample, ["PATIENT_ID"])
    sample = sample.rename(columns={s_pid: "PATIENT_ID"})
    sid=find_column(sample,['SAMPLE_ID'])
    if sid is not None:
        sample=sample[sample[sid].astype(str).str.match(r'^TCGA-..-....-01')].sort_values(['PATIENT_ID',sid])
    sample_first = sample.drop_duplicates("PATIENT_ID")
    clin = clin.merge(sample_first, on="PATIENT_ID", how="left", suffixes=("", "_sample"))
    if legacy_dir is not None and Path(legacy_dir).exists():
        legacy = read_cbioportal_table(Path(legacy_dir) / "data_clinical_patient.txt")
        lpid = find_column(legacy, ["PATIENT_ID"])
        legacy = legacy.rename(columns={lpid: "PATIENT_ID"})
        keep = [c for c in legacy.columns if c == "PATIENT_ID" or c.upper() not in {x.upper() for x in clin.columns}]
        clin = clin.merge(legacy[keep], on="PATIENT_ID", how="left")
        legacy_sample = Path(legacy_dir) / "data_clinical_sample.txt"
        if legacy_sample.exists():
            ls = read_cbioportal_table(legacy_sample)
            lspid = find_column(ls, ["PATIENT_ID"])
            ls = ls.rename(columns={lspid: "PATIENT_ID"}).drop_duplicates("PATIENT_ID")
            keep = [c for c in ls.columns if c == "PATIENT_ID" or c.upper() not in {x.upper() for x in clin.columns}]
            clin = clin.merge(ls[keep], on="PATIENT_ID", how="left")

    found: Dict[str, Optional[str]] = {}

    def col(key, cands, pats=()):
        c = find_column(clin, cands, pats)
        found[key] = c
        return clin[c] if c is not None else pd.Series(np.nan, index=clin.index)

    out_df = pd.DataFrame({"patient_id": clin["PATIENT_ID"].astype(str)})
    os_months = pd.to_numeric(col("os_months", ["OS_MONTHS"]), errors="coerce")
    os_status = col("os_status", ["OS_STATUS"]).astype(str).str.upper()
    out_df["time_months"] = os_months
    dead=os_status.str.contains('DECEASED|DEAD') | os_status.str.startswith('1')
    alive=os_status.str.contains('LIVING|ALIVE') | os_status.str.startswith('0')
    out_df["event"] = np.select([dead,alive],[1.,0.],default=np.nan)
    out_df["pam50"] = col("pam50", ["SUBTYPE", "PAM50_SUBTYPE", "PAM50"]).astype(str)
    out_df["cohort"] = np.nan
    out_df["age_at_diagnosis"] = pd.to_numeric(col("age", ["AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS"]), errors="coerce")
    out_df["stage"] = parse_ajcc_stage(col("stage", ["AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_PATHOLOGIC_STAGE", "TUMOR_STAGE"]))
    node_count = pd.to_numeric(col("node_count", ["NUMBER_OF_LYMPHNODES_POSITIVE_BY_HE",
                                                  "LYMPH_NODES_POSITIVE", "LYMPH_NODE_POSITIVE_COUNT"]), errors="coerce")
    pn = parse_pn(col("pn", ["PATH_N_STAGE", "AJCC_NODES_PATHOLOGIC_PN", "PATHOLOGIC_N"]))
    ng = node_group_from_count(node_count)
    out_df["node_group"] = ng.where(ng.notna(), pn)
    out_df["lymph_nodes_positive"] = node_count
    out_df["er_status"] = to_binary(col("er", ["ER_STATUS_BY_IHC", "ER_STATUS", "BREAST_CARCINOMA_ESTROGEN_RECEPTOR_STATUS"]))
    out_df["pr_status"] = to_binary(col("pr", ["PR_STATUS_BY_IHC", "PR_STATUS", "BREAST_CARCINOMA_PROGESTERONE_RECEPTOR_STATUS"]))
    ihc = col("her2_ihc", ["IHC_HER2", "HER2_STATUS_BY_IHC", "LAB_PROC_HER2_NEU_IMMUNOHISTOCHEMISTRY_RECEPTOR_STATUS"]).astype(str).str.lower()
    fish = col("her2_fish", ["HER2_FISH_STATUS", "LAB_PROCEDURE_HER2_NEU_IN_SITU_HYBRID_OUTCOME_TYPE"]).astype(str).str.lower()
    her2 = pd.Series(np.nan, index=clin.index)
    her2[ihc.str.startswith("neg")] = 0.0
    her2[ihc.str.startswith("equiv") & fish.str.startswith("neg")] = 0.0
    her2[ihc.str.startswith("pos") | fish.str.startswith("pos")] = 1.0
    out_df["her2_status"] = her2
    meno = col("menopause", ["MENOPAUSE_STATUS"]).astype(str).str.lower()
    out_df["menopausal_status"] = np.where(meno.str.startswith("post"), 1.0,
                                           np.where(meno.str.startswith("pre") | meno.str.startswith("peri"), 0.0, np.nan))
    hist = col("histology", ["CANCER_TYPE_DETAILED", "HISTOLOGICAL_DIAGNOSIS", "ONCOTREE_CODE"]).astype(str).str.lower()
    out_df["histology_type"] = np.where(hist.str.contains("lobular") | (hist == "ilc"), 1.0,
                                        np.where(hist.isin(["nan", "none", ""]), np.nan, 0.0))
    out_df["radiotherapy"] = to_binary(col("radiotherapy", ["RADIATION_THERAPY", "RADIOTHERAPY"]))
    for c in ("tumor_size", "grade", "npi", "chemotherapy", "hormone_therapy", "surgery_type", "mutation_count_log1p"):
        out_df[c] = np.nan

    # ---- expression (RSEM -> log2(x+1)) ----
    expr_path = _first_existing(pancan_dir, ["data_mrna_seq_v2_rsem.txt", "data_RNA_Seq_v2_expression_median.txt"])
    if expr_path is None:
        raise FileNotFoundError("RSEM expression file not found in the PanCancer Atlas folder")
    chunks = []
    for chunk in pd.read_csv(expr_path, sep="\t", comment="#", chunksize=2000, low_memory=False):
        chunk = chunk[chunk["Hugo_Symbol"].astype(str).str.upper().isin(expr_genes)]
        chunks.append(chunk)
    expr = pd.concat(chunks)
    expr["Hugo_Symbol"] = expr["Hugo_Symbol"].astype(str).str.upper()
    expr = expr.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in expr.columns])
    expr = expr.groupby("Hugo_Symbol").mean(numeric_only=True)  # duplicates averaged
    expr = np.log2(expr.clip(lower=0) + 1.0)
    # Both full TCGA aliquot barcodes and cBioPortal's shortened 15-character
    # primary-tumour sample IDs are accepted; non-primary samples are excluded.
    samples = [c for c in expr.columns if re.match(r"^TCGA-..-....-01", c)]
    expr = expr[samples].T
    expr.index = [s[:12] for s in expr.index]
    expr = expr[~expr.index.duplicated(keep="first")]
    expr.columns = [f"expr:{g}" for g in expr.columns]

    # ---- mutations (non-synonymous, METABRIC panel genes) ----
    mut_df = None
    maf = _first_existing(pancan_dir, ["data_mutations.txt", "data_mutations_extended.txt"])
    if maf is not None:
        m = pd.read_csv(maf, sep="\t", comment="#", low_memory=False,
                        usecols=lambda c: c in {"Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"})
        m = m[m["Variant_Classification"].isin(NONSYNONYMOUS)]
        m["patient"] = m["Tumor_Sample_Barcode"].astype(str).str[:12]
        m["gene"] = m["Hugo_Symbol"].astype(str).str.upper()
        m = m[m["gene"].isin(mut_genes)]
        mutated = m.groupby(["patient", "gene"]).size().unstack(fill_value=0).clip(upper=1)
        sequenced = None
        case_list = pancan_dir / "case_lists" / "cases_sequenced.txt"
        if case_list.exists():
            text = case_list.read_text(encoding="utf-8", errors="replace")
            ids = re.findall(r"TCGA-..-....-0\d", text)
            sequenced = {i[:12] for i in ids}
        patients = out_df["patient_id"].tolist()
        mut_df = pd.DataFrame(0.0, index=patients, columns=sorted(mut_genes))
        common = mutated.index.intersection(mut_df.index)
        for g in mutated.columns:
            mut_df.loc[common, g] = mutated.loc[common, g].astype(float)
        if sequenced is not None:
            mut_df.loc[~mut_df.index.isin(sequenced), :] = np.nan
        mut_df.columns = [f"mut:{g}" for g in mut_df.columns]

    df = out_df.set_index("patient_id").join(expr, how="inner")
    if mut_df is not None:
        df = df.join(mut_df, how="left")
    df = df.reset_index().rename(columns={"index": "patient_id"})
    print("TCGA field mapping:")
    for k, v in found.items():
        print(f"  {k:<12s} <- {v}")
    return _finish(df, out, "TCGA-BRCA")


# =====================================================================
# SCAN-B (GSE96058)
# =====================================================================


def parse_series_matrix(path: Path) -> pd.DataFrame:
    titles: List[str] = []
    accessions: List[str] = []
    chars: Dict[str, List[str]] = {}
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                accessions = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                values = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                key = None
                for v in values:
                    if ":" in v:
                        key = v.split(":", 1)[0].strip().lower()
                        break
                if key is None:
                    continue
                parsed = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in values]
                chars.setdefault(key, parsed)
            elif line.startswith("!series_matrix_table_begin"):
                break
    df = pd.DataFrame(chars)
    df.insert(0, "title", titles[: len(df)] if titles else np.arange(len(df)).astype(str))
    if accessions:
        df.insert(1, "geo_accession", accessions[: len(df)])
    return df


def _char(df: pd.DataFrame, *patterns: str) -> pd.Series:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if rx.search(c):
                return df[c]
    return pd.Series(np.nan, index=df.index)


def prepare_scanb(series_matrix: Sequence[Path], expression: Path, out: Path) -> pd.DataFrame:
    expr_genes, _ = metabric_gene_sets()
    meta = pd.concat([parse_series_matrix(p) for p in series_matrix], ignore_index=True)
    print("SCAN-B characteristic keys:", ", ".join(c for c in meta.columns if c not in {"title", "geo_accession"}))
    meta = meta[~meta["title"].astype(str).str.contains("repl", case=False)]
    ext_id = _char(meta, r"external id", r"scan-b")
    meta = meta.assign(_ext=ext_id.astype(str))
    meta = meta[~(meta["_ext"].duplicated(keep="first") & meta["_ext"].ne("nan"))]

    def num(s):
        return pd.to_numeric(s.replace({"NA": np.nan, "na": np.nan}), errors="coerce")

    out_df = pd.DataFrame({"patient_id": meta["title"].astype(str).to_numpy()})
    days = num(_char(meta, r"overall survival days", r"os days", r"overall survival")).to_numpy()
    out_df["time_months"] = days / 30.4375
    out_df["event"] = num(_char(meta, r"overall survival event", r"os event")).to_numpy()
    out_df["pam50"] = _char(meta, r"pam50").astype(str).to_numpy()
    out_df["cohort"] = np.nan
    out_df["age_at_diagnosis"] = num(_char(meta, r"^age")).to_numpy()
    size = num(_char(meta, r"tumor size", r"tumour size", r"^size"))
    if size.median() < 8:  # recorded in cm
        size = size * 10.0
    out_df["tumor_size"] = size.to_numpy()
    lng = _char(meta, r"lymph node group").astype(str).str.lower()
    ng = pd.Series(np.nan, index=lng.index)
    ng[lng.str.contains("negative") | lng.str.contains("submicro")] = 0.0
    ng[lng.str.contains("1to3")] = 1.0
    ng[lng.str.contains("4to")] = 2.0
    out_df["node_group"] = ng.to_numpy()
    out_df["grade"] = num(_char(meta, r"^nhg$", r"nhg", r"grade").astype(str).str.extract(r"(\d)")[0]).to_numpy()
    out_df["er_status"] = to_binary(_char(meta, r"^er status")).to_numpy()
    out_df["pr_status"] = to_binary(_char(meta, r"^pgr status", r"^pr status")).to_numpy()
    out_df["her2_status"] = to_binary(_char(meta, r"^her2 status")).to_numpy()
    out_df["chemotherapy"] = to_binary(_char(meta, r"chemo")).to_numpy()
    out_df["hormone_therapy"] = to_binary(_char(meta, r"endocrine")).to_numpy()
    out_df["npi"] = npi_formula(out_df["tumor_size"], out_df["grade"], out_df["node_group"]).to_numpy()
    for c in ("lymph_nodes_positive", "stage", "radiotherapy", "menopausal_status", "histology_type",
              "surgery_type", "mutation_count_log1p"):
        out_df[c] = np.nan

    rows = []
    for chunk in pd.read_csv(expression, chunksize=2000, low_memory=False):
        gene_col = chunk.columns[0]
        chunk = chunk[chunk[gene_col].astype(str).str.upper().isin(expr_genes)]
        rows.append(chunk)
    expr = pd.concat(rows)
    gene_col = expr.columns[0]
    expr[gene_col] = expr[gene_col].astype(str).str.upper()
    expr = expr.groupby(gene_col).mean(numeric_only=True).T  # samples x genes (already log2 FPKM)
    expr.index = expr.index.astype(str)
    expr.columns = [f"expr:{g}" for g in expr.columns]
    df = out_df.set_index("patient_id").join(expr, how="inner").reset_index()
    df = df.rename(columns={"index": "patient_id"})
    if len(df) == 0:
        raise RuntimeError("No SCAN-B samples matched between the series matrix titles and the expression columns; "
                           "check the sample naming in the report.")
    return _finish(df, out, "SCAN-B (GSE96058)")


# =====================================================================
# Synthetic cohort for smoke tests
# =====================================================================


def make_synthetic_external(out: Path, kind: str = "scanb_like", seed: int = 0) -> pd.DataFrame:
    """METABRIC cohorts 4-5 re-expressed in the harmonised schema with fields removed."""
    table = load_metabric()
    idx = np.where(np.isin(table.cohorts, [4.0, 5.0]))[0]
    df = table.subset(idx).to_harmonized_frame()
    rng = np.random.default_rng(seed)
    expr_cols = [c for c in df.columns if c.startswith("expr:")]
    df[expr_cols] = df[expr_cols] * 1.5 + 2.0 + rng.normal(scale=0.1, size=(len(df), len(expr_cols)))
    if kind == "scanb_like":
        for c in ("stage", "radiotherapy", "menopausal_status", "histology_type", "surgery_type",
                  "mutation_count_log1p", "lymph_nodes_positive"):
            df[c] = np.nan
        df = df.drop(columns=[c for c in df.columns if c.startswith("mut:")])
    else:  # tcga_like
        for c in ("tumor_size", "grade", "npi", "chemotherapy", "hormone_therapy", "surgery_type"):
            df[c] = np.nan
        df = df.drop(columns=expr_cols[: len(expr_cols) // 4])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tcga")
    t.add_argument("--pancan-dir", required=True)
    t.add_argument("--legacy-dir", default=None)
    t.add_argument("--out", default="external_data/tcga_brca_harmonized.csv")
    s = sub.add_parser("scanb")
    s.add_argument("--series-matrix", nargs="+", required=True)
    s.add_argument("--expression", required=True)
    s.add_argument("--out", default="external_data/scanb_harmonized.csv")
    y = sub.add_parser("synthetic")
    y.add_argument("--kind", choices=["scanb_like", "tcga_like"], default="scanb_like")
    y.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "tcga":
        prepare_tcga(Path(args.pancan_dir), Path(args.out), Path(args.legacy_dir) if args.legacy_dir else None)
    elif args.cmd == "scanb":
        prepare_scanb([Path(x) for x in args.series_matrix], Path(args.expression), Path(args.out))
    else:
        make_synthetic_external(Path(args.out), args.kind)


if __name__ == "__main__":
    main()
