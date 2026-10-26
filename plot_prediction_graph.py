"""Figure 5: schematic of the KG-MSPF prediction graph (revised for legibility).

Designed for a printed width of ~7.2 in (double column): node labels are
>= 8 pt at print size.  Only genes that are present in the METABRIC
expression panel are shown (ESR1, PGR, FOXA1, CD8A and others are absent from
the public table and therefore never enter the graph).

    python plot_prediction_graph.py [--out outputs/revision/manuscript/figures/fig_prediction_graph.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from config import FIGURE_DPI, OUTPUT_DIR
from hetero_data import CLINICAL_PATHWAY_MAP, PATHWAY_GENE_MAP, SUBTYPE_PATHWAY_MAP, load_metabric

TYPE_COLORS = {
    "gene": "#D6EAF8", "pathway": "#E8DAEF", "clinical": "#FDEBD0", "subtype": "#D1F2EB", "patient": "#F5B7B1",
}
EDGE_COLORS = {"gene": "#2E86C1", "pathway": "#7D3C98", "clinical": "#CA6F1E", "subtype": "#17A589"}

PATHWAYS_SHOWN = ("ER_signaling", "HER2_ERBB", "PI3K_AKT_mTOR", "p53_DNA_repair", "Cell_cycle", "Immune")
GENES_PREFERRED = {
    "ER_signaling": ("GATA3", "TBX3"),
    "HER2_ERBB": ("ERBB2", "ERBB3"),
    "PI3K_AKT_mTOR": ("PIK3CA", "PTEN"),
    "p53_DNA_repair": ("TP53", "BRCA1"),
    "Cell_cycle": ("CCND1", "CDK4"),
    "Immune": ("STAT1", "JAK2"),
}
CLINICAL_SHOWN = ("er_status", "her2_status", "grade", "stage", "lymph_nodes_positive", "chemotherapy", "npi")
CLINICAL_LABELS = {
    "er_status": "ER status", "her2_status": "HER2 status", "grade": "Grade", "stage": "Stage",
    "lymph_nodes_positive": "Positive nodes", "chemotherapy": "Chemotherapy", "npi": "NPI",
}
PATHWAY_LABELS = {
    "ER_signaling": "ER signalling", "HER2_ERBB": "HER2/ERBB", "PI3K_AKT_mTOR": "PI3K/AKT/mTOR",
    "p53_DNA_repair": "p53/DNA repair", "Cell_cycle": "Cell cycle", "Immune": "Immune",
}
SUBTYPE_LABELS = {"Luminal_A": "Luminal A", "Luminal_B": "Luminal B", "HER2_enriched": "HER2-enriched",
                  "Basal_like": "Basal-like", "Other": "Other"}


def _box(ax, xy, text, ntype, width=1.25, height=0.42, fontsize=8.5, bold=False):
    x, y = xy
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.12", fc=TYPE_COLORS[ntype],
                           ec="#566573", lw=0.8, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4,
            fontweight="bold" if bold else "normal")


def _line(ax, a, b, color, style="-", lw=0.9, alpha=0.85, arrow=False, rad=0.0):
    ax.annotate("", xy=b, xytext=a, zorder=1,
                arrowprops=dict(arrowstyle="-|>" if arrow else "-", color=color, lw=lw, alpha=alpha, ls=style,
                                shrinkA=0, shrinkB=0, mutation_scale=8, connectionstyle=f"arc3,rad={rad}"))


def plot_prediction_graph(path: Path, qa_hook=None) -> None:
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    panel = set(load_metabric().expression.columns)
    genes = []
    for p in PATHWAYS_SHOWN:
        for g in GENES_PREFERRED[p]:
            if g in panel and g in PATHWAY_GENE_MAP[p]:
                genes.append((g, p))

    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    xg, xp, xc, xpt = 0.0, 2.25, 4.55, 6.75
    wg, wp, wc, wpt = 1.05, 1.55, 1.45, 1.35
    gene_y = {g: 5.2 - i * (10.4 / max(len(genes) - 1, 1)) for i, (g, _) in enumerate(genes)}
    pw_y = {p: 4.68 - i * (9.36 / (len(PATHWAYS_SHOWN) - 1)) for i, p in enumerate(PATHWAYS_SHOWN)}
    clin_y = {c: 5.2 - i * 0.74 for i, c in enumerate(CLINICAL_SHOWN)}
    subtypes = list(SUBTYPE_LABELS)
    sub_y = {s: -1.3 - i * 0.74 for i, s in enumerate(subtypes)}
    patient = (xpt, 0.6)

    for g, p in genes:
        _line(ax, (xg + wg / 2, gene_y[g]), (xp - wp / 2, pw_y[p]), EDGE_COLORS["gene"], lw=1.1)
    for c in CLINICAL_SHOWN:
        for p in CLINICAL_PATHWAY_MAP.get(c, ()):
            if p in pw_y:
                _line(ax, (xc - wc / 2, clin_y[c]), (xp + wp / 2, pw_y[p]), EDGE_COLORS["clinical"], style="--", lw=0.9)
    for s in subtypes:
        for p in SUBTYPE_PATHWAY_MAP.get(s, ()):
            if p in pw_y:
                _line(ax, (xc - wc / 2, sub_y[s]), (xp + wp / 2, pw_y[p]), EDGE_COLORS["subtype"], style=":", lw=1.3)
    agg = "#909497"
    for y in list(clin_y.values()) + list(sub_y.values()):
        _line(ax, (xc + wc / 2, y), (xpt - wpt / 2, patient[1]), agg, lw=0.7, alpha=0.7)
    # bundled aggregation from all gene and pathway nodes (drawn once for readability)
    # Route aggregate buses around the clinical labels, rather than behind
    # their filled boxes, so vector geometry remains clear and unambiguous.
    _line(ax,(xg,6.35),(xpt,6.35),agg,lw=1.3,alpha=.8)
    _line(ax,(xpt,6.35),(xpt,patient[1]+.45),agg,lw=1.3,alpha=.8,arrow=True)
    _line(ax,(xp,5.8),(xpt-.25,5.8),agg,lw=1.3,alpha=.8)
    _line(ax,(xpt-.25,5.8),(xpt-.25,patient[1]+.45),agg,lw=1.3,alpha=.8,arrow=True)
    ax.text(xg+.15,6.42,"all genes -> patient",ha='left',va='bottom',fontsize=7.5,color='#616A6B')
    ax.text(xp+.15,5.87,"all pathways -> patient",ha='left',va='bottom',fontsize=7.5,color='#616A6B')

    for g, _ in genes:
        _box(ax, (xg, gene_y[g]), g, "gene", width=wg, fontsize=8.5)
    for p, y in pw_y.items():
        _box(ax, (xp, y), PATHWAY_LABELS[p], "pathway", width=wp, fontsize=8.5)
    for c, y in clin_y.items():
        _box(ax, (xc, y), CLINICAL_LABELS[c], "clinical", width=wc, fontsize=8.5)
    for s, y in sub_y.items():
        _box(ax, (xc, y), SUBTYPE_LABELS[s], "subtype", width=wc, fontsize=8.5)
    _box(ax, patient, "Patient\n(risk readout)", "patient", width=wpt, height=0.85, fontsize=9, bold=True)

    for x, title in ((xg, "Selected genes"), (xp, "Selected pathways"), (xc, "Clinical variables")):
        ax.text(x, 6.75, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.text(xc, -0.75, "PAM50 subtypes", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    handles = [Patch(fc=TYPE_COLORS[k], ec="#566573", label=k.capitalize() + " node") for k in TYPE_COLORS]
    handles += [
        Line2D([0], [0], color=EDGE_COLORS["gene"], lw=1.2, label="gene-pathway membership"),
        Line2D([0], [0], color=EDGE_COLORS["clinical"], lw=1.2, ls="--", label="clinical-pathway association"),
        Line2D([0], [0], color=EDGE_COLORS["subtype"], lw=1.4, ls=":", label="subtype-pathway association"),
        Line2D([0], [0], color=agg, lw=1.2, label="aggregation into patient node"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=7.5, frameon=False, ncol=3)
    ax.set_xlim(-0.75, 7.5)
    ax.set_ylim(-5.6, 7.15)
    ax.axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if qa_hook is not None:
        qa_hook(fig,path)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(OUTPUT_DIR) / "revision" / "manuscript" / "figures" / "fig_prediction_graph.png"))
    plot_prediction_graph(Path(p.parse_args().out))


if __name__ == "__main__":
    main()
