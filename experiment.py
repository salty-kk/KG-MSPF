"""Experiment engine for KG-MSPF: matched splits, model fitting, prediction storage and metrics."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import tempfile
import zipfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import OUTPUT_DIR, TRAIN_CHUNK_SIZE
from hetero_data import (
    CohortTable,
    FoldPreprocessor,
    GraphConfig,
    GraphDataset,
    choose_transportable_features,
    load_metabric,
)
from survival_utils import evaluate_predictions

REVISION_DIR = Path(OUTPUT_DIR) / "revision"
TIME_GRID = np.arange(6.0, 181.0, 6.0)          # months, IBS / calibration grid
AUC_TIMES = (12, 36, 60, 120)
BRIER_TIMES = (36, 60, 120)
TAU = 180.0

MAIN_MODELS: Tuple[str, ...] = (
    "KG-MSPF", "Clinical-Cox", "Clinical+PAM50-Cox", "FullConcat-Cox", "FullConcat-LASSO-Cox",
    "RSF", "DeepSurv", "DeepHit", "SurvTRACE", "Cox-PASNet", "HomoGAT",
)

# Derived blends computed from every KG-MSPF fit (no re-training needed)
DERIVED_BLENDS: Dict[str, Tuple[str, ...]] = {
    "neural_only": ("heterogat",),
    "anchors_only": ("clinical_cox", "subtype_cox", "full_cox"),
    "no_clinical_cox": ("subtype_cox", "full_cox", "heterogat"),
    "no_subtype_cox": ("clinical_cox", "full_cox", "heterogat"),
    "no_full_cox": ("clinical_cox", "subtype_cox", "heterogat"),
}


@dataclass
class Settings:
    epochs: int = 160
    patience: int = 30
    clinical_warmup: int = 0
    top_k: int = 30
    max_genes: int = 90
    hidden_dim: int = 16
    heads: int = 2
    layers: int = 1
    dropout: float = 0.3
    prior_strength: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    subtype_penalty: float = 1e-3
    validation_fraction: float = 0.25
    device: str = "auto"
    verbose: bool = False
    base_seed: int = 42
    smoke: bool = False
    strict: bool = False
    rsf_trees: int = 300
    interpretability: bool = True
    save_checkpoints: bool = False
    train_chunk_size: int = TRAIN_CHUNK_SIZE
    # Frozen small-model development choice; baseline schedules are independent.
    graph_scale_init: float = 0.0
    graph_head_gain: float = 0.1
    graph_aux_weight: float = 0.0
    min_epochs: int = 20
    gradient_mode: str = "auto"
    inference_batch_size: int = 128

    def seed_for(self, repeat: int) -> int:
        return self.base_seed + 1000 * int(repeat)


# =====================================================================
# Splits
# =====================================================================


def stratification_labels(table: CohortTable) -> np.ndarray:
    return np.asarray([f"{s}_{e}" for s, e in zip(table.pam50, table.events)])


def make_splits(table: CohortTable, n_repeats: int, n_folds: int = 5, seed: int = 42,
                subset: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    labels = stratification_labels(table)
    base = np.arange(len(table)) if subset is None else np.asarray(subset)
    splits = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(base.size), labels[base]), start=1):
            splits.append({"name": f"r{r}_f{f}", "repeat": r, "fold": f,
                           "train": base[tr], "test": base[te]})
    return splits


def holdout_split(table: CohortTable) -> Dict[str, object]:
    tr = np.where(np.isin(table.cohorts, [1.0, 2.0, 3.0]))[0]
    te = np.where(np.isin(table.cohorts, [4.0, 5.0]))[0]
    return {"name": "cohort_holdout", "repeat": 0, "fold": 0, "train": tr, "test": te}


# =====================================================================
# Result store
# =====================================================================


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _base(self, experiment: str, config: str, split: str) -> Path:
        safe = config.replace("/", "_").replace(" ", "_")
        return self.root / experiment / safe / split

    def exists(self, experiment: str, config: str, split: str) -> bool:
        base = self._base(experiment, config, split)
        if not base.with_suffix(".json").exists() or not base.with_suffix(".npz").exists():
            return False
        try:
            meta = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(base.with_suffix('.npz'), allow_pickle=False) as arrays:
                if 'risk' not in arrays or not np.isfinite(arrays['risk']).all():
                    return False
                expected = meta.get('arrays_sha256')
                if expected and hashlib.sha256(base.with_suffix('.npz').read_bytes()).hexdigest() != expected:
                    return False
        except (OSError, ValueError, KeyError,zipfile.BadZipFile):
            return False
        return True

    def ensure_protocol(self, experiment, settings, table, external=None, preproc_kwargs=None, eval_kwargs=None, split=None):
        """Refuse to silently reuse results from different code/data/settings."""
        h = hashlib.sha256()
        for name in ("experiment.py", "hetero_data.py", "model_api.py", "model_heterogat.py", "baselines.py", "survival_utils.py",
                     "config.py","run_experiments.py","external_data.py","optimizers.py"):
            h.update((Path(__file__).parent / name).read_bytes())
        for cohort in (table, external):
            if cohort is None:
                continue
            for frame in (cohort.clinical, cohort.expression, cohort.mutation):
                if frame is not None:
                    h.update(str(list(frame.columns)).encode())
                    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
            for values in (cohort.patient_ids, cohort.times, cohort.events, cohort.pam50, cohort.cohorts):
                h.update(pd.util.hash_array(np.asarray(values)).tobytes())
        versions = {}
        for pkg in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-survival', 'torch','torch-directml'):
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = None
        payload = {"settings": asdict(settings), "code_and_data_sha256": h.hexdigest(), "versions": versions,
                   "execution_environment":{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','KGMSPF_BASELINE_CHUNK','KGMSPF_RSF_JOBS')},
                   "preprocessing": preproc_kwargs or {}, "evaluation": eval_kwargs or {}}
        encoded = json.dumps(payload, sort_keys=True, default=str)
        path = self.root / experiment / "protocol.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != json.loads(encoded):
                raise ValueError(f"Protocol changed for {experiment}; choose a NEW --out directory")
        else:
            if list(path.parent.glob("*/*.json")):
                raise ValueError(f"Unversioned cache in {experiment}; choose a NEW --out directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded, encoding="utf-8")
        if split is not None:
            signature = hashlib.sha256(np.asarray(split['train'], dtype=np.int64).tobytes()
                                       + np.asarray(split['test'], dtype=np.int64).tobytes()).hexdigest()
            split_path = path.parent / ('.' + split['name'] + '.sha256')
            if split_path.exists() and split_path.read_text() != signature:
                raise ValueError(f"Split changed for {experiment}/{split['name']}; choose a NEW --out directory")
            split_path.write_text(signature)

    def save(self, experiment: str, config: str, split: str, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
        base = self._base(experiment, config, split)
        base.parent.mkdir(parents=True, exist_ok=True)
        # Write arrays completely, then publish metadata last. Interrupted
        # writes and stale array/metadata pairs are detected on resume.
        fd, temporary = tempfile.mkstemp(dir=base.parent, suffix='.npz')
        os.close(fd)
        try:
            np.savez_compressed(temporary, **arrays)
            metadata = dict(meta)
            metadata['arrays_sha256'] = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
            os.replace(temporary, base.with_suffix('.npz'))
            def clean(value):
                value = _to_jsonable(value)
                if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
                if isinstance(value, list): return [clean(v) for v in value]
                if isinstance(value, float) and not np.isfinite(value): return None
                return value
            fd, temporary = tempfile.mkstemp(dir=base.parent, suffix='.json')
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(clean(metadata), fh, indent=1, allow_nan=False)
            os.replace(temporary, base.with_suffix('.json'))
        finally:
            if Path(temporary).exists(): Path(temporary).unlink()

    def load(self, experiment: str, config: str, split: str):
        base = self._base(experiment, config, split)
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        arrays = dict(np.load(base.with_suffix(".npz"), allow_pickle=False))
        return arrays, meta

    def records(self, experiment: str) -> List[Dict]:
        out = []
        root = self.root / experiment
        if not root.exists():
            return out
        for js in sorted(root.glob("*/*.json")):
            meta = json.loads(js.read_text(encoding="utf-8"))
            meta["_npz"] = str(js.with_suffix(".npz"))
            out.append(meta)
        return out


# =====================================================================
# Model factory
# =====================================================================


def kgmspf_estimator(settings: Settings, seed: int, **overrides):
    from model_heterogat import COMPONENTS, HeteroGATSurvivalEstimator

    kwargs = dict(
        hidden_dim=settings.hidden_dim, n_heads=settings.heads, n_layers=settings.layers,
        dropout=settings.dropout, prior_strength=settings.prior_strength, learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay, subtype_penalty=settings.subtype_penalty, max_epochs=settings.epochs,
        patience=settings.patience, validation_fraction=settings.validation_fraction,
        clinical_warmup_epochs=settings.clinical_warmup, refit_full=True, seed=seed, device=settings.device,
        verbose=settings.verbose, blend_components=COMPONENTS,
        train_chunk_size=settings.train_chunk_size,
        graph_scale_init=settings.graph_scale_init, graph_head_gain=settings.graph_head_gain,
        graph_aux_weight=settings.graph_aux_weight, min_epochs=settings.min_epochs,
        gradient_mode=settings.gradient_mode, inference_batch_size=settings.inference_batch_size,
    )
    kwargs.update(overrides)
    return HeteroGATSurvivalEstimator(**kwargs)


def build_model(name: str, settings: Settings, seed: int, extra_blends: Optional[Dict] = None):
    import model_api as api

    if name == "Clinical-Cox":
        return api.CoxBaseline(name, blocks=("clinical",), seed=seed)
    if name == "Clinical+PAM50-Cox":
        return api.CoxBaseline(name, blocks=("clinical", "subtype"), seed=seed)
    if name == "FullConcat-Cox":
        return api.CoxBaseline(name, blocks=None, seed=seed)
    if name == "FullConcat-LASSO-Cox":
        return api.CoxBaseline(name, blocks=None, penalty="lasso", seed=seed)
    if name == "KG-MSPF":
        est = kgmspf_estimator(settings, seed, extra_blends=extra_blends or {})
        return api.EstimatorWrapper("KG-MSPF", est)
    import baselines as bl

    torch_kwargs = dict(seed=seed, device=settings.device, verbose=settings.verbose,
                        validation_fraction=settings.validation_fraction)
    if settings.smoke:
        torch_kwargs.update(max_epochs=4, patience=2)
    if name == "RSF":
        return bl.RSFBaseline(seed=seed, n_estimators=20 if settings.smoke else settings.rsf_trees,
                              leaf_grid=(15,) if settings.smoke else (5, 15, 30),
                              validation_fraction=settings.validation_fraction)
    if name == "DeepSurv":
        kw = dict(torch_kwargs)
        if settings.smoke:
            kw["grid"] = [{"hidden": (16,), "dropout": 0.1}]
        return bl.DeepSurvBaseline(**kw)
    if name == "DeepHit":
        kw = dict(torch_kwargs)
        if settings.smoke:
            kw["grid"] = [{"hidden": (16,), "dropout": 0.1, "alpha": 0.2, "sigma": 0.1}]
        return bl.DeepHitBaseline(**kw)
    if name == "SurvTRACE":
        kw = dict(torch_kwargs)
        if settings.smoke:
            kw["grid"] = [{"d": 16, "heads": 2, "layers": 1, "dropout": 0.1}]
        return bl.SurvTraceBaseline(**kw)
    if name == "Cox-PASNet":
        kw = dict(torch_kwargs)
        if settings.smoke:
            kw["grid"] = [{"hidden": 8, "dropout": 0.1}]
        return bl.CoxPASNetBaseline(**kw)
    if name == "HomoGAT":
        if settings.smoke:
            return bl.make_homogat(seed, settings.device, settings.verbose, max_epochs=4, patience=2,
                                   train_chunk_size=settings.train_chunk_size)
        return bl.make_homogat(seed, settings.device, settings.verbose,
                               train_chunk_size=settings.train_chunk_size,
                               validation_fraction=settings.validation_fraction,
                               min_epochs=60)  # Preserve the baseline schedule when KG-MSPF is tuned.
    raise ValueError(f"Unknown model {name}")


# =====================================================================
# Configurations for re-trained KG-MSPF variants
# =====================================================================


@dataclass
class VariantSpec:
    """A KG-MSPF variant: preprocessing, graph and estimator overrides."""

    description: str
    preproc: Dict[str, object] = field(default_factory=dict)
    graph: Optional[GraphConfig] = None
    estimator: Dict[str, object] = field(default_factory=dict)


def ablation_variants() -> Dict[str, VariantSpec]:
    from model_api import ModelFlags

    return {
        "full": VariantSpec("Full KG-MSPF"),
        "no_subtype_residual": VariantSpec("w/o subtype residual", estimator={"flags": ModelFlags(use_subtype_residual=False)}),
        "no_pathway_residual": VariantSpec("w/o pathway residual", estimator={"flags": ModelFlags(use_pathway_residual=False)}),
        "no_graph_residual": VariantSpec("w/o graph residual (no GAT branch)", estimator={"flags": ModelFlags(use_graph_residual=False)}),
        "no_message_passing": VariantSpec("w/o message passing (0 GAT layers)", estimator={"n_layers": 0}),
        "two_layers": VariantSpec("2 relational layers (selected full model uses 1)", estimator={"n_layers": 2}),
        "relation_agnostic": VariantSpec("relation-agnostic attention (one edge type)", graph=GraphConfig(relation_aware=False)),
        "no_edge_priors": VariantSpec("w/o edge priors (gamma = 0)", estimator={"flags": ModelFlags(use_priors=False)}),
        "graph_only": VariantSpec(
            "graph branch only (no clinical head/residuals/anchors)",
            estimator={"flags": ModelFlags(use_clinical_head=False, use_subtype_residual=False,
                                           use_pathway_residual=False, graph_only=True),
                       "blend_components": ("heterogat",)}),
        "no_mutations": VariantSpec("w/o somatic mutations", preproc={"use_mutations": False}),
        "molecular_graph_only": VariantSpec(
            "molecular graph only (no patient-specific clinical or PAM50 input)",
            preproc={"subtype_source": "none"},
            estimator={"flags": ModelFlags(use_clinical_head=False,use_subtype_residual=False,
                        use_pathway_residual=False,graph_only=True,mask_clinical_graph_inputs=True),
                       "blend_components": ("heterogat",)}),
        "molecular_offset": VariantSpec(
            "molecular residual trained conditional on a frozen Clinical-Cox offset (experimental)",
            preproc={"subtype_source": "none"},
            estimator={"flags": ModelFlags(use_clinical_head=False,use_subtype_residual=False,
                        use_pathway_residual=False,graph_only=True,mask_clinical_graph_inputs=True),
                       "blend_components": ("heterogat",),"clinical_offset":True}),
    }


# ---------------------------------------------------------------------
# Rescue experiments: why does the graph branch contribute ~nothing?
# ---------------------------------------------------------------------
# Diagnosis (see diagnostics/diag_rescue.py):
#   * the graph path is silenced at init -- graph_risk_head gain 0.1 AND
#     graph_scale_raw = -2.5 (sigmoid -> 0.076), i.e. ~0.0076 of the
#     log-hazard, so it receives a ~130x smaller gradient than the clinical
#     head and its gates never leave their initial values;
#   * in ``graph_only`` mode early stopping locks onto epoch 1 in 4/5 folds,
#     although the branch reaches C=0.671 in the fold that does train.
# The variants below isolate each cause.  Every knob defaults to the current
# behaviour, so ``r_base`` reproduces the submitted configuration exactly.

ES_FIX: Dict[str, object] = {"min_epochs": 60, "patience": 40, "validation_fraction": 0.25}
SCALE_FIX: Dict[str, object] = {"graph_scale_init": 0.0, "graph_head_gain": 1.0}


def rescue_variants() -> Dict[str, VariantSpec]:
    from model_api import ModelFlags

    graph_only_flags = ModelFlags(use_clinical_head=False, use_subtype_residual=False,
                                  use_pathway_residual=False, graph_only=True)
    return {
        "r_base": VariantSpec("rescue reference: unchanged configuration",
                              estimator={"graph_scale_init": -2.5, "graph_head_gain": 0.1,
                                         "graph_aux_weight": 0.0, "min_epochs": 0,
                                         "patience": 20, "validation_fraction": 0.15}),
        "r_scale": VariantSpec("graph un-attenuated (scale init 0.0, head gain 1.0)",
                               estimator=dict(SCALE_FIX)),
        "r_es": VariantSpec("early-stopping fix (min_epochs=60, patience=40, val=0.25)",
                            estimator=dict(ES_FIX)),
        "r_scale_es": VariantSpec("un-attenuated + early-stopping fix",
                                  estimator={**SCALE_FIX, **ES_FIX}),
        "r_aux": VariantSpec("auxiliary Cox loss on the graph branch",
                             estimator={"graph_aux_weight": 1.0}),
        "r_all": VariantSpec("un-attenuated + early-stopping fix + auxiliary graph loss",
                             estimator={**SCALE_FIX, **ES_FIX, "graph_aux_weight": 1.0}),
        "r_graphonly_es": VariantSpec("graph branch only, early-stopping fix",
                                      estimator={"flags": graph_only_flags,
                                                 "blend_components": ("heterogat",), **ES_FIX}),
        "r_graphonly_scale_es": VariantSpec("graph branch only, un-attenuated + early-stopping fix",
                                            estimator={"flags": graph_only_flags,
                                                       "blend_components": ("heterogat",),
                                                       **SCALE_FIX, **ES_FIX}),
    }


def prior_variants(draws: int = 3) -> Dict[str, VariantSpec]:
    out = {
        "default": VariantSpec("default tier priors"),
        "uniform": VariantSpec("uniform priors (pi = 1, prior term vanishes)", graph=GraphConfig(prior_mode="uniform")),
        "inverted": VariantSpec("inverted tier order", graph=GraphConfig(prior_mode="inverted")),
    }
    for d in range(draws):
        out[f"perturb_d{d}"] = VariantSpec(f"log-normal perturbation sigma=0.3 (draw {d})",
                                           graph=GraphConfig(prior_mode="perturb", perturb_sigma=0.3, seed=d))
        out[f"permute_d{d}"] = VariantSpec(f"randomly permuted prior values (draw {d})",
                                           graph=GraphConfig(prior_mode="permute", seed=d))
        out[f"rewired_d{d}"] = VariantSpec(f"degree-preserving rewired knowledge edges (draw {d})",
                                           graph=GraphConfig(rewire_knowledge=True, seed=100 + d))
    return out


def pam50_variants() -> Dict[str, VariantSpec]:
    from model_api import ModelFlags

    return {
        "pam50_full": VariantSpec("PAM50 as node and residual (default)"),
        "no_pam50": VariantSpec("PAM50 removed everywhere", preproc={"subtype_source": "none"},
                                estimator={"blend_components": ("clinical_cox", "full_cox", "heterogat")}),
        "pam50_nodes_only": VariantSpec(
            "PAM50 as graph node only",
            estimator={"flags": ModelFlags(use_subtype_residual=False),
                       "blend_components": ("clinical_cox", "full_cox", "heterogat")}),
        "pam50_residual_only": VariantSpec("PAM50 as residual only (no subtype nodes)",
                                           graph=GraphConfig(include_subtype_nodes=False)),
        "ihc_surrogate": VariantSpec("IHC surrogate subtype instead of PAM50", preproc={"subtype_source": "ihc"}),
    }


# =====================================================================
# Running one model on one split
# =====================================================================


def _risk_quantiles(train_risk: np.ndarray) -> np.ndarray:
    return np.percentile(train_risk, [100 / 3, 200 / 3])


def evaluate_and_pack(model, train_ds: GraphDataset, test_ds: GraphDataset, time_grid=TIME_GRID,
                      auc_times=AUC_TIMES, brier_times=BRIER_TIMES, tau=TAU, blend: Optional[str] = None):
    kwargs = {} if blend is None else {"blend": blend}
    risk = model.predict_risk(test_ds, **kwargs)
    surv = model.predict_survival(test_ds, time_grid, **kwargs)
    train_risk = model.predict_risk(train_ds, **kwargs)
    metrics = evaluate_predictions(train_ds.times, train_ds.events, test_ds.times, test_ds.events, risk, surv,
                                   time_grid, auc_times, brier_times, tau)
    arrays = {"risk": risk.astype(np.float64), "surv": surv.astype(np.float32),
              "train_risk_quantiles": _risk_quantiles(train_risk)}
    return metrics, arrays


def build_datasets(table: CohortTable, split: Dict, settings: Settings, preproc_kwargs: Optional[Dict] = None,
                   graph: Optional[GraphConfig] = None, external: Optional[CohortTable] = None):
    """Fit preprocessing on the training indices and transform train / test."""
    kwargs = dict(data_driven_top_k=settings.top_k, max_genes=settings.max_genes)
    kwargs.update(preproc_kwargs or {})
    if graph is not None:
        kwargs["graph_config"] = graph
    pre = FoldPreprocessor(**kwargs).fit(table, split["train"])
    train_ds = pre.transform(table, split["train"])
    if external is None:
        test_ds = pre.transform(table, split["test"])
    else:
        test_ds = pre.transform(external, np.arange(len(external)), expression_scaling="within")
    return pre, train_ds, test_ds


def _datasets_for_model(model, pre: FoldPreprocessor, table, split, train_ds, test_ds, external=None):
    cfg = getattr(model, "graph_config", None)
    if cfg is None:
        return train_ds, test_ds
    pre2 = pre.with_graph_config(cfg)
    tr = pre2.transform(table, split["train"])
    if external is None:
        te = pre2.transform(table, split["test"])
    else:
        te = pre2.transform(external, np.arange(len(external)), expression_scaling="within")
    return tr, te


def interpretability_payload(model, test_ds: GraphDataset) -> Dict[str, object]:
    est = getattr(model, "estimator", None)
    if est is None:
        return {}
    out: Dict[str, object] = {}
    try:
        out["attention"] = est.attention_statistics(test_ds)
        out["knockout"] = est.knockout_analysis(test_ds)
        out["branch_contributions"] = est.branch_contributions(test_ds)
    except Exception as exc:  # interpretability must never break the run
        out["error"] = repr(exc)
    return out


def prepare_selection_data(model, table, split, settings, preproc_kwargs=None, graph=None):
    """Fit gene screening/imputation/scaling without inner-validation patients.

    Final refitting still uses the outer-training preprocessor. The inner
    graph can have a different gene set; both stages build their own network.
    """
    from sklearn.model_selection import train_test_split
    if getattr(model, "penalty", None) == "lasso":
        outer = np.asarray(split["train"])
        kwargs = dict(data_driven_top_k=settings.top_k, max_genes=settings.max_genes)
        kwargs.update(preproc_kwargs or {})
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=model.seed)
        model.inner_cv_datasets = []
        for tr, va in folds.split(np.zeros(len(outer)), table.events[outer]):
            pre = FoldPreprocessor(**kwargs).fit(table, outer[tr])
            model.inner_cv_datasets.append((model._X(pre.transform(table, outer[tr])),
                                            model._X(pre.transform(table, outer[va]))))
        model.selection_metadata = {"inner_folds": 3, "preprocessing": "refitted independently in each inner fold"}
        return
    if getattr(model, "estimator", None) is None and not hasattr(model, "validation_fraction"):
        return
    outer = np.asarray(split["train"])
    subtable = table.subset(outer)
    # Use the same source of subtype labels as the variant being trained.
    source = (preproc_kwargs or {}).get("subtype_source", "pam50")
    subs = subtable.ihc_subtype if source == "ihc" else subtable.pam50
    if source == "none":
        subs = np.full(len(subtable), "none")
    labels = np.asarray([f"{s}_{e}" for s, e in zip(subs, subtable.events)])
    if min(np.unique(labels, return_counts=True)[1]) < 2:
        labels = subtable.events.astype(str)
    estimator = getattr(model, "estimator", model)
    tr, va = train_test_split(np.arange(len(outer)), test_size=estimator.validation_fraction,
                             random_state=estimator.seed, stratify=labels)
    kwargs = dict(data_driven_top_k=settings.top_k, max_genes=settings.max_genes)
    kwargs.update(preproc_kwargs or {})
    if graph is not None:
        kwargs["graph_config"] = graph
    pre = FoldPreprocessor(**kwargs).fit(table, outer[tr])
    model.selection_dataset = pre.transform(table, outer)
    model.selection_metadata = {"n_inner_train": len(tr), "n_inner_validation": len(va),
                                "selected_genes": list(pre.selected_genes), "fit_patient_ids": table.patient_ids[outer[tr]].tolist()}


def log_error(root: Path, context: str, exc: BaseException) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "errors.log").open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {context}\n")
        fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    print(f"    !! {context} failed: {exc!r} (see errors.log)")


def run_models_on_split(store: ResultStore, experiment: str, table: CohortTable, split: Dict,
                        model_names: Sequence[str], settings: Settings, external: Optional[CohortTable] = None,
                        preproc_kwargs: Optional[Dict] = None, eval_kwargs: Optional[Dict] = None) -> None:
    """Fit every requested model on one split (shared preprocessing)."""
    store.ensure_protocol(experiment, settings, table, external, preproc_kwargs, eval_kwargs, split)
    pending = [m for m in model_names if not store.exists(experiment, m, split["name"])]
    if not pending:
        return
    seed = settings.seed_for(split["repeat"])
    t0 = time.time()
    pre, train_ds, test_ds = build_datasets(table, split, settings, preproc_kwargs, external=external)
    print(f"  [{experiment}/{split['name']}] train={len(train_ds)} test={len(test_ds)} "
          f"genes={len(pre.selected_genes)} clinical={len(pre.clinical_features)} ({time.time() - t0:.1f}s)")
    eval_kwargs = eval_kwargs or {}
    for name in pending:
        t1 = time.time()
        try:
            extra = DERIVED_BLENDS if name == "KG-MSPF" else None
            model = build_model(name, settings, seed, extra_blends=extra)
            tr_ds, te_ds = _datasets_for_model(model, pre, table, split, train_ds, test_ds, external)
            prepare_selection_data(model, table, split, settings, preproc_kwargs, getattr(model, "graph_config", None))
            model.fit(tr_ds)
            metrics, arrays = evaluate_and_pack(model, tr_ds, te_ds, **eval_kwargs)
            meta = {"experiment": experiment, "config": name, "split": split["name"], "repeat": split["repeat"],
                    "fold": split["fold"], "metrics": metrics, "summary": model.summary(),
                    "n_train": len(tr_ds), "n_test": len(te_ds), "seconds": time.time() - t1,
                    "preprocessor": pre.metadata() if name == "KG-MSPF" else None,
                    "settings": asdict(settings), "inner_preprocessing": getattr(model, "selection_metadata", None)}
            arrays["test_index"] = np.asarray(split["test"]) if external is None else np.arange(len(te_ds))
            arrays["times"] = np.asarray(te_ds.times, dtype=np.float64)
            arrays["events"] = np.asarray(te_ds.events, dtype=np.int64)
            if name == "KG-MSPF":
                est = model.estimator
                derived = {}
                for blend in est.blend_names:
                    if blend == "main":
                        continue
                    m_b, a_b = evaluate_and_pack(model, tr_ds, te_ds, blend=blend, **eval_kwargs)
                    derived[blend] = m_b
                    arrays[f"risk__{blend}"] = a_b["risk"]
                    arrays[f"surv__{blend}"] = a_b["surv"]
                    arrays[f"train_risk_quantiles__{blend}"] = a_b["train_risk_quantiles"]
                meta["derived_blends"] = derived
                comps = est.predict_components(te_ds)
                for c, v in comps.items():
                    arrays[f"component__{c}"] = np.asarray(v, dtype=np.float64)
                if settings.interpretability:
                    meta["interpretability"] = interpretability_payload(model, te_ds)
                if settings.save_checkpoints:
                    _save_checkpoint(store, experiment, name, split["name"], model, pre)
            store.save(experiment, name, split["name"], arrays, meta)
            print(f"    {name:<22s} C={metrics['c_index']:.4f} IBS={metrics.get('ibs', float('nan')):.4f} "
                  f"({time.time() - t1:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            if settings.strict:
                raise
            log_error(store.root, f"{experiment}/{name}/{split['name']}", exc)


def run_variants_on_split(store: ResultStore, experiment: str, table: CohortTable, split: Dict,
                          variants: Dict[str, VariantSpec], settings: Settings,
                          reuse_from: Optional[Tuple[str, str]] = None, reuse_names: Sequence[str] = ()) -> None:
    """Fit KG-MSPF variants on one split; base preprocessing is shared when possible."""
    store.ensure_protocol(experiment, settings, table, split=split)
    pending = {k: v for k, v in variants.items() if not store.exists(experiment, k, split["name"])}
    if not pending:
        return
    seed = settings.seed_for(split["repeat"])
    # Reuse the default configuration from the main experiment if available
    if reuse_from is not None:
        # Reuse is allowed only with identical code, data, runtime, settings
        # and per-split provenance, not merely the same config name.
        source_protocol = store.root / reuse_from[0] / 'protocol.json'
        dest_protocol = store.root / experiment / 'protocol.json'
        source_split = store.root / reuse_from[0] / ('.' + split['name'] + '.sha256')
        dest_split = store.root / experiment / ('.' + split['name'] + '.sha256')
        compatible = (source_protocol.exists() and source_split.exists()
                      and source_protocol.read_bytes() == dest_protocol.read_bytes()
                      and source_split.read_bytes() == dest_split.read_bytes())
        for name in list(pending):
            if compatible and name in reuse_names and store.exists(reuse_from[0], reuse_from[1], split["name"]):
                arrays, meta = store.load(reuse_from[0], reuse_from[1], split["name"])
                meta = dict(meta)
                meta.update({"experiment": experiment, "config": name, "reused_from": f"{reuse_from[0]}/{reuse_from[1]}"})
                store.save(experiment, name, split["name"], arrays, meta)
                pending.pop(name)
    if not pending:
        return
    base_cache: Dict[str, Tuple] = {}
    for name, spec in pending.items():
        t1 = time.time()
        try:
            key = json.dumps(spec.preproc, sort_keys=True)
            if key not in base_cache:
                base_cache[key] = build_datasets(table, split, settings, spec.preproc)
            pre, train_ds, test_ds = base_cache[key]
            if spec.graph is not None:
                pre_v = pre.with_graph_config(spec.graph)
                train_ds = pre_v.transform(table, split["train"])
                test_ds = pre_v.transform(table, split["test"])
            else:
                pre_v = pre
            est = kgmspf_estimator(settings, seed, **spec.estimator)
            import model_api as api
            model = api.EstimatorWrapper(name, est)
            prepare_selection_data(model, table, split, settings, spec.preproc, spec.graph)
            model.fit(train_ds)
            metrics, arrays = evaluate_and_pack(model, train_ds, test_ds)
            arrays["test_index"] = np.asarray(split["test"])
            arrays["times"] = np.asarray(test_ds.times, dtype=np.float64)
            arrays["events"] = np.asarray(test_ds.events, dtype=np.int64)
            comps = est.predict_components(test_ds)
            for c, v in comps.items():
                arrays[f"component__{c}"] = np.asarray(v, dtype=np.float64)
            meta = {"experiment": experiment, "config": name, "description": spec.description,
                    "split": split["name"], "repeat": split["repeat"], "fold": split["fold"],
                    "metrics": metrics, "summary": model.summary(), "graph": pre_v.graph_config.describe(),
                    "preprocessor": pre_v.metadata(), "seconds": time.time() - t1,
                    "n_train": len(train_ds), "n_test": len(test_ds),
                    "settings": asdict(settings), "inner_preprocessing": model.selection_metadata}
            store.save(experiment, name, split["name"], arrays, meta)
            print(f"    {name:<24s} C={metrics['c_index']:.4f} ({time.time() - t1:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            if settings.strict:
                raise
            log_error(store.root, f"{experiment}/{name}/{split['name']}", exc)


def _save_checkpoint(store: ResultStore, experiment: str, name: str, split: str, model, pre) -> None:
    import torch

    path = store.root / experiment / "checkpoints" / f"{name}_{split}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    estimator=model.estimator
    original_device=estimator.device
    try:
        # Portable checkpoints contain CPU tensors, even when training uses
        # DirectML private-use tensors that cannot be reliably pickled.
        estimator.model.to('cpu')
        estimator.device=torch.device('cpu')
        torch.save({"estimator":estimator,"preprocessor":pre},path)
    finally:
        estimator.device=original_device
        estimator.model.to(original_device)


# =====================================================================
# External validation helpers
# =====================================================================


def external_feature_plan(metabric: CohortTable, external: CohortTable, min_coverage: float = 0.6) -> Dict:
    feats = [f for f in choose_transportable_features(external, min_coverage)
             if metabric.clinical[f].notna().mean() >= min_coverage]
    use_mut = external.mutation is not None and external.mutation.shape[1] > 0
    shared_genes = sorted(set(metabric.expression.columns) & set(external.expression.columns))
    if not feats or not shared_genes:
        raise ValueError("External cohort has no usable shared clinical/expression inputs")
    return {"clinical_features": feats, "use_mutations": bool(use_mut), "n_shared_genes": len(shared_genes),
            "shared_genes": shared_genes, "expression_scaling": "within", "feature_coverage_threshold": min_coverage}


def align_external_features(metabric, external, plan):
    """Restrict screening to actually measured shared genes before fitting."""
    source = metabric.subset(np.arange(len(metabric)))
    target = external.subset(np.arange(len(external)))
    genes = plan['shared_genes']
    source.expression = source.expression.loc[:, genes]
    target.expression = target.expression.loc[:, genes]
    return source, target


def external_eval_settings(external: CohortTable) -> Dict[str, object]:
    """Horizons adapted to the follow-up of the external cohort."""
    t = external.times
    tau = float(min(120.0, np.floor(np.quantile(t, 0.8) / 6.0) * 6.0))
    tau = max(tau, 24.0)
    auc = tuple(x for x in (12, 36, 60, 120) if x <= tau)
    brier = tuple(x for x in (36, 60, 120) if x <= tau) or (24,)
    grid = np.arange(6.0, tau + 0.1, 6.0)
    return {"time_grid": grid, "auc_times": auc, "brier_times": brier, "tau": tau}
