"""Model-level experiment commands. Use reproduce.py for the fixed paper configuration.

Run python run_experiments.py --help for individual model and split options."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import hashlib
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from config import OUTPUT_DIR
from experiment import (
    MAIN_MODELS,
    REVISION_DIR,
    ResultStore,
    Settings,
    ablation_variants,
    external_eval_settings,
    external_feature_plan,
    align_external_features,
    holdout_split,
    make_splits,
    pam50_variants,
    prior_variants,
    run_models_on_split,
    run_variants_on_split,
    stratification_labels,
)
from hetero_data import CohortTable, load_metabric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["smoke", "main", "holdout", "ablation", "prior", "pam50", "external", "all"])
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split-names", default=None, help="explicit split names, e.g. r0_f1,r0_f2; does not change the declared CV protocol")
    p.add_argument("--draws", type=int, default=3, help="random draws per prior-sensitivity setting")
    p.add_argument("--models", default=None, help="comma-separated subset of models (main/holdout/external)")
    p.add_argument("--skip-models", default="", help="comma-separated models to skip")
    p.add_argument("--configs", default=None, help="comma-separated subset of variant configs")
    p.add_argument("--cohort", default=None, help="external cohort name (tcga, scanb, ...)")
    p.add_argument("--data", default=None, help="harmonised CSV of the external cohort")
    p.add_argument("--out", default=str(REVISION_DIR))
    p.add_argument("--no-interpretability", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--strict", action="store_true", help="stop at the first error")
    p.add_argument("--keep-going",action='store_true',help='continue other jobs after a failure, then exit with failure status')
    for f in fields(Settings):
        if f.name in {"smoke", "strict", "interpretability", "save_checkpoints"}:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(flag, action="store_true", default=f.default)
        else:
            caster = {"int": int, "float": float, "str": str}.get(str(f.type), type(f.default))
            p.add_argument(flag, type=caster, default=f.default)
    return p.parse_args()


def settings_from_args(args: argparse.Namespace) -> Settings:
    kwargs = {}
    for f in fields(Settings):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    s = Settings(**kwargs)
    s.strict = bool(args.strict) or not bool(args.keep_going)
    s.interpretability = not args.no_interpretability
    s.save_checkpoints = bool(args.save_checkpoints)
    return s


def model_list(args) -> list:
    models = list(MAIN_MODELS) if not args.models else [m.strip() for m in args.models.split(",") if m.strip()]
    skip = {m.strip() for m in args.skip_models.split(",") if m.strip()}
    unknown = set(models) - set(MAIN_MODELS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    selected = [m for m in models if m not in skip]
    if not selected:
        raise ValueError("No models selected")
    return selected


def filter_variants(variants: dict, args) -> dict:
    if not args.configs:
        return variants
    keep = {c.strip() for c in args.configs.split(",")}
    unknown = keep - set(variants)
    if unknown:
        raise ValueError(f"Unknown variant configs: {sorted(unknown)}")
    return {k: v for k, v in variants.items() if k in keep}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def select_splits(splits, args):
    if not args.split_names:
        return splits
    names = set(args.split_names.split(','))
    unknown = names - {s['name'] for s in splits}
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    return [s for s in splits if s['name'] in names]


# =====================================================================
# Commands
# =====================================================================


def cmd_main(table, store, settings, args, splits=None):
    section("Repeated cross-validation: KG-MSPF and baselines")
    repeats = args.repeats or 5
    splits = splits or make_splits(table, repeats, args.folds, seed=settings.base_seed)
    splits = select_splits(splits,args)
    models = model_list(args)
    for split in splits:
        run_models_on_split(store, "main", table, split, models, settings)


def cmd_holdout(table, store, settings, args):
    section("Cohort hold-out: cohorts 1-3 -> 4-5")
    run_models_on_split(store, "holdout", table, holdout_split(table), model_list(args), settings)


def cmd_variants(kind, table, store, settings, args, splits=None):
    section(f"KG-MSPF variants: {kind}")
    repeats = args.repeats or 1
    splits = splits or make_splits(table, repeats, args.folds, seed=settings.base_seed)
    splits = select_splits(splits,args)
    if kind == "ablation":
        variants, reuse = ablation_variants(), ("full",)
    elif kind == "prior":
        variants, reuse = prior_variants(args.draws), ("default",)
    else:
        variants, reuse = pam50_variants(), ("pam50_full",)
    variants = filter_variants(variants, args)
    for split in splits:
        print(f"  [{kind}/{split['name']}]")
        run_variants_on_split(store, kind, table, split, variants, settings,
                              reuse_from=("main", "KG-MSPF"), reuse_names=reuse)


def cmd_external(table, store, settings, args, data_path=None, cohort=None):
    cohort = cohort or args.cohort
    data_path = data_path or args.data
    if not cohort or not data_path:
        raise SystemExit("external requires --cohort and --data")
    section(f"External validation: train on METABRIC, test on {cohort}")
    ext = CohortTable.from_harmonized_csv(data_path, name=cohort)
    overlap=set(map(str,table.patient_ids)) & set(map(str,ext.patient_ids))
    if overlap and not (settings.smoke and cohort.startswith('synthetic_')):
        raise ValueError(f'External validation overlaps training patients ({len(overlap)} IDs)')
    plan = external_feature_plan(table, ext)
    table, ext = align_external_features(table, ext, plan)
    ev = external_eval_settings(ext)
    print(f"  external n={len(ext)} deaths={int(ext.events.sum())} shared genes={plan['n_shared_genes']}")
    print(f"  transportable clinical features: {plan['clinical_features']}")
    print(f"  mutations used: {plan['use_mutations']}  tau={ev['tau']} AUC times={ev['auc_times']}")
    exp_name = f"external_{cohort}"
    provenance_path = Path(str(data_path) + '.provenance.json')
    source_provenance = (json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.exists() else {})
    if source_provenance:
        actual = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        if source_provenance.get('harmonized_sha256') != actual:
            raise ValueError('External harmonized data changed since its provenance was recorded')
    (store.root / exp_name).mkdir(parents=True, exist_ok=True)
    (store.root / exp_name / "plan.json").write_text(json.dumps(
        {**plan, "source_provenance": source_provenance,"independent_patient_ids":not bool(overlap),
         "synthetic_smoke_only":bool(settings.smoke and cohort.startswith('synthetic_')),
         "tau": ev["tau"], "auc_times": list(ev["auc_times"]), "brier_times": list(ev["brier_times"]),
         "n_external": len(ext), "events_external": int(ext.events.sum()),
         "median_followup_months": float(np.median(ext.times))}, indent=2), encoding="utf-8")
    split = {"name": "metabric_to_" + cohort, "repeat": 0, "fold": 0, "train": np.arange(len(table)),
             "test": np.arange(len(ext))}
    run_models_on_split(store, exp_name, table, split, model_list(args), settings, external=ext,
                        preproc_kwargs={"clinical_features": plan["clinical_features"],
                                        "use_mutations": plan["use_mutations"]},
                        eval_kwargs=ev)


def cmd_smoke(table, args):
    import external_data

    root = Path(OUTPUT_DIR) / "revision_smoke"
    if root.exists():
        if root.resolve().parent!=Path(OUTPUT_DIR).resolve() or root.is_symlink():
            raise ValueError('Refusing to clear an unexpected smoke path')
        shutil.rmtree(root)
    store = ResultStore(root)
    settings = Settings(epochs=4, patience=2, min_epochs=1, clinical_warmup=1, device=args.device, verbose=False, smoke=True)
    rng = np.random.default_rng(0)
    labels = stratification_labels(table)
    # stratified subset of ~600 patients for speed
    subset = np.sort(np.concatenate([
        rng.choice(np.where(labels == lab)[0], size=max(2, int(round((labels == lab).sum() * 0.32))), replace=False)
        for lab in np.unique(labels)
    ]))
    splits = make_splits(table, 1, 2, seed=settings.base_seed, subset=subset)[:1]
    t0 = time.time()
    args.models = None
    args.configs = None
    args.repeats = 1
    cmd_main(table, store, settings, args, splits=splits)
    cmd_holdout(table, store, settings, args)
    for kind in ("ablation", "prior", "pam50"):
        args.draws = 1
        cmd_variants(kind, table, store, settings, args, splits=splits)
    for kind in ("scanb_like", "tcga_like"):
        path = root / f"synthetic_{kind}.csv"
        external_data.make_synthetic_external(path, kind)
        cmd_external(table, store, settings, args, data_path=path, cohort=f"synthetic_{kind}")
    section("Smoke post-processing")
    import postprocess
    try:
        postprocess.run_all(root, smoke=True)
        post_ok = True
    except Exception as exc:  # noqa: BLE001
        import traceback
        post_ok = False
        with (root / "errors.log").open("a", encoding="utf-8") as fh:
            fh.write("\n=== postprocess\n" + traceback.format_exc())
        print(f"postprocess failed: {exc!r}")

    # Report
    expected = []
    for m in MAIN_MODELS:
        expected += [("main", m), ("holdout", m), ("external_synthetic_scanb_like", m), ("external_synthetic_tcga_like", m)]
    for k in ablation_variants():
        expected.append(("ablation", k))
    for k in prior_variants(1):
        expected.append(("prior", k))
    for k in pam50_variants():
        expected.append(("pam50", k))
    lines = []
    n_fail = 0
    for exp, cfg in expected:
        ok = any((root / exp / cfg.replace("/", "_").replace(" ", "_")).glob("*.json"))
        n_fail += 0 if ok else 1
        lines.append(f"{'PASS' if ok else 'FAIL'}  {exp:<34s} {cfg}")
    lines.append(f"{'PASS' if post_ok else 'FAIL'}  postprocess")
    n_fail += 0 if post_ok else 1
    lines.append(f"\n{len(expected) + 1 - n_fail} passed, {n_fail} failed, {time.time() - t0:.0f} s")
    report = "\n".join(lines)
    try:
        import torch
        report = f"torch {torch.__version__}, cuda={torch.cuda.is_available()}\n" + report
    except Exception:
        pass
    try:
        import sksurv
        report = f"scikit-survival {sksurv.__version__}\n" + report
    except Exception:
        report = "scikit-survival NOT installed (RSF will fail)\n" + report
    (root / "smoke_report.txt").write_text(report, encoding="utf-8")
    section("Smoke report")
    print(report)
    if (root / "errors.log").exists():
        print(f"\nErrors were logged to {root / 'errors.log'} - please send this file together with smoke_report.txt")
    if n_fail:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    table = load_metabric()
    print(f"METABRIC: n={len(table)}, deaths={int(table.events.sum())}")
    if args.command == "smoke":
        cmd_smoke(table, args)
        return
    settings = settings_from_args(args)
    store = ResultStore(Path(args.out))
    store.root.mkdir(parents=True, exist_ok=True)
    invocations=store.root/'invocations'
    invocations.mkdir(exist_ok=True)
    (invocations/f'{time.time_ns()}_{args.command}.json').write_text(
        json.dumps({'args':vars(args),'settings':asdict(settings)},indent=2,default=str),encoding='utf-8')
    error_path=store.root/'errors.log'
    errors_before=error_path.stat().st_size if error_path.exists() else 0
    (store.root / f"run_config_{args.command}.json").write_text(
        json.dumps({"args": vars(args), "settings": asdict(settings)}, indent=2, default=str), encoding="utf-8")
    t0 = time.time()
    if args.command in ("main", "all"):
        cmd_main(table, store, settings, args)
    if args.command in ("holdout", "all"):
        cmd_holdout(table, store, settings, args)
    if args.command in ("ablation", "all"):
        cmd_variants("ablation", table, store, settings, args)
    if args.command in ("prior", "all"):
        cmd_variants("prior", table, store, settings, args)
    if args.command in ("pam50", "all"):
        cmd_variants("pam50", table, store, settings, args)
    if args.command == "external":
        cmd_external(table, store, settings, args)
    section(f"Finished {args.command} in {(time.time() - t0) / 60:.1f} min; results in {store.root}")
    if (store.root / "errors.log").exists():
        print(f"Some runs failed; see {store.root / 'errors.log'}")
        if (store.root/'errors.log').stat().st_size>errors_before:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
