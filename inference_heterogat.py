"""Score patients with a saved KG-MSPF checkpoint.

Checkpoints are written with ``python run_experiments.py holdout --save-checkpoints``
(or any other command) to ``outputs/revision/<experiment>/checkpoints/KG-MSPF_<split>.pt``.

    python inference_heterogat.py --checkpoint outputs/revision/holdout/checkpoints/KG-MSPF_cohort_holdout.pt \
        [--input harmonised.csv] [--output predictions.csv]

Without --input the METABRIC table is scored. Single-patient inputs use the
training reference scale by default; cohort scaling requires an explicit flag.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from model_heterogat import resolve_device

from experiment import TIME_GRID
from hetero_data import CohortTable, load_metabric


def load_bundle(path: str, device: str = "cpu"):
    # DirectML and 'auto' are application device selectors, not torch.load
    # storage locations. Portable checkpoints are always deserialized on CPU.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    est, pre = payload["estimator"], payload["preprocessor"]
    est.device = resolve_device(device)
    est.inference_batch_size = getattr(est,'inference_batch_size',128)
    est.clinical_offset = getattr(est,'clinical_offset',False)
    for layer in est.model.layers:
        if not getattr(layer,'_layout_names',None):
            layer.prepare_directml_layouts(est.model.schema)
    est.model.to(est.device)
    est.model.eval()
    return est, pre


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", default=None, help="harmonised CSV (see external_data.py); default: METABRIC")
    p.add_argument("--output", default="kgmspf_predictions.csv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--expression-scaling",choices=['train','within'],default='train',
                   help='train for individual prediction; within requires an external cohort/reference batch')
    args = p.parse_args()
    est, pre = load_bundle(args.checkpoint, args.device)
    if args.input:
        table = CohortTable.from_harmonized_csv(args.input,require_outcomes=False)
        if args.expression_scaling=='within' and len(table)<2:
            raise ValueError('Within-cohort scaling cannot be estimated from one patient')
        ds = pre.transform(table, np.arange(len(table)), expression_scaling=args.expression_scaling)
    else:
        table = load_metabric()
        ds = pre.transform(table, np.arange(len(table)))
    risk = est.predict_risk(ds)
    surv = est.predict_survival(ds, TIME_GRID)
    out = pd.DataFrame({"patient_id": ds.patient_ids, "risk_score": risk})
    for comp, values in est.predict_components(ds).items():
        out[f"component_{comp}"] = values
    for t in (36, 60, 120):
        k = int(np.argmin(np.abs(TIME_GRID - t)))
        out[f"survival_{t}m"] = surv[:, k]
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
