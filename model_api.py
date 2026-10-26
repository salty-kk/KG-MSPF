"""Torch-free model interface shared by KG-MSPF and the baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from hetero_data import GraphConfig, GraphDataset
from survival_utils import BreslowSurvival, CoxPHRegression, fit_penalized_cox_cv


@dataclass
class ModelFlags:
    """Branch switches of the KG-MSPF network (used for re-trained ablations)."""

    use_clinical_head: bool = True
    use_subtype_residual: bool = True
    use_pathway_residual: bool = True
    use_graph_residual: bool = True
    use_priors: bool = True
    graph_only: bool = False  # graph risk without gate (generic GNN baseline)
    mask_clinical_graph_inputs: bool = False


class BaseSurvivalModel:
    name: str = "base"
    graph_config: Optional[GraphConfig] = None  # dataset variant required by the model

    def fit(self, train: GraphDataset) -> "BaseSurvivalModel":
        raise NotImplementedError

    def predict_risk(self, ds: GraphDataset) -> np.ndarray:
        raise NotImplementedError

    def predict_survival(self, ds: GraphDataset, grid: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def summary(self) -> Dict[str, object]:
        return {}


# =====================================================================
# Linear Cox baselines
# =====================================================================


class CoxBaseline(BaseSurvivalModel):
    def __init__(self, name: str, blocks: Optional[Sequence[str]] = None, penalty: str = "none",
                 alpha: float = 1e-6, seed: int = 42) -> None:
        self.name = name
        self.blocks = tuple(blocks) if blocks else None
        self.penalty = penalty
        self.alpha = alpha
        self.seed = seed
        self.model: Optional[CoxPHRegression] = None
        self.info: Dict[str, object] = {}

    def _X(self, ds: GraphDataset) -> np.ndarray:
        return (ds.block(*self.blocks) if self.blocks else ds.flat_features).astype(float)

    def fit(self, train: GraphDataset) -> "CoxBaseline":
        X = self._X(train)
        if self.penalty == "lasso":
            self.model, self.info = fit_penalized_cox_cv(X, train.times, train.events, l1_ratio=1.0, seed=self.seed,
                                                       fold_data=getattr(self, "inner_cv_datasets", None))
        else:
            self.model = CoxPHRegression(alpha=self.alpha).fit(X, train.times, train.events)
        self.breslow = BreslowSurvival().fit(self.model.predict_risk(X), train.times, train.events)
        return self

    def predict_risk(self, ds: GraphDataset) -> np.ndarray:
        return self.model.predict_risk(self._X(ds))

    def predict_survival(self, ds: GraphDataset, grid: Sequence[float]) -> np.ndarray:
        return self.breslow.survival(self.predict_risk(ds), grid)

    def summary(self) -> Dict[str, object]:
        out = {"n_features": int(self.model.coef_.size), "n_nonzero": self.model.n_nonzero}
        if self.info:
            out.update({"best_lambda": self.info.get("best_lambda")})
        return out



class EstimatorWrapper(BaseSurvivalModel):
    """Wrap ``HeteroGATSurvivalEstimator`` in the common interface."""

    def __init__(self, name: str, estimator, blend: str = "main",
                 graph_config: Optional[GraphConfig] = None) -> None:
        self.name = name
        self.estimator = estimator
        self.blend = blend
        self.graph_config = graph_config

    def fit(self, train: GraphDataset) -> "EstimatorWrapper":
        self.estimator.fit(train, selection_dataset=getattr(self, "selection_dataset", None))
        return self

    def predict_risk(self, ds: GraphDataset, blend: Optional[str] = None) -> np.ndarray:
        return self.estimator.predict_risk(ds, blend or self.blend)

    def predict_survival(self, ds: GraphDataset, grid: Sequence[float], blend: Optional[str] = None,
                         risk: Optional[np.ndarray] = None) -> np.ndarray:
        return self.estimator.predict_survival(ds, grid, blend or self.blend, risk=risk)

    def summary(self) -> Dict[str, object]:
        return self.estimator.summary()
