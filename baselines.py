"""Baseline survival models evaluated under the same fold-safe protocol.

All models receive exactly the inputs that KG-MSPF receives in a fold:
the standardised clinical block, the fold-selected gene expression and
mutation indicators, pathway summaries and the PAM50 one-hot block
(``GraphDataset.flat_features``).  Hyper-parameters of the learned baselines
are chosen on the same stratified internal validation split of the outer
training fold that KG-MSPF uses for early stopping, and the selected
configuration is refitted on the full outer training fold for the selected
number of epochs.  The outer test fold is never touched.

Models
------
Clinical-Cox, Clinical+PAM50-Cox, FullConcat-Cox   : Cox PH (numerical-stability ridge only)
FullConcat-LASSO-Cox                                 : LASSO Cox, penalty by inner 3-fold CV
RSF                                                  : random survival forest (scikit-survival)
DeepSurv                                             : MLP + Cox partial likelihood (Katzman et al. 2018)
DeepHit                                              : discrete-time PMF, NLL + ranking loss (Lee et al. 2018)
SurvTRACE                                            : SurvTRACE-style Transformer re-implementation
                                                       (feature tokens + encoder + discrete hazard head,
                                                       single event; Wang & Sun 2022)
Cox-PASNet                                           : pathway-masked network (Hao et al. 2019), same pathway map,
                                                       without the sparse-coding pruning step
HomoGAT                                              : homogeneous GAT on the same patient graph (one edge type,
                                                       no priors, no clinical head, no linear anchors)
"""

from __future__ import annotations

import copy
import gc
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

from hetero_data import GraphConfig, GraphDataset, PATHWAY_GENE_MAP
from model_heterogat import (
    HeteroGATSurvivalEstimator,
    ModelFlags,
    cox_partial_likelihood_loss,
    resolve_device,
    set_deterministic_seed,
)
from survival_utils import BreslowSurvival, concordance_index_fast, step_survival_on_grid
from model_api import BaseSurvivalModel, CoxBaseline, EstimatorWrapper  # noqa: F401  (re-exported)
from optimizers import backend_adamw


# =====================================================================
# Random survival forest
# =====================================================================


class RSFBaseline(BaseSurvivalModel):
    name = "RSF"

    def __init__(self, seed: int = 42, n_estimators: int = 300, leaf_grid: Sequence[int] = (5, 15, 30),
                 validation_fraction: float = 0.15, n_jobs: Optional[int] = None) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.leaf_grid = tuple(leaf_grid)
        self.validation_fraction = validation_fraction
        # scikit-survival builds the trees with joblib ``prefer="threads"``, so
        # ``n_jobs=-1`` grows n_jobs trees concurrently and transiently needs a
        # lot of RAM.  ``n_jobs`` does not influence the fitted forest (every
        # tree is seeded from ``random_state``), so it can be lowered on
        # memory-constrained machines via ``KGMSPF_RSF_JOBS``.
        if n_jobs is None:
            n_jobs = int(os.environ.get("KGMSPF_RSF_JOBS", "1"))
        self.n_jobs = n_jobs
        self.model = None
        self.best_leaf = None
        self.val_scores: Dict[int, float] = {}

    def _make(self, leaf: int):
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as exc:  # pragma: no cover
            raise ImportError("RSF requires scikit-survival: pip install scikit-survival") from exc
        return RandomSurvivalForest(n_estimators=self.n_estimators, min_samples_leaf=leaf,
                                    min_samples_split=2 * leaf, max_features="sqrt", n_jobs=self.n_jobs,
                                    random_state=self.seed, low_memory=False)

    @staticmethod
    def _y(times, events):
        from sksurv.util import Surv
        return Surv.from_arrays(event=np.asarray(events).astype(bool), time=np.asarray(times, dtype=float))

    def fit(self, train: GraphDataset) -> "RSFBaseline":
        X = train.flat_features.astype(float)
        idx = np.arange(len(train))
        tr, va = train_test_split(idx, test_size=self.validation_fraction, random_state=self.seed,
                                  stratify=HeteroGATSurvivalEstimator.stratification_labels(train))
        selection = getattr(self, "selection_dataset", train)
        X_selection = selection.flat_features.astype(float)
        for leaf in self.leaf_grid:
            m = self._make(leaf).fit(X_selection[tr], self._y(train.times[tr], train.events[tr]))
            self.val_scores[leaf] = concordance_index_fast(train.times[va], m.predict(X_selection[va]), train.events[va])
            del m
            gc.collect()
        self.best_leaf = max(self.val_scores, key=self.val_scores.get)
        self.model = self._make(self.best_leaf).fit(X, self._y(train.times, train.events))
        return self

    def predict_risk(self, ds: GraphDataset) -> np.ndarray:
        return np.asarray(self.model.predict(ds.flat_features.astype(float)), dtype=float)

    def predict_survival(self, ds: GraphDataset, grid: Sequence[float]) -> np.ndarray:
        surv = self.model.predict_survival_function(ds.flat_features.astype(float), return_array=True)
        times = getattr(self.model, "unique_times_", None)
        if times is None:
            times = getattr(self.model, "event_times_")
        return step_survival_on_grid(np.asarray(times, dtype=float), np.asarray(surv, dtype=float), grid)

    def summary(self) -> Dict[str, object]:
        return {"best_min_samples_leaf": self.best_leaf, "val_cindex": self.val_scores}


# =====================================================================
# Torch baselines: shared trainer
# =====================================================================


def _mlp(in_dim: int, hidden: Sequence[int], dropout: float, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _TorchSurvivalModel(BaseSurvivalModel):
    """Grid search on the internal validation split, then refit on the fold."""

    discrete: bool = False

    def __init__(self, grid: Sequence[Dict[str, object]], learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 300, patience: int = 30, batch_size: Optional[int] = None,
                 validation_fraction: float = 0.15, seed: int = 42, device: str = "auto", verbose: bool = False,
                 n_durations: int = 20) -> None:
        self.grid = list(grid)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.device_request = device
        self.verbose = verbose
        self.n_durations = n_durations
        self.model: Optional[nn.Module] = None
        self.best_cfg: Optional[Dict[str, object]] = None
        self.best_epoch = 0
        self.val_scores: List[Tuple[Dict[str, object], int, float]] = []

    # -- to implement -------------------------------------------------
    def _inputs(self, ds: GraphDataset) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _build(self, cfg: Dict[str, object], ds: GraphDataset) -> nn.Module:
        raise NotImplementedError

    def _loss(self, out: torch.Tensor, idx: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
        return cox_partial_likelihood_loss(out.reshape(-1), self._t_times[idx], self._t_events[idx])

    def _risk_from_output(self, out: torch.Tensor) -> np.ndarray:
        return out.reshape(-1).detach().cpu().numpy().astype(float)

    # -- helpers ------------------------------------------------------
    def _prepare_targets(self, ds: GraphDataset) -> None:
        self._t_times = torch.as_tensor(ds.times, dtype=torch.float32, device=self.device)
        self._t_events = torch.as_tensor(ds.events, dtype=torch.float32, device=self.device)

    def _forward(self, model: nn.Module, inputs: Tuple[torch.Tensor, ...], idx: Optional[torch.Tensor] = None):
        selected=inputs if idx is None else tuple(t[idx] for t in inputs)
        chunk=int(os.environ.get('KGMSPF_BASELINE_CHUNK','64'))
        if selected[0].device.type=='privateuseone':
            # DirectML private-use RNG/checkpoint APIs are not implemented.
            # Use the available GPU VRAM for the complete differentiable loss.
            chunk=0
        n=len(selected[0])
        if chunk<=0 or n<=chunk:
            return model(*selected)
        outputs=[]
        for start in range(0,n,chunk):
            xs=tuple(t[start:start+chunk] for t in selected)
            if torch.is_grad_enabled() and model.training:
                out=torch.utils.checkpoint.checkpoint(model,*xs,use_reentrant=False)
            else:
                out=model(*xs)
            outputs.append(out)
        return torch.cat(outputs,dim=0)

    def _predict_output(self, model: nn.Module, inputs: Tuple[torch.Tensor, ...], batch: int = 512) -> torch.Tensor:
        model.eval()
        n = inputs[0].shape[0]
        outs = []
        with torch.no_grad():
            for s in range(0, n, batch):
                idx = torch.arange(s, min(s + batch, n), device=self.device)
                outs.append(self._forward(model, inputs, idx))
        return torch.cat(outs, dim=0)

    def _train(self, model: nn.Module, cfg: Dict[str, object], inputs, tr: np.ndarray,
               va: Optional[np.ndarray], val_times=None, val_events=None, n_epochs: Optional[int] = None):
        lr = float(cfg.get("lr", self.learning_rate))
        wd = float(cfg.get("weight_decay", self.weight_decay))
        opt = backend_adamw(model.parameters(), lr=lr, weight_decay=wd)
        tr_t = torch.as_tensor(tr, dtype=torch.long, device=self.device)
        best_state = copy.deepcopy(model.state_dict())
        best_c, best_epoch, since = -np.inf, 0, 0
        epochs = n_epochs if n_epochs is not None else self.max_epochs
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        for epoch in range(1, epochs + 1):
            model.train()
            if self.batch_size is None or self.batch_size >= tr.size:
                batches = [tr_t]
            else:
                perm = torch.randperm(tr.size, generator=gen).to(self.device)
                batches = [tr_t[perm[s:s + self.batch_size]] for s in range(0, tr.size, self.batch_size)]
            for b in batches:
                if b.numel() < 2:
                    continue
                opt.zero_grad(set_to_none=True)
                out = self._forward(model, inputs, b)
                loss = self._loss(out, b, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{self.name}: non-finite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
            if va is None:
                continue
            va_t = torch.as_tensor(va, dtype=torch.long, device=self.device)
            model.eval()
            with torch.no_grad():
                out = self._forward(model, inputs, va_t)
            c = concordance_index_fast(val_times, self._risk_from_output(out), val_events)
            if c > best_c + 1e-4:
                best_c, best_epoch, since = c, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since += 1
                if since >= self.patience:
                    break
        if va is not None:
            model.load_state_dict(best_state)
        model.eval()
        return best_epoch, best_c

    # -- public API ---------------------------------------------------
    def fit(self, train: GraphDataset) -> "_TorchSurvivalModel":
        set_deterministic_seed(self.seed)
        self.device = resolve_device(self.device_request)
        idx = np.arange(len(train))
        tr, va = train_test_split(idx, test_size=self.validation_fraction, random_state=self.seed,
                                  stratify=HeteroGATSurvivalEstimator.stratification_labels(train))
        selection = getattr(self, "selection_dataset", train)
        self._fit_preprocessing(selection.subset(tr))
        self._prepare_targets(selection)
        inputs = self._inputs(selection)
        best = None
        for cfg in self.grid:
            set_deterministic_seed(self.seed)
            model = self._build(cfg, selection).to(self.device)
            ep, c = self._train(model, cfg, inputs, tr, va, train.times[va], train.events[va])
            self.val_scores.append((cfg, ep, c))
            if self.verbose:
                print(f"      {self.name} cfg={cfg} best_epoch={ep} val_C={c:.4f}")
            if best is None or c > best[2]:
                best = (cfg, ep, c)
        self.best_cfg, self.best_epoch, _ = best
        set_deterministic_seed(self.seed + 1009)
        self._fit_preprocessing(train)
        self._prepare_targets(train)
        inputs = self._inputs(train)
        self.model = self._build(self.best_cfg, train).to(self.device)
        self._train(self.model, self.best_cfg, inputs, idx, None, n_epochs=max(int(self.best_epoch), 1))
        self._post_fit(train, inputs)
        return self

    def _fit_preprocessing(self, train: GraphDataset) -> None:
        pass

    def _post_fit(self, train: GraphDataset, inputs) -> None:
        risk = self._risk_from_output(self._predict_output(self.model, inputs))
        self.breslow = BreslowSurvival().fit(risk, train.times, train.events)

    def predict_risk(self, ds: GraphDataset) -> np.ndarray:
        return self._risk_from_output(self._predict_output(self.model, self._inputs(ds)))

    def predict_survival(self, ds: GraphDataset, grid: Sequence[float]) -> np.ndarray:
        return self.breslow.survival(self.predict_risk(ds), grid)

    def summary(self) -> Dict[str, object]:
        return {"best_cfg": self.best_cfg, "best_epoch": self.best_epoch,
                "grid": [{"cfg": c, "epoch": e, "val_c": v} for c, e, v in self.val_scores]}


# =====================================================================
# DeepSurv
# =====================================================================


class DeepSurvBaseline(_TorchSurvivalModel):
    name = "DeepSurv"

    def __init__(self, **kwargs) -> None:
        grid = kwargs.pop("grid", [
            {"hidden": (32,), "dropout": 0.1}, {"hidden": (32,), "dropout": 0.3},
            {"hidden": (64, 32), "dropout": 0.1}, {"hidden": (64, 32), "dropout": 0.3},
            {"hidden": (128, 64), "dropout": 0.1}, {"hidden": (128, 64), "dropout": 0.3},
        ])
        super().__init__(grid=grid, **kwargs)

    def _inputs(self, ds: GraphDataset):
        return (torch.as_tensor(ds.flat_features, dtype=torch.float32, device=self.device),)

    def _build(self, cfg, ds):
        return _mlp(ds.flat_features.shape[1], cfg["hidden"], float(cfg["dropout"]), 1)


# =====================================================================
# Discrete-time models (DeepHit, SurvTRACE-style)
# =====================================================================


class _DiscreteTimeModel(_TorchSurvivalModel):
    discrete = True

    def _fit_preprocessing(self, train: GraphDataset) -> None:
        ev_times = np.asarray(train.times[train.events == 1], dtype=float)
        qs = np.quantile(ev_times, np.linspace(0, 1, self.n_durations + 1))
        cuts = np.unique(np.concatenate([[0.0], qs[1:]]))
        self.cuts = cuts
        self.K = cuts.size - 1

    def _duration_index(self, times: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.cuts, np.asarray(times, dtype=float), side="left") - 1
        return np.clip(idx, 0, self.K - 1)

    def _prepare_targets(self, ds: GraphDataset) -> None:
        super()._prepare_targets(ds)
        self._t_idx = torch.as_tensor(self._duration_index(ds.times), dtype=torch.long, device=self.device)

    def _survival_from_output(self, out: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def _risk_from_output(self, out: torch.Tensor) -> np.ndarray:
        S = self._survival_from_output(out)  # n x K, survival at the end of each interval
        widths = np.diff(self.cuts)
        S_prev = np.hstack([np.ones((S.shape[0], 1)), S[:, :-1]])
        rmst = np.sum(S_prev * widths[None, :], axis=1)
        return -rmst

    def _post_fit(self, train, inputs) -> None:
        pass

    def predict_survival(self, ds: GraphDataset, grid: Sequence[float]) -> np.ndarray:
        S = self._survival_from_output(self._predict_output(self.model, self._inputs(ds)))
        return step_survival_on_grid(self.cuts[1:], S, grid)


class DeepHitBaseline(_DiscreteTimeModel):
    name = "DeepHit"

    def __init__(self, **kwargs) -> None:
        grid = kwargs.pop("grid", [
            {"hidden": (64, 64), "dropout": 0.1, "alpha": 0.2, "sigma": 0.1},
            {"hidden": (64, 64), "dropout": 0.1, "alpha": 0.5, "sigma": 0.1},
            {"hidden": (128, 64), "dropout": 0.2, "alpha": 0.2, "sigma": 0.1},
            {"hidden": (128, 64), "dropout": 0.2, "alpha": 0.5, "sigma": 0.1},
        ])
        super().__init__(grid=grid, **kwargs)

    def _inputs(self, ds: GraphDataset):
        return (torch.as_tensor(ds.flat_features, dtype=torch.float32, device=self.device),)

    def _build(self, cfg, ds):
        return _mlp(ds.flat_features.shape[1], cfg["hidden"], float(cfg["dropout"]), self.K)

    @staticmethod
    def _pmf(out: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([out, out.new_zeros(out.shape[0], 1)], dim=1)
        return torch.softmax(padded, dim=1)

    def _loss(self, out, idx, cfg):
        pmf = self._pmf(out)
        cdf = torch.cumsum(pmf[:, : self.K], dim=1)
        k = self._t_idx[idx]
        e = self._t_events[idx]
        t = self._t_times[idx]
        rows = torch.arange(out.shape[0], device=out.device)
        eps = 1e-7
        nll_event = -torch.log(pmf[rows, k].clamp_min(eps))
        nll_cens = -torch.log((1.0 - cdf[rows, k]).clamp_min(eps))
        nll = (e * nll_event + (1.0 - e) * nll_cens).mean()
        # ranking loss: F_i(T_i) - F_j(T_i) for comparable pairs
        F_at = cdf[:, k]                      # [j, i] = F_j(T_i)
        diag = F_at.diagonal()                # F_i(T_i)
        R = diag.unsqueeze(1) - F_at.t()      # [i, j]
        comparable = ((t.unsqueeze(1) < t.unsqueeze(0)) & (e.unsqueeze(1) == 1)) | (
            (t.unsqueeze(1) == t.unsqueeze(0)) & (e.unsqueeze(1) == 1) & (e.unsqueeze(0) == 0))
        rank = (comparable.float() * torch.exp(-R / float(cfg["sigma"]))).mean()
        alpha = float(cfg["alpha"])
        return alpha * nll + (1.0 - alpha) * rank

    def _survival_from_output(self, out):
        pmf = self._pmf(out)
        cdf = torch.cumsum(pmf[:, : self.K], dim=1)
        return (1.0 - cdf).detach().cpu().numpy().astype(float)


class _FeatureTokenizer(nn.Module):
    def __init__(self, n_tokens: int, in_dim: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_tokens, in_dim, d) * (1.0 / np.sqrt(max(in_dim, 1))) * 0.5)
        self.bias = nn.Parameter(torch.randn(n_tokens, d) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, in]
        return torch.einsum("bti,tid->btd", x, self.weight) + self.bias


class _SurvTraceNet(nn.Module):
    def __init__(self, n_clin, n_genes, gene_in, n_path, path_in, n_subtypes, d, heads, layers, dropout, K):
        super().__init__()
        self.clin_tok = _FeatureTokenizer(n_clin, 1, d)
        self.gene_tok = _FeatureTokenizer(n_genes, gene_in, d)
        self.path_tok = _FeatureTokenizer(n_path, path_in, d)
        self.sub_emb = nn.Embedding(n_subtypes, d) if n_subtypes > 1 else None
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, K))

    def forward(self, clin, genes, paths, sub):
        b = clin.shape[0]
        tokens = [self.cls.expand(b, -1, -1), self.clin_tok(clin.unsqueeze(-1)), self.gene_tok(genes),
                  self.path_tok(paths)]
        if self.sub_emb is not None:
            tokens.append(self.sub_emb(sub).unsqueeze(1))
        h = self.encoder(torch.cat(tokens, dim=1))
        return self.head(self.norm(h[:, 0]))


class SurvTraceBaseline(_DiscreteTimeModel):
    """SurvTRACE-style Transformer (single event, logistic discrete hazard)."""

    name = "SurvTRACE"

    def __init__(self, **kwargs) -> None:
        grid = kwargs.pop("grid", [{"d": 32, "heads": 4, "layers": 2, "dropout": 0.1}])
        kwargs.setdefault("batch_size", 256)
        kwargs.setdefault("max_epochs", 100)
        kwargs.setdefault("patience", 15)
        super().__init__(grid=grid, **kwargs)

    def _inputs(self, ds: GraphDataset):
        dev = self.device
        clin = torch.as_tensor(ds.block("clinical"), dtype=torch.float32, device=dev)
        use_mut = bool(ds.gene_has_mutation.any())
        g = [ds.gene_expression[:, :, None]]
        if use_mut:
            g.append(ds.gene_mutation[:, :, None])
        genes = torch.as_tensor(np.concatenate(g, axis=2), dtype=torch.float32, device=dev)
        p = [ds.block("pathway_expr")[:, :, None]]
        if "pathway_mut" in ds.flat_blocks:
            p.append(ds.block("pathway_mut")[:, :, None])
        paths = torch.as_tensor(np.concatenate(p, axis=2), dtype=torch.float32, device=dev)
        sub = torch.as_tensor(ds.subtype_ids, dtype=torch.long, device=dev)
        return clin, genes, paths, sub

    def _build(self, cfg, ds):
        use_mut = bool(ds.gene_has_mutation.any())
        return _SurvTraceNet(
            n_clin=ds.clinical_dim, n_genes=len(ds.selected_genes), gene_in=2 if use_mut else 1,
            n_path=ds.block("pathway_expr").shape[1], path_in=2 if "pathway_mut" in ds.flat_blocks else 1,
            n_subtypes=ds.n_subtypes, d=int(cfg["d"]), heads=int(cfg["heads"]), layers=int(cfg["layers"]),
            dropout=float(cfg["dropout"]), K=self.K,
        )

    def _loss(self, out, idx, cfg):
        k = self._t_idx[idx]
        e = self._t_events[idx]
        steps = torch.arange(self.K, device=out.device).unsqueeze(0)
        mask = (steps <= k.unsqueeze(1)).float()
        target = ((steps == k.unsqueeze(1)) & (e.unsqueeze(1) == 1)).float()
        bce = (F.softplus(out)-out*target if out.device.type=='privateuseone'
               else F.binary_cross_entropy_with_logits(out, target, reduction="none"))
        return (bce * mask).sum(dim=1).mean()

    def _survival_from_output(self, out):
        log_s = torch.cumsum(-F.softplus(out) if out.device.type=='privateuseone' else F.logsigmoid(-out), dim=1)
        return torch.exp(log_s).detach().cpu().numpy().astype(float)


# =====================================================================
# Cox-PASNet
# =====================================================================


class _CoxPASNetNet(nn.Module):
    def __init__(self, mask: np.ndarray, n_clin: int, hidden: int, dropout: float) -> None:
        super().__init__()
        n_path, n_gene = mask.shape
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        self.gene_to_pathway = nn.Linear(n_gene, n_path)
        self.hidden = nn.Linear(n_path, hidden)
        self.out = nn.Linear(hidden + n_clin, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gene_to_pathway.weight)
        with torch.no_grad():
            self.gene_to_pathway.weight.mul_(self.mask)

    def forward(self, genes, clin):
        w = self.gene_to_pathway.weight * self.mask
        p = torch.tanh(F.linear(self.drop(genes), w, self.gene_to_pathway.bias))
        h = torch.tanh(self.hidden(self.drop(p)))
        return self.out(torch.cat([h, clin], dim=-1))


class CoxPASNetBaseline(_TorchSurvivalModel):
    name = "Cox-PASNet"

    def __init__(self, **kwargs) -> None:
        grid = kwargs.pop("grid", [
            {"hidden": 16, "dropout": 0.1}, {"hidden": 16, "dropout": 0.3},
            {"hidden": 32, "dropout": 0.1}, {"hidden": 32, "dropout": 0.3},
        ])
        super().__init__(grid=grid, **kwargs)

    def _fit_preprocessing(self, train: GraphDataset) -> None:
        members = sorted({g for gs in train.pathway_members.values() for g in gs})
        self.member_cols = [train.selected_genes.index(g) for g in members]
        pathways = list(PATHWAY_GENE_MAP)
        mask = np.zeros((len(pathways), len(members)), dtype=np.float32)
        for i, p in enumerate(pathways):
            for g in train.pathway_members.get(p, ()):
                mask[i, members.index(g)] = 1.0
        self.mask = mask

    def _inputs(self, ds: GraphDataset):
        genes = torch.as_tensor(ds.gene_expression[:, self.member_cols], dtype=torch.float32, device=self.device)
        clin = torch.as_tensor(ds.block("clinical"), dtype=torch.float32, device=self.device)
        return genes, clin

    def _build(self, cfg, ds):
        return _CoxPASNetNet(self.mask, ds.clinical_dim, int(cfg["hidden"]), float(cfg["dropout"]))


# =====================================================================
# Homogeneous GAT and KG-MSPF wrappers
# =====================================================================


def make_homogat(seed: int, device: str, verbose: bool, max_epochs: int = 200, patience: int = 25,
                 train_chunk_size: int = 0, validation_fraction: float = 0.25, min_epochs: int = 60) -> EstimatorWrapper:
    flags = ModelFlags(use_clinical_head=False, use_subtype_residual=False, use_pathway_residual=False,
                       use_graph_residual=True, use_priors=False, graph_only=True)
    est = HeteroGATSurvivalEstimator(flags=flags, blend_components=("heterogat",), max_epochs=max_epochs,
                                     patience=patience, seed=seed, device=device, verbose=verbose,
                                     train_chunk_size=train_chunk_size, validation_fraction=validation_fraction,
                                     min_epochs=min(min_epochs, max_epochs))
    return EstimatorWrapper("HomoGAT", est, graph_config=GraphConfig(relation_aware=False))
