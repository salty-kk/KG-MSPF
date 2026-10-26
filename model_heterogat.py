"""Relation-aware heterogeneous GAT for breast-cancer survival (KG-MSPF).

Neural risk  eta_nn = w'c + s(a) r_sub + s(k) r_pw + s(b) r_graph   (Eq. 7)
Final risk   eta_KG = sum_k w_k * standardised(r_k)                    (Eq. 10)

Revision notes (compared with the submitted version)
----------------------------------------------------
* Blend weights are searched over the full 4-simplex.  The submitted grid only
  searched the first two weights and split the remainder equally between
  FullConcat-Cox and the neural score, which is why those two columns of the
  blend-weight table were always identical.
* ``ModelFlags`` switch individual branches off for genuine (re-trained)
  ablations; ``n_layers=0`` removes message passing; ``use_priors=False``
  removes the prior term from the attention logits.
* The blended score is recalibrated with a one-dimensional Cox slope estimated
  on the internal validation split, and a Breslow baseline hazard is fitted on
  the training fold, so that survival probabilities (Brier, IBS, calibration)
  are on a proper log-hazard scale.  Ranking is unchanged.
* Interpretability is quantified by relation gates, attention divergence from
  uniform (only meaningful for destinations with in-degree > 1), relation /
  node-type / pathway knockouts on held-out patients, and branch variance
  shares.  Mean attention per relation is NOT reported because it equals
  1 / mean in-degree by construction.
"""

from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
try:
    # Register the optional backend BEFORE the first CPU autograd call. Lazy
    # registration after CPU backward leaves torch 2.4's device queues stale.
    import torch_directml as _torch_directml
except ImportError:
    _torch_directml = None
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

from hetero_data import GraphDataset, GraphSchema
from model_api import ModelFlags  # noqa: F401  (torch-free definition)
from survival_utils import BreslowSurvival, CoxPHRegression, concordance_index_fast, cox_fit_newton
from optimizers import backend_adamw


COMPONENTS: Tuple[str, ...] = ("clinical_cox", "subtype_cox", "full_cox", "heterogat")


def _safe_buffer_name(prefix: str, relation: str) -> str:
    return f"{prefix}_{re.sub(r'[^a-zA-Z0-9_]', '_', relation)}"


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(request: str = "auto") -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        try:
            import torch_directml
            if torch_directml.device_count():
                torch.backends.mha.set_fastpath_enabled(False)
                return torch_directml.device()
        except ImportError:
            pass
        return torch.device('cpu')
    if request.startswith('directml'):
        import torch_directml
        index=int(request.split(':')[1]) if ':' in request else 0
        if index>=torch_directml.device_count():
            raise RuntimeError('Requested DirectML adapter is unavailable')
        torch.backends.mha.set_fastpath_enabled(False)
        return torch_directml.device(index)
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def cox_partial_likelihood_loss(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow ties (mean per event).

    Every event at time t must include ALL patients observed at t in its
    risk set, including patients censored at t. A plain cumulative sum uses
    arbitrary within-tie ordering and produces a different loss/gradient.
    """
    if risk.device.type=='privateuseone':
        # DirectML has no native logcumsumexp/unique-consecutive kernel.
        # Only scalar risks and targets cross to CPU; the GNN stays on GPU.
        # ToCopyBackward preserves the exact full-risk-set gradient.
        return cox_partial_likelihood_loss(risk.cpu(),times.cpu(),events.cpu()).to(risk.device)
    risk = risk.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1).float()
    order = torch.argsort(times, descending=True)
    ordered_risk = risk[order]
    ordered_events = events[order]
    log_risk_sum = torch.logcumsumexp(ordered_risk, dim=0)
    _, counts = torch.unique_consecutive(times[order], return_counts=True)
    ends = counts.cumsum(0) - 1
    log_risk_sum = torch.repeat_interleave(log_risk_sum[ends], counts)
    contributions = ordered_risk - log_risk_sum
    n_events = ordered_events.sum().clamp_min(1.0)
    return -(contributions * ordered_events).sum() / n_events


def _standardize(risk: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(risk, dtype=float) - mean) / (std if std > 1e-8 else 1.0)


def _simplex_grid(k: int, steps: int):
    if k == 1:
        yield (steps,)
        return
    for i in range(steps + 1):
        for rest in _simplex_grid(k - 1, steps - i):
            yield (i,) + rest


def optimize_blend_weights(
    component_risks: Sequence[np.ndarray],
    times: np.ndarray,
    events: np.ndarray,
    grid: int = 13,
    component_stats: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, float]:
    """Non-negative weights on the simplex maximising the validation C-index.

    All compositions with step 1/(grid-1) are evaluated (455 points for four
    components and grid=13).  Exact ties are resolved in favour of the most
    even weight vector (highest entropy).
    """
    risks = [np.asarray(r, dtype=float).reshape(-1) for r in component_risks]
    k = len(risks)
    if k == 0:
        raise ValueError("No components to blend")
    standardized = []
    if component_stats is not None and len(component_stats) != k:
        raise ValueError("One training mean/std is required for every component")
    for i, r in enumerate(risks):
        mean, sd = component_stats[i] if component_stats is not None else (float(np.mean(r)), float(np.std(r)))
        standardized.append(_standardize(r, mean, sd))
    if k == 1:
        return np.ones(1), float(concordance_index_fast(times, standardized[0], events))
    S = np.vstack(standardized)  # k x n
    steps = grid - 1
    best_w = None
    best_c = -np.inf
    best_h = -np.inf
    for comp in _simplex_grid(k, steps):
        w = np.asarray(comp, dtype=float) / steps
        c = float(concordance_index_fast(times, w @ S, events))
        nz = w[w > 0]
        h = float(-(nz * np.log(nz)).sum())
        if c > best_c + 1e-9 or (abs(c - best_c) <= 1e-9 and h > best_h):
            best_w, best_c, best_h = w, c, h
    return best_w, best_c


def _predict_subtype_clinical(global_cox, subtype_models, clinical_x, subtype_ids) -> np.ndarray:
    risk = global_cox.predict_risk(clinical_x)
    for subtype_id, model in subtype_models.items():
        mask = subtype_ids == subtype_id
        if mask.any():
            risk = risk.copy()
            risk[mask] = model.predict_risk(clinical_x[mask])
    return risk


# =====================================================================
# Network
# =====================================================================


class RelationalGATLayer(nn.Module):
    """Sparse edge-list multi-head attention with relation-specific parameters."""

    def __init__(self, hidden_dim: int, n_heads: int, relation_names: Tuple[str, ...],
                 dropout: float = 0.15, prior_strength: float = 0.5, use_priors: bool = True) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.relation_names = relation_names
        self.use_priors = use_priors
        self.dropout = nn.Dropout(dropout)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source_projection = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim, bias=False) for r in relation_names})
        self.destination_projection = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim, bias=False) for r in relation_names})
        self.attention_source = nn.ParameterDict({r: nn.Parameter(torch.empty(n_heads, self.head_dim)) for r in relation_names})
        self.attention_destination = nn.ParameterDict({r: nn.Parameter(torch.empty(n_heads, self.head_dim)) for r in relation_names})
        self.relation_gate = nn.ParameterDict({r: nn.Parameter(torch.tensor(-0.5)) for r in relation_names})
        initial = math.log(math.exp(max(prior_strength, 1e-4)) - 1.0)
        self.prior_strength_raw = nn.Parameter(torch.tensor(initial, dtype=torch.float32), requires_grad=use_priors)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()
        self.force_dense_attention=False
        self.force_fused_attention=False
        self._layout_names={}

    def prepare_directml_layouts(self,schema):
        """Fixed graph layouts for native DirectML attention, without scatter."""
        for number,relation in enumerate(self.relation_names):
            edges=schema.relation_edges[relation]
            dest=edges[1]
            unique=np.unique(dest)
            if not len(unique):
                continue
            groups=[np.where(dest==v)[0] for v in unique]
            width=max(map(len,groups))
            gather=np.zeros((len(groups),width),dtype=np.int64)
            valid=np.zeros((len(groups),width),dtype=bool)
            inverse=np.zeros(len(dest),dtype=np.int64)
            for row,indices in enumerate(groups):
                gather[row,:len(indices)]=indices
                valid[row,:len(indices)]=True
                inverse[indices]=row*width+np.arange(len(indices))
            incidence=np.zeros((len(dest),schema.n_nodes),dtype=np.float32)
            incidence[np.arange(len(dest)),dest]=1
            names=[]
            for label,values in [('gather',gather),('valid',valid),('inverse',inverse),('incidence',incidence)]:
                name=f'_dml_{number}_{label}'
                self.register_buffer(name,torch.as_tensor(values),persistent=False)
                names.append(name)
            self._layout_names[relation]=tuple(names)
        all_edges=[];rel_ids=[];offsets=[];cursor=0
        for rid,relation in enumerate(self.relation_names):
            edges=schema.relation_edges[relation]
            length=edges.shape[1]
            all_edges.append(edges);rel_ids.extend([rid]*length)
            offsets.append((relation,cursor,cursor+length));cursor+=length
        edges=np.concatenate(all_edges,axis=1)
        rel_ids=np.asarray(rel_ids,dtype=np.int64)
        destinations=edges[1]
        incidence=np.zeros((len(destinations),schema.n_nodes),dtype=np.float32)
        incidence[np.arange(len(destinations)),destinations]=1
        for name,values in [('_fused_source',edges[0]),('_fused_dest',destinations),
                            ('_fused_relation',rel_ids),('_fused_incidence',incidence)]:
            self.register_buffer(name,torch.as_tensor(values),persistent=False)
        groups={}
        keys=rel_ids*schema.n_nodes+destinations
        for key in np.unique(keys):
            group=np.where(keys==key)[0]
            groups.setdefault(len(group),[]).append(group)
        self._degree_buckets=[];permutation=[]
        for degree,members in sorted(groups.items()):
            values=np.stack(members)
            name=f'_fused_degree_{degree}'
            self.register_buffer(name,torch.as_tensor(values),persistent=False)
            self._degree_buckets.append((name,len(members),degree))
            permutation.extend(values.reshape(-1))
        inverse=np.empty(len(keys),dtype=np.int64)
        inverse[np.asarray(permutation)]=np.arange(len(keys))
        self.register_buffer('_fused_inverse',torch.as_tensor(inverse),persistent=False)
        self._relation_offsets=offsets

    def _fused_forward(self,x,relation_priors,return_attention,drop_relations):
        """All relation projections and message aggregation in shared GEMMs.

        Degree buckets contain exactly the observed edges, so softmax never
        pads the homogeneous graph to (all nodes x maximum in-degree).
        Parameters and relation-specific normalization match the sparse path.
        """
        batch,n_nodes,_=x.shape
        relations=self.relation_names
        source_weight=torch.cat([self.source_projection[r].weight for r in relations],dim=0)
        projected=F.linear(x,source_weight).view(batch,n_nodes,len(relations),self.n_heads,self.head_dim)
        source=projected[:,self._fused_source,self._fused_relation,:,:]
        source_attention=torch.stack([self.attention_source[r] for r in relations])[self._fused_relation]
        scores=(source*source_attention.unsqueeze(0)).sum(-1)
        destination_kernels=torch.stack([(self.destination_projection[r].weight.view(
            self.n_heads,self.head_dim,self.hidden_dim)*self.attention_destination[r].unsqueeze(-1)).sum(1)
            for r in relations])
        dst_all=F.linear(x,destination_kernels.reshape(-1,self.hidden_dim)).view(batch,n_nodes,len(relations),self.n_heads)
        scores=(scores+dst_all[:,self._fused_dest,self._fused_relation,:]).permute(0,2,1)
        if self.use_priors:
            prior=torch.cat([relation_priors[r] for r in relations]).clamp_min(1e-4)
            scores=scores+F.softplus(self.prior_strength_raw)*torch.log(prior).view(1,1,-1)
        # DirectML LeakyReLU backward misinterprets non-contiguous strides
        # after the (batch,edge,head)->(batch,head,edge) permutation. Forward
        # still matches CPU, so a prediction-only test does not catch it.
        scores=F.leaky_relu(scores.contiguous(),negative_slope=.2)
        normalized=[]
        for name,count,degree in self._degree_buckets:
            ids=getattr(self,name).reshape(-1)
            group=scores[:,:,ids].reshape(batch,self.n_heads,count,degree)
            centered=(group-group.max(-1,keepdim=True).values.detach()).clamp(-30,30)
            exponents=centered.exp()
            alpha=exponents/exponents.sum(-1,keepdim=True).clamp_min(1e-9)
            normalized.append(alpha.reshape(batch,self.n_heads,-1))
        alpha=torch.cat(normalized,dim=-1)[:,:,self._fused_inverse]
        attention={r:alpha[:,:,start:stop].detach() for r,start,stop in self._relation_offsets
                   if return_attention and stop>start and r not in set(drop_relations)}
        gates=torch.sigmoid(torch.stack([self.relation_gate[r] for r in relations]))
        if drop_relations:
            active=x.new_tensor([float(r not in set(drop_relations)) for r in relations])
            gates=gates*active
        edge_gates=gates[self._fused_relation].view(1,-1,1,1)
        messages=source*self.dropout(alpha).permute(0,2,1).unsqueeze(-1)*edge_gates
        aggregated=torch.matmul(messages.permute(0,2,3,1),self._fused_incidence)
        aggregated=aggregated.permute(0,3,1,2).reshape(batch,n_nodes,self.hidden_dim)
        updated=self.output_projection(self.self_projection(x)+aggregated)
        activated=(torch.where(updated>=0,updated,torch.exp(updated.clamp(max=0))-1)
                   if x.device.type=='privateuseone' else F.elu(updated))
        updated=self.norm(x+self.dropout(activated))
        return updated,(attention if return_attention else None)

    def _directml_attention(self,scores,relation):
        gather,valid,inverse,_=[getattr(self,n) for n in self._layout_names[relation]]
        batch,heads,_=scores.shape
        padded=scores[:,:,gather.reshape(-1)].reshape(batch,heads,*gather.shape)
        padded=padded.masked_fill(~valid.view(1,1,*valid.shape),-1e30)
        centered=(padded-padded.max(dim=-1,keepdim=True).values.detach()).clamp(-30,30)
        exponents=centered.exp()*valid.view(1,1,*valid.shape).to(scores.dtype)
        normalized=exponents/exponents.sum(dim=-1,keepdim=True).clamp_min(1e-9)
        return normalized.reshape(batch,heads,-1)[:,:,inverse]

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.self_projection.weight)
        for relation in self.relation_names:
            nn.init.xavier_uniform_(self.source_projection[relation].weight)
            nn.init.xavier_uniform_(self.destination_projection[relation].weight)
            nn.init.xavier_uniform_(self.attention_source[relation])
            nn.init.xavier_uniform_(self.attention_destination[relation])
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def prior_strength(self) -> float:
        return float(F.softplus(self.prior_strength_raw).detach().cpu()) if self.use_priors else 0.0

    @staticmethod
    def _edge_softmax(scores: torch.Tensor, destinations: torch.Tensor, n_nodes: int) -> torch.Tensor:
        batch, heads, n_edges = scores.shape
        index = destinations.view(1, 1, n_edges).expand(batch, heads, n_edges)
        max_values = torch.full((batch, heads, n_nodes), -torch.inf, dtype=scores.dtype, device=scores.device)
        max_values.scatter_reduce_(2, index, scores, reduce="amax", include_self=True)
        centered = scores - max_values.gather(2, index)
        exponentials = torch.exp(torch.clamp(centered, min=-30.0, max=30.0))
        denominators = torch.zeros((batch, heads, n_nodes), dtype=scores.dtype, device=scores.device)
        denominators.scatter_add_(2, index, exponentials)
        return exponentials / denominators.gather(2, index).clamp_min(1e-9)

    def forward(self, x: torch.Tensor, relation_edges: Dict[str, torch.Tensor],
                relation_priors: Dict[str, torch.Tensor], return_attention: bool = False,
                drop_relations: Iterable[str] = ()):
        if x.device.type=='privateuseone' or getattr(self,'force_fused_attention',False):
            return self._fused_forward(x,relation_priors,return_attention,drop_relations)
        batch, n_nodes, _ = x.shape
        accumulated = self.self_projection(x)
        attention_output: Dict[str, torch.Tensor] = {}
        drop = set(drop_relations)
        prior_strength = F.softplus(self.prior_strength_raw) if self.use_priors else None
        for relation in self.relation_names:
            edge_index = relation_edges[relation]
            if edge_index.numel() == 0 or relation in drop:
                continue
            sources, destinations = edge_index[0], edge_index[1]
            n_edges = sources.numel()
            source_states = self.source_projection[relation](x[:, sources, :])
            source_states = source_states.view(batch, n_edges, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            scores = (source_states * self.attention_source[relation].view(1, self.n_heads, 1, self.head_dim)).sum(dim=-1)
            # (W_d x) dot a_d == x dot (W_d^T a_d). The destination
            # projection is used only in this scalar, never as a message.
            # Contract weights first to avoid materialising B x E x D states.
            destination_kernel = (self.destination_projection[relation].weight.view(
                self.n_heads, self.head_dim, self.hidden_dim)
                * self.attention_destination[relation].unsqueeze(-1)).sum(dim=1)
            scores = scores + F.linear(x[:, destinations, :], destination_kernel).permute(0, 2, 1)
            if self.use_priors:
                prior = relation_priors[relation].view(1, 1, n_edges).clamp_min(1e-4)
                scores = scores + prior_strength * torch.log(prior)
            scores = F.leaky_relu(scores, negative_slope=0.2)
            dense=x.device.type=='privateuseone' or getattr(self,'force_dense_attention',False)
            alpha = self._directml_attention(scores,relation) if dense else self._edge_softmax(scores, destinations, n_nodes)
            if return_attention:
                attention_output[relation] = alpha.detach()
            alpha = self.dropout(alpha)
            messages = source_states * alpha.unsqueeze(-1)
            if dense:
                incidence=getattr(self,self._layout_names[relation][3])
                relation_result=torch.matmul(messages.permute(0,1,3,2),incidence).permute(0,1,3,2)
            else:
                relation_result = torch.zeros((batch, self.n_heads, n_nodes, self.head_dim), dtype=x.dtype, device=x.device)
                destination_index = destinations.view(1, 1, n_edges, 1).expand(batch, self.n_heads, n_edges, self.head_dim)
                relation_result.scatter_add_(2, destination_index, messages)
            relation_result = relation_result.permute(0, 2, 1, 3).reshape(batch, n_nodes, self.hidden_dim)
            gate = torch.sigmoid(self.relation_gate[relation])
            accumulated = accumulated + gate * relation_result
        updated = self.output_projection(accumulated)
        # DirectML lacks a native ELU kernel. This is the same alpha=1 ELU
        # expressed with supported elementwise operations.
        activated=(torch.where(updated>=0,updated,torch.exp(updated.clamp(max=0))-1)
                   if updated.device.type=='privateuseone' else F.elu(updated))
        updated = self.dropout(activated)
        updated = self.norm(x + updated)
        return updated, (attention_output if return_attention else None)


class HeteroGATSurvivalNet(nn.Module):
    """Clinical-primary heterogeneous GAT with subtype, pathway and graph residuals."""

    def __init__(self, schema: GraphSchema, clinical_features: Sequence[str], hidden_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.15, prior_strength: float = 0.5,
                 n_subtypes: int = 5, flags: Optional[ModelFlags] = None,
                 graph_scale_init: float = -2.5, graph_head_gain: float = 0.1) -> None:
        super().__init__()
        if n_layers not in (0, 1, 2, 3):
            raise ValueError("n_layers should be 0, 1, 2 or 3")
        self.flags = copy.deepcopy(flags) if flags is not None else ModelFlags()
        if n_subtypes <= 1:
            self.flags.use_subtype_residual = False
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.graph_scale_init = float(graph_scale_init)
        self.graph_head_gain = float(graph_head_gain)
        self.relation_names = schema.relation_names
        self.patient_node_index = schema.patient_node_index
        self.clinical_features = tuple(clinical_features)

        clinical_indices = [schema.clinical_node_indices[name] for name in self.clinical_features]
        pathway_indices = list(schema.pathway_node_indices.values())
        self.register_buffer("clinical_indices", torch.as_tensor(clinical_indices, dtype=torch.long))
        self.register_buffer("pathway_indices", torch.as_tensor(pathway_indices, dtype=torch.long))
        n_clinical = len(clinical_indices)
        n_pathway = max(len(pathway_indices), 1)

        self.clinical_risk_head = nn.Linear(n_clinical, 1)
        self.subtype_clinical_weight = nn.Embedding(max(n_subtypes, 1), n_clinical)
        self.subtype_clinical_bias = nn.Embedding(max(n_subtypes, 1), 1)
        self.subtype_clinical_scale_raw = nn.Parameter(torch.tensor(-1.0))
        self.pathway_risk_head = nn.Linear(n_pathway, 1)
        self.pathway_scale_raw = nn.Parameter(torch.tensor(-2.0))

        self.input_projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.node_embedding = nn.Embedding(schema.n_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(int(schema.node_type_ids.max()) + 1, hidden_dim)
        self.layers = nn.ModuleList([
            RelationalGATLayer(hidden_dim, n_heads, self.relation_names, dropout, prior_strength, self.flags.use_priors)
            for _ in range(n_layers)
        ])
        for layer in self.layers:
            layer.prepare_directml_layouts(schema)
        self.patient_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.graph_risk_head = nn.Linear(hidden_dim, 1)
        self.graph_scale_raw = nn.Parameter(torch.tensor(self.graph_scale_init))
        self.subtype_graph_weight = nn.Embedding(max(n_subtypes, 1), hidden_dim)
        self.subtype_graph_bias = nn.Embedding(max(n_subtypes, 1), 1)
        self.subtype_graph_scale_raw = nn.Parameter(torch.tensor(-2.5))

        self.register_buffer("node_type_ids", torch.as_tensor(schema.node_type_ids, dtype=torch.long))
        self._edge_buffer_names: Dict[str, Tuple[str, str]] = {}
        for relation in self.relation_names:
            edge_name = _safe_buffer_name("edge", relation)
            prior_name = _safe_buffer_name("prior", relation)
            self.register_buffer(edge_name, torch.as_tensor(schema.relation_edges[relation], dtype=torch.long))
            self.register_buffer(prior_name, torch.as_tensor(schema.relation_priors[relation], dtype=torch.float32))
            self._edge_buffer_names[relation] = (edge_name, prior_name)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.clinical_risk_head.weight)
        nn.init.zeros_(self.clinical_risk_head.bias)
        prior_weights = {
            "age_at_diagnosis": 0.15, "lymph_nodes_positive": 0.25, "node_group": 0.25, "grade": 0.12,
            "stage": 0.18, "npi": 0.20, "tumor_size": 0.08, "er_status": -0.12, "pr_status": -0.08,
            "her2_status": 0.05,
        }
        with torch.no_grad():
            weight = self.clinical_risk_head.weight.view(-1)
            for name, value in prior_weights.items():
                if name in self.clinical_features:
                    weight[self.clinical_features.index(name)] = value
        nn.init.normal_(self.subtype_clinical_weight.weight, std=0.01)
        nn.init.zeros_(self.subtype_clinical_bias.weight)
        nn.init.zeros_(self.pathway_risk_head.weight)
        nn.init.zeros_(self.pathway_risk_head.bias)
        nn.init.normal_(self.node_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.graph_risk_head.weight,
                                gain=1.0 if self.flags.graph_only else self.graph_head_gain)
        nn.init.zeros_(self.graph_risk_head.bias)
        nn.init.normal_(self.subtype_graph_weight.weight, std=0.01)
        nn.init.zeros_(self.subtype_graph_bias.weight)

    def _relation_tensors(self):
        edges, priors = {}, {}
        for relation, (edge_name, prior_name) in self._edge_buffer_names.items():
            edges[relation] = getattr(self, edge_name)
            priors[relation] = getattr(self, prior_name)
        return edges, priors

    def clinical_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.flags.graph_only:
            return params
        if self.flags.use_clinical_head:
            params.extend(self.clinical_risk_head.parameters())
        if self.flags.use_subtype_residual:
            params.extend(self.subtype_clinical_weight.parameters())
            params.extend(self.subtype_clinical_bias.parameters())
            params.append(self.subtype_clinical_scale_raw)
        if self.flags.use_pathway_residual:
            params.extend(self.pathway_risk_head.parameters())
            params.append(self.pathway_scale_raw)
        return params

    def graph_parameters(self) -> List[nn.Parameter]:
        """Parameters of the graph branch (independent of requires_grad state)."""
        if not (self.flags.use_graph_residual or self.flags.graph_only):
            return []
        excluded = {id(p) for m in (self.clinical_risk_head, self.subtype_clinical_weight,
                                    self.subtype_clinical_bias, self.pathway_risk_head) for p in m.parameters()}
        excluded |= {id(self.subtype_clinical_scale_raw), id(self.pathway_scale_raw)}
        if not self.flags.use_priors:
            excluded |= {id(layer.prior_strength_raw) for layer in self.layers}
        if self.flags.graph_only or not self.flags.use_subtype_residual:
            excluded |= {id(p) for m in (self.subtype_graph_weight, self.subtype_graph_bias) for p in m.parameters()}
            excluded.add(id(self.subtype_graph_scale_raw))
        if self.flags.graph_only:
            excluded.add(id(self.graph_scale_raw))
        return [p for p in self.parameters() if id(p) not in excluded]

    def graph_embedding(self, node_features: torch.Tensor, return_attention: bool = False,
                        drop_relations: Iterable[str] = ()):
        if self.flags.mask_clinical_graph_inputs:
            node_features = node_features.clone()
            node_features[:, self.clinical_indices, :] = 0
        node_ids = torch.arange(self.schema.n_nodes, device=node_features.device)
        h = self.input_projection(node_features)
        h = h + self.node_embedding(node_ids).unsqueeze(0)
        h = h + self.type_embedding(self.node_type_ids).unsqueeze(0)
        edges, priors = self._relation_tensors()
        attentions: Dict[str, List[torch.Tensor]] = {r: [] for r in self.relation_names}
        for layer in self.layers:
            h, layer_attention = layer(h, edges, priors, return_attention=return_attention, drop_relations=drop_relations)
            if return_attention and layer_attention is not None:
                for relation, values in layer_attention.items():
                    attentions[relation].append(values)
        patient_state = self.patient_projection(torch.cat([h[:, self.patient_node_index, :], h.mean(dim=1)], dim=-1))
        return patient_state, attentions

    def forward(self, node_features: torch.Tensor, subtype_ids: torch.Tensor, return_attention: bool = False,
                return_components: bool = False, drop_relations: Iterable[str] = ()):
        batch = node_features.shape[0]
        zeros = node_features.new_zeros(batch)
        flags = self.flags
        clinical_x = node_features[:, self.clinical_indices, 0]
        comp: Dict[str, torch.Tensor] = {}
        if flags.use_clinical_head and not flags.graph_only:
            comp["clinical"] = self.clinical_risk_head(clinical_x).squeeze(-1)
        else:
            comp["clinical"] = zeros
        if flags.use_subtype_residual and not flags.graph_only:
            subtype_w = self.subtype_clinical_weight(subtype_ids)
            subtype_b = self.subtype_clinical_bias(subtype_ids).squeeze(-1)
            term = (clinical_x * subtype_w).sum(dim=-1) / math.sqrt(max(clinical_x.shape[-1], 1)) + subtype_b
            comp["subtype"] = torch.sigmoid(self.subtype_clinical_scale_raw) * term
        else:
            comp["subtype"] = zeros
        if flags.use_pathway_residual and not flags.graph_only:
            pathway_x = node_features[:, self.pathway_indices, 0]
            comp["pathway"] = torch.sigmoid(self.pathway_scale_raw) * self.pathway_risk_head(pathway_x).squeeze(-1)
        else:
            comp["pathway"] = zeros
        attentions: Dict[str, List[torch.Tensor]] = {}
        graph_raw = zeros
        if flags.use_graph_residual or flags.graph_only:
            patient_state, attentions = self.graph_embedding(node_features, return_attention, drop_relations)
            graph_risk = self.graph_risk_head(patient_state).squeeze(-1)
            graph_raw = graph_risk
            if flags.graph_only:
                comp["graph"] = graph_risk
            else:
                term = torch.sigmoid(self.graph_scale_raw) * graph_risk
                if flags.use_subtype_residual:
                    sw = self.subtype_graph_weight(subtype_ids)
                    sb = self.subtype_graph_bias(subtype_ids).squeeze(-1)
                    graph_subtype = (patient_state * sw).sum(dim=-1) / math.sqrt(self.hidden_dim) + sb
                    term = term + 0.5 * torch.sigmoid(self.subtype_graph_scale_raw) * graph_subtype
                comp["graph"] = term
        else:
            comp["graph"] = zeros
        risk = comp["clinical"] + comp["subtype"] + comp["pathway"] + comp["graph"]
        if not (return_attention or return_components):
            return risk
        extras: Dict[str, object] = {}
        if return_components:
            extras["components"] = comp
            extras["graph_raw"] = graph_raw
        if return_attention:
            extras["attention"] = {r: torch.stack(v, dim=0) for r, v in attentions.items() if v}
        return risk, extras

    def subtype_penalty(self) -> torch.Tensor:
        total = self.graph_risk_head.weight.pow(2).mean() if not self.flags.graph_only else torch.zeros((), device=self.graph_risk_head.weight.device)
        if self.flags.use_subtype_residual and not self.flags.graph_only:
            total = total + self.subtype_clinical_weight.weight.pow(2).mean() + self.subtype_clinical_bias.weight.pow(2).mean()
            total = total + self.subtype_graph_weight.weight.pow(2).mean() + self.subtype_graph_bias.weight.pow(2).mean()
        return total


# =====================================================================
# Estimator
# =====================================================================


@dataclass
class TrainingHistory:
    epoch: List[int]
    train_loss: List[float]
    validation_cindex: List[float]
    best_epoch: int
    best_validation_cindex: float
    blend_weights: Dict[str, Dict[str, float]]
    blend_validation_cindex: Dict[str, float]


@dataclass
class BlendState:
    components: Tuple[str, ...]
    weights: Dict[str, float]
    validation_cindex: float
    calibration_slope: float = 1.0
    breslow: Optional[BreslowSurvival] = None


def _risk_only(model, x: torch.Tensor, subtypes: torch.Tensor) -> torch.Tensor:
    return model(x, subtypes)


def _risk_and_graph_raw(model, x: torch.Tensor, subtypes: torch.Tensor):
    risk, extras = model(x, subtypes, return_components=True)
    return risk, extras["graph_raw"]


class HeteroGATSurvivalEstimator:
    """KG-MSPF estimator: neural risk plus validation-selected linear anchors."""

    def __init__(
        self,
        hidden_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        prior_strength: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        subtype_penalty: float = 1e-3,
        max_epochs: int = 120,
        patience: int = 40,
        validation_fraction: float = 0.25,
        clinical_warmup_epochs: int = 25,
        refit_full: bool = True,
        seed: int = 42,
        device: str = "auto",
        verbose: bool = True,
        flags: Optional[ModelFlags] = None,
        blend_components: Sequence[str] = COMPONENTS,
        extra_blends: Optional[Dict[str, Sequence[str]]] = None,
        blend_grid: int = 13,
        cox_alpha: float = 1e-6,
        train_chunk_size: int = 0,
        graph_scale_init: float = 0.0,
        graph_head_gain: float = 1.0,
        graph_aux_weight: float = 0.0,
        min_epochs: int = 60,
        gradient_mode: str = "auto",
        inference_batch_size: int = 128,
        clinical_offset: bool = False,
        clinical_initialization: str = "heuristic",
        graph_learning_rate_ratio: Optional[float] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.prior_strength = prior_strength
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.subtype_penalty_coefficient = subtype_penalty
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.clinical_warmup_epochs = clinical_warmup_epochs
        self.refit_full = refit_full
        self.seed = seed
        self.device_request = device
        self.verbose = verbose
        self.flags = flags or ModelFlags()
        self.blend_components = tuple(blend_components)
        self.extra_blends = {k: tuple(v) for k, v in (extra_blends or {}).items()}
        self.blend_grid = blend_grid
        self.cox_alpha = cox_alpha
        self.train_chunk_size = int(train_chunk_size or 0)
        self.graph_scale_init = float(graph_scale_init)
        self.graph_head_gain = float(graph_head_gain)
        self.graph_aux_weight = float(graph_aux_weight)
        self.min_epochs = min(int(min_epochs or 0), int(max_epochs))
        if gradient_mode not in {"auto","full","streamed", "checkpoint"}:
            raise ValueError("gradient_mode must be auto, full, streamed or checkpoint")
        self.gradient_mode = gradient_mode
        self.inference_batch_size = int(inference_batch_size)
        self.clinical_offset = bool(clinical_offset)
        if clinical_initialization not in {"heuristic", "cox"}:
            raise ValueError("clinical_initialization must be heuristic or cox")
        self.clinical_initialization = clinical_initialization
        if graph_learning_rate_ratio is not None and graph_learning_rate_ratio <= 0:
            raise ValueError("graph_learning_rate_ratio must be positive")
        self.graph_learning_rate_ratio = graph_learning_rate_ratio
        if clinical_offset and not (self.flags.graph_only and self.flags.mask_clinical_graph_inputs):
            raise ValueError("clinical_offset requires a molecular graph with graph_only and masked clinical inputs")
        self.model: Optional[HeteroGATSurvivalNet] = None
        self.device = torch.device("cpu")
        self.history: Optional[TrainingHistory] = None
        self.clinical_cox: Optional[CoxPHRegression] = None
        self.full_cox: Optional[CoxPHRegression] = None
        self.subtype_cox: Dict[int, CoxPHRegression] = {}
        self.component_stats: Dict[str, Tuple[float, float]] = {}
        self.blends: Dict[str, BlendState] = {}
        self.clinical_features: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    @staticmethod
    def stratification_labels(dataset: GraphDataset) -> np.ndarray:
        combined = np.asarray([f"{s}_{e}" for s, e in zip(dataset.subtypes, dataset.events)])
        counts = {label: int((combined == label).sum()) for label in np.unique(combined)}
        if all(count >= 2 for count in counts.values()):
            return combined
        return dataset.events.astype(str)

    @staticmethod
    def _clinical_matrix(dataset: GraphDataset) -> np.ndarray:
        return dataset.block("clinical").astype(float)

    @staticmethod
    def _full_matrix(dataset: GraphDataset) -> np.ndarray:
        return dataset.flat_features.astype(float)

    def _build_model(self, dataset: GraphDataset) -> HeteroGATSurvivalNet:
        return HeteroGATSurvivalNet(
            schema=dataset.schema, clinical_features=dataset.clinical_features, hidden_dim=self.hidden_dim,
            n_heads=self.n_heads, n_layers=self.n_layers, dropout=self.dropout,
            prior_strength=self.prior_strength, n_subtypes=dataset.n_subtypes, flags=self.flags,
            graph_scale_init=self.graph_scale_init, graph_head_gain=self.graph_head_gain,
        ).to(self.device)

    def _needed_components(self) -> Tuple[str, ...]:
        needed = set(self.blend_components)
        for comps in self.extra_blends.values():
            needed |= set(comps)
        return tuple(c for c in COMPONENTS if c in needed)

    def _initialize_from_clinical_cox(self, model, anchor):
        """Start the clinical head at the training-only fitted Cox risk.

        CoxPHRegression standardizes internally, even when the graph's clinical
        values have already been scaled. Fold its centering/scaling into the
        linear head exactly; copying coef_ alone would give the wrong risk.
        Refitting calls this again with the outer-training anchor and schema.
        """
        if getattr(self, "clinical_initialization", "heuristic") != "cox":
            return
        if not model.flags.use_clinical_head or model.flags.graph_only:
            return
        weight = np.asarray(anchor.coef_) / np.asarray(anchor.scale_)
        bias = -float(np.asarray(anchor.mean_) @ weight)
        with torch.no_grad():
            model.clinical_risk_head.weight.copy_(torch.as_tensor(
                weight[None, :], dtype=model.clinical_risk_head.weight.dtype, device=self.device))
            model.clinical_risk_head.bias.fill_(bias)

    # ------------------------------------------------------------------
    def _fit_anchors(self, dataset: GraphDataset, indices: np.ndarray):
        clinical_x = self._clinical_matrix(dataset)[indices]
        T = dataset.times[indices]
        E = dataset.events[indices]
        needed = self._needed_components()
        clinical_cox = CoxPHRegression(alpha=self.cox_alpha).fit(clinical_x, T, E)
        full_cox = None
        if "full_cox" in needed:
            full_cox = CoxPHRegression(alpha=self.cox_alpha).fit(self._full_matrix(dataset)[indices], T, E)
        subtype_cox: Dict[int, CoxPHRegression] = {}
        if "subtype_cox" in needed and dataset.n_subtypes > 1:
            sub = dataset.subtype_ids[indices]
            for subtype_id in np.unique(sub):
                mask = sub == subtype_id
                if int(mask.sum()) < 40 or int(E[mask].sum()) < 12:
                    continue
                try:
                    subtype_cox[int(subtype_id)] = CoxPHRegression(alpha=self.cox_alpha).fit(clinical_x[mask], T[mask], E[mask])
                except Exception:
                    continue
        return clinical_cox, full_cox, subtype_cox

    def _component_risks(self, dataset: GraphDataset, model, clinical_cox, full_cox, subtype_cox) -> Dict[str, np.ndarray]:
        needed = self._needed_components()
        clinical_x = self._clinical_matrix(dataset)
        out: Dict[str, np.ndarray] = {}
        if "clinical_cox" in needed:
            out["clinical_cox"] = clinical_cox.predict_risk(clinical_x)
        if "subtype_cox" in needed:
            out["subtype_cox"] = _predict_subtype_clinical(clinical_cox, subtype_cox, clinical_x, dataset.subtype_ids)
        if "full_cox" in needed and full_cox is not None:
            out["full_cox"] = full_cox.predict_risk(self._full_matrix(dataset))
        if "heterogat" in needed:
            out["heterogat"] = self._neural_risk(model, dataset, offset_model=clinical_cox)
        return out

    # ------------------------------------------------------------------
    def _forward_train(self, model, x: torch.Tensor, subtypes: torch.Tensor, want_graph_raw: bool = False):
        """Forward pass used during training, with a bounded peak memory.

        The per-patient sub-graphs are independent, so the network can be
        evaluated on chunks of patients and the risks concatenated without
        changing the result.  ``torch.utils.checkpoint`` makes the activations
        be recomputed during the backward pass instead of being retained, which
        is what actually bounds peak RAM: without it every relation of every
        layer keeps its intermediate tensors alive until backward.  A chunk
        size >= the number of patients therefore still checkpoints, just in a
        single call.  ``train_chunk_size == 0`` keeps the original behaviour.

        ``want_graph_raw`` additionally returns the ungated graph risk, which is
        what the auxiliary graph supervision (``graph_aux_weight``) needs.
        """
        fn = _risk_and_graph_raw if want_graph_raw else _risk_only
        chunk = self.train_chunk_size
        if self.gradient_mode=='full' or (self.gradient_mode=='auto' and x.device.type=='privateuseone'):
            chunk=0
        if chunk>0 and x.device.type=='privateuseone':
            raise ValueError('DirectML checkpoint replay is unsupported; use gradient_mode=streamed for bounded memory or full/auto')
        if chunk <= 0:
            return fn(model, x, subtypes)
        risks, raws = [], []
        for start in range(0, x.shape[0], chunk):
            stop = min(start + chunk, x.shape[0])
            out = torch.utils.checkpoint.checkpoint(fn, model, x[start:stop], subtypes[start:stop],
                                                    use_reentrant=False)
            if want_graph_raw:
                risks.append(out[0])
                raws.append(out[1])
            else:
                risks.append(out)
        risk = risks[0] if len(risks) == 1 else torch.cat(risks, dim=0)
        if not want_graph_raw:
            return risk
        raw = raws[0] if len(raws) == 1 else torch.cat(raws, dim=0)
        return risk, raw

    def _backward_streamed(self, model, x, subtypes, times, events, offset=None):
        """Exact full-risk-set Cox gradient, retaining only one chunk's graph.

        First collect risks without activations. Differentiate the full Cox
        loss with respect to these risks, then replay each chunk with the SAME
        dropout random state and apply its vector-Jacobian product. No minibatch
        approximation to the Cox risk set is made. This model has no BatchNorm
        or mutable training buffers, so chunk replay does not update buffers.
        """
        chunk = self.train_chunk_size or 64
        want_aux = self.graph_aux_weight > 0 and bool(model.graph_parameters())
        risks, raws, states = [], [], []
        cuda = x.device.type == "cuda"
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                states.append((torch.get_rng_state(), torch.cuda.get_rng_state(x.device) if cuda else None))
                if want_aux:
                    r, g = _risk_and_graph_raw(model, x[start:start + chunk], subtypes[start:start + chunk])
                    raws.append(g)
                else:
                    r = model(x[start:start + chunk], subtypes[start:start + chunk])
                risks.append(r)
        final_cpu = torch.get_rng_state()
        final_cuda = torch.cuda.get_rng_state(x.device) if cuda else None
        risk = torch.cat(risks).detach().requires_grad_(True)
        loss = cox_partial_likelihood_loss(risk if offset is None else risk + offset, times, events)
        leaves = [risk]
        if want_aux:
            raw = torch.cat(raws).detach().requires_grad_(True)
            leaves.append(raw)
            loss = loss + self.graph_aux_weight * cox_partial_likelihood_loss(raw, times, events)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite streamed Cox loss")
        grads = torch.autograd.grad(loss, leaves)
        try:
            for j, start in enumerate(range(0, len(x), chunk)):
                torch.set_rng_state(states[j][0])
                if cuda:
                    torch.cuda.set_rng_state(states[j][1], x.device)
                if want_aux:
                    r, g = _risk_and_graph_raw(model, x[start:start + chunk], subtypes[start:start + chunk])
                    surrogate = (r * grads[0][start:start + chunk]).sum() + (g * grads[1][start:start + chunk]).sum()
                else:
                    r = model(x[start:start + chunk], subtypes[start:start + chunk])
                    surrogate = (r * grads[0][start:start + chunk]).sum()
                surrogate.backward()
        finally:
            torch.set_rng_state(final_cpu)
            if cuda:
                torch.cuda.set_rng_state(final_cuda, x.device)
        penalty = self.subtype_penalty_coefficient * model.subtype_penalty()
        if penalty.requires_grad:
            penalty.backward()
        return loss.detach() + penalty.detach()

    def _train_loop(self, model, x, subtypes, times, events, n_epochs, validation=None,
                    offset=None, validation_offset=None):
        """Two-phase training.  With ``validation`` the best epoch is tracked."""
        warmup = self.clinical_warmup_epochs if model.clinical_parameters() else 0
        has_graph = bool(model.graph_parameters())
        if not has_graph:
            warmup = n_epochs  # nothing to unfreeze
        phase = "clin" if warmup > 0 else "full"

        def make_optimizer(current_phase):
            for p in model.parameters():
                p.requires_grad_(False)
            if current_phase == "clin":
                params = model.clinical_parameters()
                for p in params:
                    p.requires_grad_(True)
                return backend_adamw(params, lr=self.learning_rate, weight_decay=self.weight_decay)
            groups = []
            clin = model.clinical_parameters()
            graph = model.graph_parameters()
            for p in clin + graph:
                p.requires_grad_(True)
            if clin:
                groups.append({"params": clin, "lr": self.learning_rate})
            if graph:
                ratio = getattr(self, "graph_learning_rate_ratio", None)
                if ratio is None:
                    ratio = 1.0 if model.flags.graph_only else 0.5
                groups.append({"params": graph,
                               "lr": self.learning_rate * ratio})
            return backend_adamw(groups, weight_decay=self.weight_decay)

        optimizer = make_optimizer(phase)
        best_state = copy.deepcopy(model.state_dict())
        best_c, best_epoch, since = -np.inf, 0, 0
        hist_e, hist_l, hist_c = [], [], []
        for epoch in range(1, n_epochs + 1):
            if phase == "clin" and epoch > warmup:
                phase = "full"
                optimizer = make_optimizer(phase)
                since = 0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if self.gradient_mode == "streamed" or (self.gradient_mode=='auto' and x.device.type!='privateuseone'):
                loss = self._backward_streamed(model, x, subtypes, times, events, offset=offset)
            else:
                if self.graph_aux_weight > 0 and bool(model.graph_parameters()):
                    risk, graph_raw = self._forward_train(model, x, subtypes, want_graph_raw=True)
                else:
                    risk, graph_raw = self._forward_train(model, x, subtypes), None
                loss = cox_partial_likelihood_loss(risk if offset is None else risk + offset, times, events) + self.subtype_penalty_coefficient * model.subtype_penalty()
                if graph_raw is not None:
                    loss = loss + self.graph_aux_weight * cox_partial_likelihood_loss(graph_raw, times, events)
                loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            optimizer.step()
            if validation is None:
                continue
            xv, sv, tv, ev = validation
            model.eval()
            with torch.no_grad():
                val_risk = torch.cat([model(xv[j:j + self.inference_batch_size], sv[j:j + self.inference_batch_size])
                                      for j in range(0, len(xv), self.inference_batch_size)]).cpu().numpy()
            if validation_offset is not None:
                val_risk = val_risk + validation_offset
            val_c = concordance_index_fast(tv, val_risk, ev)
            if not np.isfinite(val_c):
                val_c = -np.inf
            hist_e.append(epoch)
            hist_l.append(float(loss.detach().cpu()))
            hist_c.append(float(val_c))
            # Never deploy a warmup-only checkpoint when a graph must be trained.
            eligible = epoch >= self.min_epochs and (phase == "full" or not has_graph)
            if eligible and val_c > best_c + 1e-4:
                best_c, best_epoch, since = float(val_c), epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            elif eligible:
                since += 1
            if self.verbose and (epoch == 1 or epoch % 20 == 0):
                print(f"      epoch={epoch:03d}[{phase}] loss={hist_l[-1]:.4f} val_C={val_c:.4f} best={best_c:.4f}")
            if phase == "full" and eligible and since >= self.patience:
                break
        if validation is not None:
            model.load_state_dict(best_state)
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        return best_epoch, best_c, hist_e, hist_l, hist_c

    def _tensors(self, dataset: GraphDataset, indices: Optional[np.ndarray] = None):
        idx = np.arange(len(dataset)) if indices is None else indices
        return (
            torch.as_tensor(dataset.node_features[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(dataset.subtype_ids[idx], dtype=torch.long, device=self.device),
            torch.as_tensor(dataset.times[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(dataset.events[idx], dtype=torch.float32, device=self.device),
        )

    def _neural_risk(self, model, dataset: GraphDataset, batch_size: Optional[int] = None,
                     offset_model=None, **forward_kwargs) -> np.ndarray:
        batch_size = batch_size or self.inference_batch_size
        node_features = forward_kwargs.pop("node_features", dataset.node_features)
        risks = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                x = torch.as_tensor(node_features[start:stop], dtype=torch.float32, device=self.device)
                s = torch.as_tensor(dataset.subtype_ids[start:stop], dtype=torch.long, device=self.device)
                risks.append(model(x, s, **forward_kwargs).detach().cpu().numpy())
        result = np.concatenate(risks).astype(float)
        if self.clinical_offset:
            anchor = self.clinical_cox if offset_model is None else offset_model
            result = result + anchor.predict_risk(self._clinical_matrix(dataset))
        return result

    # ------------------------------------------------------------------
    def fit(self, dataset: GraphDataset, selection_dataset: Optional[GraphDataset] = None) -> "HeteroGATSurvivalEstimator":
        set_deterministic_seed(self.seed)
        self.device = resolve_device(self.device_request)
        self.clinical_features = dataset.clinical_features
        all_idx = np.arange(len(dataset))
        tr_idx, va_idx = train_test_split(all_idx, test_size=self.validation_fraction, random_state=self.seed,
                                          stratify=self.stratification_labels(dataset))
        selection = dataset if selection_dataset is None else selection_dataset
        if not (np.array_equal(selection.patient_ids, dataset.patient_ids)
                and np.array_equal(selection.times, dataset.times)
                and np.array_equal(selection.events, dataset.events)):
            raise ValueError("Selection and refit data must contain the same ordered patients/targets")
        clinical_cox, full_cox, subtype_cox = self._fit_anchors(selection, tr_idx)

        model = self._build_model(selection)
        self._initialize_from_clinical_cox(model, clinical_cox)
        x_tr, s_tr, t_tr, e_tr = self._tensors(selection, tr_idx)
        x_va, s_va, _, _ = self._tensors(selection, va_idx)
        best_epoch, best_c, he, hl, hc = self._train_loop(
            model, x_tr, s_tr, t_tr, e_tr, self.max_epochs,
            validation=(x_va, s_va, dataset.times[va_idx], dataset.events[va_idx]),
            offset=(torch.as_tensor(clinical_cox.predict_risk(self._clinical_matrix(selection)[tr_idx]),
                                    dtype=torch.float32, device=self.device) if self.clinical_offset else None),
            validation_offset=(clinical_cox.predict_risk(self._clinical_matrix(selection)[va_idx]) if self.clinical_offset else None),
        )

        val_ds = selection.subset(va_idx)
        val_components = self._component_risks(val_ds, model, clinical_cox, full_cox, subtype_cox)
        fit_components = self._component_risks(selection.subset(tr_idx), model, clinical_cox, full_cox, subtype_cox)
        selection_stats = {c: (float(np.mean(r)), float(np.std(r))) for c, r in fit_components.items()}
        blend_specs = {"main": self.blend_components, **self.extra_blends}
        self.blends = {}
        for name, comps in blend_specs.items():
            comps = tuple(c for c in comps if c in val_components)
            weights, c_val = optimize_blend_weights([val_components[c] for c in comps], val_ds.times, val_ds.events,
                                                    grid=self.blend_grid,
                                                    component_stats=[selection_stats[c] for c in comps])
            z = sum(w * _standardize(val_components[c], *selection_stats[c])
                    for w, c in zip(weights, comps))
            try:
                slope = float(cox_fit_newton(np.asarray(z)[:, None], val_ds.times, val_ds.events)["coef"][0])
            except Exception:
                slope = float("nan")
            if not np.isfinite(slope) or slope <= 0:
                slope = 1.0
            self.blends[name] = BlendState(components=comps, weights=dict(zip(comps, map(float, weights))),
                                           validation_cindex=float(c_val), calibration_slope=slope)
        if self.verbose:
            w = self.blends["main"].weights
            print("      blend " + " ".join(f"{k}={v:.2f}" for k, v in w.items()) +
                  f" val_C={self.blends['main'].validation_cindex:.4f}")

        if self.refit_full and best_epoch > 0:
            set_deterministic_seed(self.seed + 1009)
            self.clinical_cox, self.full_cox, self.subtype_cox = self._fit_anchors(dataset, all_idx)
            refit_model = self._build_model(dataset)
            self._initialize_from_clinical_cox(refit_model, self.clinical_cox)
            x, s, t, e = self._tensors(dataset)
            offset = (torch.as_tensor(self.clinical_cox.predict_risk(self._clinical_matrix(dataset)),
                                      dtype=torch.float32, device=self.device) if self.clinical_offset else None)
            self._train_loop(refit_model, x, s, t, e, best_epoch, validation=None, offset=offset)
            self.model = refit_model
        else:
            if selection_dataset is not None:
                raise ValueError("Nested preprocessing requires refit_full and a valid selected epoch")
            self.model = model
            self.clinical_cox, self.full_cox, self.subtype_cox = clinical_cox, full_cox, subtype_cox
        self.model.eval()

        train_components = self._component_risks(dataset, self.model, self.clinical_cox, self.full_cox, self.subtype_cox)
        self.component_stats = {name: (float(np.mean(r)), float(np.std(r)) if np.std(r) > 1e-8 else 1.0)
                                for name, r in train_components.items()}
        for name, state in self.blends.items():
            lp = state.calibration_slope * self._blend(train_components, name)
            state.breslow = BreslowSurvival().fit(lp, dataset.times, dataset.events)

        self.history = TrainingHistory(
            epoch=he, train_loss=hl, validation_cindex=hc, best_epoch=best_epoch, best_validation_cindex=best_c,
            blend_weights={k: v.weights for k, v in self.blends.items()},
            blend_validation_cindex={k: v.validation_cindex for k, v in self.blends.items()},
        )
        return self

    # ------------------------------------------------------------------
    def _blend(self, components: Dict[str, np.ndarray], blend: str = "main") -> np.ndarray:
        state = self.blends[blend]
        total = None
        for name in state.components:
            mean, std = self.component_stats[name]
            term = state.weights[name] * _standardize(components[name], mean, std)
            total = term if total is None else total + term
        return np.asarray(total, dtype=float)

    def predict_components(self, dataset: GraphDataset) -> Dict[str, np.ndarray]:
        return self._component_risks(dataset, self.model, self.clinical_cox, self.full_cox, self.subtype_cox)

    def predict_risk(self, dataset: GraphDataset, blend: str = "main") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Estimator has not been fitted")
        return self._blend(self.predict_components(dataset), blend)

    def predict_survival(self, dataset: GraphDataset, times: Sequence[float], blend: str = "main",
                         risk: Optional[np.ndarray] = None) -> np.ndarray:
        state = self.blends[blend]
        r = self.predict_risk(dataset, blend) if risk is None else risk
        return state.breslow.survival(state.calibration_slope * r, times)

    @property
    def blend_names(self) -> Tuple[str, ...]:
        return tuple(self.blends.keys())

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------
    def relation_gates(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for i, layer in enumerate(self.model.layers):
            out[f"layer{i + 1}"] = {r: float(torch.sigmoid(layer.relation_gate[r]).detach().cpu())
                                    for r in layer.relation_names}
            out[f"layer{i + 1}"]["prior_strength_gamma"] = layer.prior_strength()
        return out

    def branch_scales(self) -> Dict[str, float]:
        m = self.model
        return {
            "subtype_gate": float(torch.sigmoid(m.subtype_clinical_scale_raw).detach().cpu()),
            "pathway_gate": float(torch.sigmoid(m.pathway_scale_raw).detach().cpu()),
            "graph_gate": float(torch.sigmoid(m.graph_scale_raw).detach().cpu()),
        }

    def pathway_coefficients(self) -> Dict[str, float]:
        m = self.model
        if not m.flags.use_pathway_residual:
            return {}
        scale = float(torch.sigmoid(m.pathway_scale_raw).detach().cpu())
        w = m.pathway_risk_head.weight.detach().cpu().numpy().reshape(-1) * scale
        names = list(m.schema.pathway_node_indices.keys())
        return {n: float(v) for n, v in zip(names, w)}

    def clinical_coefficients(self) -> Dict[str, float]:
        w = self.model.clinical_risk_head.weight.detach().cpu().numpy().reshape(-1)
        return {n: float(v) for n, v in zip(self.clinical_features, w)}

    def branch_contributions(self, dataset: GraphDataset, max_patients: int = 2048) -> Dict[str, float]:
        """Variance share of each additive term of eta_nn on the given patients."""
        n = min(max_patients, len(dataset))
        x, s, _, _ = self._tensors(dataset, np.arange(n))
        self.model.eval()
        with torch.no_grad():
            risk, extras = self.model(x, s, return_components=True)
        comps = {k: v.detach().cpu().numpy() for k, v in extras["components"].items()}
        total = risk.detach().cpu().numpy()
        if self.clinical_offset:
            comps['clinical_offset'] = self.clinical_cox.predict_risk(self._clinical_matrix(dataset)[:n])
            total = total + comps['clinical_offset']
        var = {k: float(np.var(v)) for k, v in comps.items()}
        denom = sum(var.values()) or 1.0
        out = {f"var_share_{k}": v / denom for k, v in var.items()}
        for k, v in comps.items():
            out[f"corr_{k}_with_total"] = float(np.corrcoef(v, total)[0, 1]) if np.std(v) > 1e-12 else 0.0
        return out

    def attention_statistics(self, dataset: GraphDataset, max_patients: int = 512, top_k: int = 10):
        """Attention concentration per relation (KL from uniform) and top edges.

        Only destinations with in-degree > 1 are informative: with a single
        incoming edge the softmax weight is 1 by construction.
        """
        if not self.model.layers or not (self.model.flags.use_graph_residual or self.model.flags.graph_only):
            return {"relations": {}, "top_edges": []}
        n = min(max_patients, len(dataset))
        x, s, _, _ = self._tensors(dataset, np.arange(n))
        self.model.eval()
        with torch.no_grad():
            _, extras = self.model(x, s, return_attention=True)
        schema = self.model.schema
        rel_stats = {}
        top_edges = []
        for relation, alpha in extras["attention"].items():
            a = alpha.cpu().numpy()  # layers x batch x heads x edges
            edges = schema.relation_edges[relation]
            dst = edges[1]
            deg = np.bincount(dst, minlength=schema.n_nodes)
            multi = deg[dst] > 1
            if not multi.any():
                rel_stats[relation] = {"informative": False, "mean_in_degree": float(deg[deg > 0].mean())}
                continue
            incidence = np.zeros((edges.shape[1], schema.n_nodes))
            incidence[np.arange(edges.shape[1]), dst] = 1.0
            ent_terms = -a * np.log(np.clip(a, 1e-12, 1.0))
            entropy = ent_terms @ incidence  # layers x batch x heads x nodes
            dests = np.where(deg > 1)[0]
            log_deg = np.log(deg[dests])
            kl = log_deg - entropy[..., dests]
            norm_ent = entropy[..., dests] / log_deg
            rel_stats[relation] = {
                "informative": True,
                "mean_in_degree": float(deg[dests].mean()),
                "kl_from_uniform": float(kl.mean()),
                "kl_from_uniform_last_layer": float(kl[-1].mean()),
                "normalized_entropy": float(norm_ent.mean()),
                "max_edge_attention": float(a.mean(axis=(1, 2)).max()),
            }
            mean_alpha = a[-1].mean(axis=(0, 1))  # last layer, averaged over patients and heads
            ratio = mean_alpha * deg[dst]
            order = np.argsort(-ratio)
            for e_idx in order[:top_k]:
                if deg[dst[e_idx]] <= 1:
                    continue
                top_edges.append({
                    "relation": relation,
                    "source": schema.node_names[edges[0, e_idx]],
                    "destination": schema.node_names[dst[e_idx]],
                    "mean_attention": float(mean_alpha[e_idx]),
                    "uniform": float(1.0 / deg[dst[e_idx]]),
                    "ratio_to_uniform": float(ratio[e_idx]),
                })
        return {"relations": rel_stats, "top_edges": top_edges}

    def knockout_analysis(self, dataset: GraphDataset) -> List[Dict[str, object]]:
        """C-index change on held-out patients when parts of the graph are removed."""
        T, E = dataset.times, dataset.events
        base_comp = self.predict_components(dataset)
        base_nn = base_comp.get("heterogat")
        if base_nn is None:
            return []
        c_nn = concordance_index_fast(T, base_nn, E)
        c_kg = concordance_index_fast(T, self._blend(base_comp, "main"), E)
        rows: List[Dict[str, object]] = []

        def record(kind, name, nn_risk):
            comp = dict(base_comp)
            comp["heterogat"] = nn_risk
            rows.append({
                "kind": kind, "name": name,
                "c_nn": float(concordance_index_fast(T, nn_risk, E)),
                "delta_c_nn": float(concordance_index_fast(T, nn_risk, E) - c_nn),
                "delta_c_kg": float(concordance_index_fast(T, self._blend(comp, "main"), E) - c_kg),
                "mean_abs_change_nn": float(np.mean(np.abs(nn_risk - base_nn))),
            })

        if self.model.layers and (self.model.flags.use_graph_residual or self.model.flags.graph_only):
            for relation in self.model.relation_names:
                record("relation", relation, self._neural_risk(self.model, dataset, drop_relations=(relation,)))
        schema = self.model.schema
        type_ids = schema.node_type_ids
        for t_id, t_name in enumerate(("gene", "pathway", "clinical", "subtype")):
            nodes = np.where(type_ids == t_id)[0]
            if nodes.size == 0:
                continue
            nf = dataset.node_features.copy()
            nf[:, nodes, :] = 0.0
            record("node_type", t_name, self._neural_risk(self.model, dataset, node_features=nf))
        for pathway, node_idx in schema.pathway_node_indices.items():
            nf = dataset.node_features.copy()
            nf[:, node_idx, :] = 0.0
            record("pathway", pathway, self._neural_risk(self.model, dataset, node_features=nf))
        for row in rows:
            row["base_c_nn"] = float(c_nn)
            row["base_c_kg"] = float(c_kg)
        return rows

    def summary(self) -> Dict[str, object]:
        return {
            "best_epoch": None if self.history is None else self.history.best_epoch,
            "blends": {k: {"weights": v.weights, "val_c": v.validation_cindex, "calibration_slope": v.calibration_slope}
                       for k, v in self.blends.items()},
            "relation_gates": self.relation_gates() if self.model is not None and self.model.layers else {},
            "branch_scales": self.branch_scales() if self.model is not None else {},
            "pathway_coefficients": self.pathway_coefficients() if self.model is not None else {},
            "clinical_coefficients": self.clinical_coefficients() if self.model is not None else {},
            "flags": asdict(self.flags),
            "clinical_offset": self.clinical_offset,
            "clinical_initialization": getattr(self, "clinical_initialization", "heuristic"),
            "graph_learning_rate_ratio": getattr(self, "graph_learning_rate_ratio", None),
            "training_device":str(self.device),
            "gradient_mode":self.gradient_mode,
        }
