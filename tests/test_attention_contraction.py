import copy
import numpy as np
import torch
from torch.nn import functional as F
from model_heterogat import RelationalGATLayer


def test_destination_contraction_preserves_values_and_gradients():
    torch.manual_seed(7)
    layer=RelationalGATLayer(16,4,('r',),dropout=0)
    x=torch.randn(5,11,16,requires_grad=True)
    edges=torch.tensor([[0,1,4,5,7,8],[3,3,3,9,9,9]])
    dest=x[:,edges[1],:]
    projected=layer.destination_projection['r'](dest).view(5,6,4,4).permute(0,2,1,3)
    reference=(projected*layer.attention_destination['r'].view(1,4,1,4)).sum(-1)
    kernel=(layer.destination_projection['r'].weight.view(4,4,16)
            * layer.attention_destination['r'].unsqueeze(-1)).sum(1)
    fast=F.linear(dest,kernel).permute(0,2,1)
    torch.testing.assert_close(fast,reference,atol=1e-6,rtol=1e-5)
    params=[x,layer.destination_projection['r'].weight,layer.attention_destination['r']]
    ga=torch.autograd.grad(reference.square().sum(),params,retain_graph=True)
    gb=torch.autograd.grad(fast.square().sum(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=3e-5,rtol=1e-5)
