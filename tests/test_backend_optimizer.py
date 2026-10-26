import torch
from optimizers import ElementwiseAdamW


def test_elementwise_adamw_matches_reference_update():
    torch.manual_seed(3)
    a=torch.nn.Parameter(torch.randn(12,4,dtype=torch.float64));b=torch.nn.Parameter(a.detach().clone())
    first=torch.optim.AdamW([a],lr=.002,weight_decay=.03,foreach=False)
    second=ElementwiseAdamW([b],lr=.002,weight_decay=.03)
    for _ in range(8):
        grad=torch.randn_like(a);a.grad=grad.clone();b.grad=grad.clone()
        first.step();second.step()
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
