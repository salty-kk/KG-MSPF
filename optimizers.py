"""AdamW without DirectML's unsupported lerp/foreach CPU fallback."""
import math
import torch


class ElementwiseAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,weight_decay=1e-2,betas=(.9,.999),eps=1e-8):
        super().__init__(params,dict(lr=lr,weight_decay=weight_decay,betas=betas,eps=eps))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            b1,b2=group['betas'];lr=group['lr']
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad
                if g.is_sparse:raise RuntimeError('Sparse gradients are unsupported')
                state=self.state[p]
                if not state:
                    state.update(step=0,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                state['step']+=1
                m,v=state['exp_avg'],state['exp_avg_sq']
                p.mul_(1-lr*group['weight_decay'])
                m.mul_(b1).add_(g,alpha=1-b1)
                v.mul_(b2).addcmul_(g,g,value=1-b2)
                denom=v.sqrt().div_(math.sqrt(1-b2**state['step'])).add_(group['eps'])
                p.addcdiv_(m,denom,value=-lr/(1-b1**state['step']))
        return loss


def backend_adamw(params,**kwargs):
    params=list(params)
    tensors=[p for group in params for p in group['params']] if params and isinstance(params[0],dict) else params
    return (ElementwiseAdamW(params,**kwargs) if any(p.device.type=='privateuseone' for p in tensors)
            else torch.optim.AdamW(params,**kwargs))
