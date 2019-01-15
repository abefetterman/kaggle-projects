import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class Hybrid(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 etad=1.0, etam=0.0, rho_adam=0.01,
                 weight_decay=0, amsgrad=False, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, etad=etad,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        nesterov=nesterov, etam=etam, rho_adam=rho_adam)
        super(Hybrid, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                # beta1_hat = beta1 #min(beta1,1.0-1.0/state['step'])
                beta2_hat = min(beta2,1.0-1.0/state['step'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2_hat).addcmul_(1 - beta2_hat, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                sgd_step_norm=torch.sum(exp_avg*exp_avg)
                adam_step_norm=torch.sum(torch.div(exp_avg*exp_avg,denom))

                step_mag = group['lr']*((group['etam'] * adam_step_norm) * group['rho_adam']\
                             + ((1-group['etam']) * sgd_step_norm))
                adam_step = step_mag*group['etad'] \
                            /(adam_step_norm + group['eps'])
                sgd_step = step_mag*(1-group['etad'])/(sgd_step_norm + group['eps'])

                wd = group['weight_decay']*group['lr']

                if nesterov:
                    adam_step *= beta1
                    sgd_step *= beta1

                p.data.add_(-wd, p.data)

                p.data.add_(-sgd_step, exp_avg)
                p.data.addcdiv_(-adam_step, exp_avg, denom)

                if nesterov:
                    p.data.add_(-sgd_step*(1-beta1)/beta1, grad)
                    p.data.addcdiv_(-adam_step*(1-beta1)/beta1, grad, denom)

        return loss
