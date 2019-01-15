# https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div(pred_q, pred_p):
    # pytorch KLDLoss is averaged over all dim if size_average=True
    klda = torch.sigmoid(pred_p) * (F.logsigmoid(pred_p) - F.logsigmoid(pred_q))
    kldb = torch.sigmoid(0 - pred_p) * (F.logsigmoid(0 - pred_p) - F.logsigmoid(0 - pred_q))
    return torch.mean(klda) + torch.mean(kldb)


class VATLoss(nn.Module):

    def __init__(self, xi=1e-3, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred, pred_empty, _ = model(x)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat, pred_empty_hat, _ = model(x + self.xi * d)
                adv_distance = _kl_div(pred_empty_hat, pred_empty)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat, pred_empty_hat, _ = model(x + r_adv)
            lds = _kl_div(pred_empty_hat, pred_empty)

        return lds
