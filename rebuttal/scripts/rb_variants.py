"""Pipeline variants for the attribution experiments (PREREG A6). New code only; shared modules untouched.

nosev   : constraint loss without the Severinghaus (SpO2-PaO2) term; everything else identical to PCL
tsmask  : pretraining masks whole timesteps (30% of timesteps, all observed variables at that hour)
vars9   : the 9 core variables only (first 9 columns of the 17-variable arrays; identical to the
          9-variable pipeline because normalization is a fixed per-variable affine map and
          forward fill and masks are per variable)
orig_hp : mask rate 0.5 and warmup fraction 0.3 (the original values), 17 variables
"""
import copy

import numpy as np
import torch

from src.losses.pcl_loss import PhysiologicalConstraintLoss


class NoSeveringhausLoss(PhysiologicalConstraintLoss):
    """Same loss as the main arm with the SpO2-PaO2 term removed from L_PCL."""

    def forward(self, x, c_mask, pretrain_mask=None, hh_weight=1.0):
        out = super().forward(x, c_mask, pretrain_mask=pretrain_mask, hh_weight=hh_weight)
        ls = out["losses"]
        L = ls["MAP"] + self.pp_weight * ls["PP"] + self.si_weight * ls["SI"] + ls["HH"]
        if torch.isnan(L):
            L = torch.tensor(0.0, device=x.device, requires_grad=True)
        out["L_PCL"] = L
        return out


def timestep_mask_fn(mask_prob):
    def apply(x, obs_mask, mask_prob=mask_prob):
        rand = torch.rand(x.shape[0], x.shape[1], 1, device=x.device)
        pretrain_mask = (rand < mask_prob) & obs_mask
        xm = x.clone()
        xm[pretrain_mask] = 0.0
        return xm, pretrain_mask
    return apply


def core_only(samples):
    from src.data.variables import CORE_VARIABLES
    k = len(CORE_VARIABLES)
    out = []
    for s in samples:
        t = copy.copy(s)
        t["values"] = np.asarray(s["values"])[:, :k]
        t["mask"] = np.asarray(s["mask"])[:, :k]
        out.append(t)
    return out


VARIANTS = {
    "nosev":   {"loss": "nosev", "mask": "random", "mask_prob": 0.30, "warmup": 0.4, "n_vars": 17, "arms": [0.5]},
    "tsmask":  {"loss": "main", "mask": "timestep", "mask_prob": 0.30, "warmup": 0.4, "n_vars": 17, "arms": [0.0, 0.5]},
    "vars9":   {"loss": "main", "mask": "random", "mask_prob": 0.30, "warmup": 0.4, "n_vars": 9, "arms": [0.0, 0.5]},
    "orig_hp": {"loss": "main", "mask": "random", "mask_prob": 0.50, "warmup": 0.3, "n_vars": 17, "arms": [0.0, 0.5]},
}
