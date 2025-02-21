import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()

    def forward(self, logits_ps, logits_sm, logits_mp):
        labels = torch.arange(logits_ps.shape[0], dtype=torch.int64, device=logits_ps.device)

        loss_ps = F.cross_entropy(logits_ps, labels) + F.cross_entropy(logits_ps.transpose(-1, -2), labels)
        loss_sm = F.cross_entropy(logits_sm, labels) + F.cross_entropy(logits_sm.transpose(-1, -2), labels)
        loss_mp = F.cross_entropy(logits_mp, labels) + F.cross_entropy(logits_mp.transpose(-1, -2), labels)

        return loss_ps, loss_sm, loss_mp
