import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()

    def forward(self, logits_ps, logits_sm, logits_mp):
        """
        Compute CLIP-style contrastive loss across different modalities.

        Args:
            logits_ps (torch.Tensor): Logits for Photometry-Spectra (batch_size, batch_size).
            logits_sm (torch.Tensor): Logits for Spectra-Metadata (batch_size, batch_size).
            logits_mp (torch.Tensor): Logits for Metadata-Photometry (batch_size, batch_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Losses for each modality pair.
        """
        labels = torch.arange(logits_ps.shape[0], dtype=torch.int64, device=logits_ps.device)

        loss_ps = F.cross_entropy(logits_ps, labels) + F.cross_entropy(logits_ps.transpose(-1, -2), labels)
        loss_sm = F.cross_entropy(logits_sm, labels) + F.cross_entropy(logits_sm.transpose(-1, -2), labels)
        loss_mp = F.cross_entropy(logits_mp, labels) + F.cross_entropy(logits_mp.transpose(-1, -2), labels)

        return loss_ps, loss_sm, loss_mp
