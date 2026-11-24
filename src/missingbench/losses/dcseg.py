from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor | None = None,
    logvar2: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mu2 is None:
        mu2 = mu1.new_zeros(mu1.shape)
        logvar2 = torch.log(mu1.new_ones(mu1.shape))
        eps = 0.0

    if logvar2 is None:
        raise ValueError("logvar2 must be provided when mu2 is provided")
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    return 0.5 * torch.mean(-1 + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps))


class AnatomyContrastiveLoss(nn.Module):
    def __init__(self, method: str = "cos_sim") -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.method = method

    def get_logits_using_cos_sim(self, features: torch.Tensor) -> torch.Tensor:
        features = features.mean([3, 4, 5])
        features = features.view(-1, features.size(-1))
        features = F.normalize(features, p=2, dim=1)
        return torch.matmul(features, features.t())

    def get_logits_using_ssim(self, features: torch.Tensor) -> torch.Tensor:
        try:
            import torchmetrics
        except ImportError:
            return self.get_logits_using_cos_sim(features)

        features = features.view(-1, features.size(2), features.size(3), features.size(4), features.size(5))
        features_a = []
        features_b = []
        for i in range(features.size(0)):
            for j in range(i + 1, features.size(0)):
                features_a.append(features[i])
                features_b.append(features[j])
        if not features_a:
            return torch.eye(features.size(0), device=features.device, dtype=features.dtype)

        ssim_res = torchmetrics.functional.structural_similarity_index_measure(
            torch.stack(features_a, 0),
            torch.stack(features_b, 0),
            reduction="none",
        )
        logits = torch.zeros(features.size(0), features.size(0), device=features.device, dtype=ssim_res.dtype)
        count = 0
        for i in range(features.size(0)):
            for j in range(i + 1, features.size(0)):
                logits[i, j] = ssim_res[count]
                count += 1
        logits = logits + logits.t()
        logits.fill_diagonal_(1)
        return logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        if self.method == "cos_sim":
            logits = self.get_logits_using_cos_sim(features)
        elif self.method == "ssim":
            logits = self.get_logits_using_ssim(features)
        else:
            raise NotImplementedError(f"Unsupported anatomy contrastive method: {self.method}")

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logits = logits * self.logit_scale.exp()
        target = torch.zeros_like(logits)
        for index in range(batch_size):
            target[4 * index : 4 * (index + 1), 4 * index : 4 * (index + 1)] = 1
        return F.binary_cross_entropy_with_logits(logits, target)


class ModalityContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        features = features.permute(1, 0, 2).contiguous().view(-1, features.size(-1))
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.t()) * self.logit_scale.exp()

        target = torch.zeros_like(logits)
        for index in range(4):
            target[batch_size * index : batch_size * (index + 1), batch_size * index : batch_size * (index + 1)] = 1
        return F.binary_cross_entropy_with_logits(logits, target)


__all__ = [
    "AnatomyContrastiveLoss",
    "ModalityContrastiveLoss",
    "kl_divergence",
]
