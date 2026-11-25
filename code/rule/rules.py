from __future__ import annotations

import torch
import torch.nn as nn


class HebbianUpdateRule(nn.Module):
    """
    No-backprop local rule:
    DeltaW ~ reward * outer(post_mean, pre_mean) with optional confidence scaling.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def aggregate_update(
        self,
        pre_mean: torch.Tensor,
        post_mean: torch.Tensor,
        reward: torch.Tensor,
        routing_active: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        bsz = pre_mean.shape[0]
        active = routing_active.clamp(0.0, 1.0)
        conf = confidence.clamp(0.0, 1.0)
        gain = (reward * active * conf).view(bsz, 1, 1)

        outer = post_mean.unsqueeze(2) @ pre_mean.unsqueeze(1)
        weighted = outer * gain

        denom = gain.abs().mean().clamp(min=self.eps)
        return weighted.mean(dim=0) / denom
