"""Neural network policy definitions used by Arena-RL agents.

This module provides two policy architectures tailored for image-based
observations coming from :class:`envs.crlite_env.CRLiteEnv`.  Both policies
consume RGB frames of shape ``(B, C, H, W)`` (with ``C=3``) and produce a pair
of tensors: categorical action logits for the discrete action space with nine
choices and a scalar state-value estimate.

The :class:`ConvPolicy` is a feed-forward convolutional policy suitable for
single-frame decision making.  The :class:`ConvLSTMPolicy` extends that encoder
with an ``LSTM`` core so that downstream algorithms can keep temporal state
when they require recurrent processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["ConvPolicy", "ConvLSTMPolicy", "MaskedCategoricalOutput"]


NEG_INF = -1e9


def _prepare_observation(obs: Tensor) -> Tensor:
    """Ensure the observation tensor follows the ``(B, C, H, W)`` convention."""

    if obs.dim() == 3:  # (C, H, W) or (H, W, C)
        obs = obs.unsqueeze(0)
    if obs.shape[1] != 3 and obs.shape[-1] == 3:
        obs = obs.permute(0, 3, 1, 2)
    if obs.dtype != torch.float32:
        obs = obs.float()
    return obs


def _prepare_action_mask(mask: Optional[Tensor], batch_size: int) -> Optional[Tensor]:
    """Normalise the action mask to a float tensor for masking logits."""

    if mask is None:
        return None
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != batch_size:
        mask = mask.expand(batch_size, -1)
    return mask.to(dtype=torch.bool)


@dataclass
class MaskedCategoricalOutput:
    """Container for the outputs of a masked policy network."""

    logits: Tensor
    value: Tensor
    next_state: Optional[Tuple[Tensor, Tensor]] = None


class ConvPolicy(nn.Module):
    """Feed-forward convolutional policy producing action logits and values."""

    def __init__(self, input_channels: int = 3, hidden_size: int = 512) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 12 * 12, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden_size, 9)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        obs: Tensor,
        action_mask: Optional[Tensor] = None,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor, Tensor]]]:
        del hx  # Not used but kept for API parity with recurrent policies.

        obs = _prepare_observation(obs)
        features = self.encoder(obs)
        hidden = self.head(features)

        logits = self.policy_head(hidden)
        mask = _prepare_action_mask(action_mask, logits.shape[0])
        if mask is not None:
            invalid = ~mask
            logits = logits.masked_fill(invalid, NEG_INF)

        value = self.value_head(hidden)
        return logits, value, None

    def to(self, device: torch.device | str) -> "ConvPolicy":
        super().to(device)
        return self


class ConvLSTMPolicy(nn.Module):
    """Recurrent convolutional policy with an LSTM core."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_size: int = 512,
        lstm_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64 * 12 * 12, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.policy_head = nn.Linear(lstm_hidden_size, 9)
        self.value_head = nn.Linear(lstm_hidden_size, 1)

    def forward(
        self,
        obs: Tensor,
        action_mask: Optional[Tensor] = None,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        obs = _prepare_observation(obs)
        batch_size = obs.shape[0]
        # Recurrent policies typically expect (B, T, C, H, W). We allow callers to
        # pass either ``(B, C, H, W)`` or ``(B, T, C, H, W)`` and reshape as needed.
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        elif obs.dim() != 5:
            raise ValueError("Expected observation with shape (B,T,C,H,W) or (B,C,H,W)")

        b, t, c, h, w = obs.shape
        obs = obs.reshape(b * t, c, h, w)
        features = self.encoder(obs)
        projected = self.projection(features)
        projected = projected.view(b, t, -1)

        if hx is None:
            device = projected.device
            hx = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
            )

        lstm_out, next_hx = self.lstm(projected, hx)
        # For policy/value heads we only use the last timestep output.
        last_out = lstm_out[:, -1]
        logits = self.policy_head(last_out)
        mask = _prepare_action_mask(action_mask, logits.shape[0])
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG_INF)

        value = self.value_head(last_out)
        return logits, value, next_hx

    def to(self, device: torch.device | str) -> "ConvLSTMPolicy":
        super().to(device)
        return self
