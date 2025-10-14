"""Policy network implementations used by Arena-RL agents."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["ConvPolicy", "ConvLSTMPolicy"]

NEG_INF = -1e9


def _format_observation(obs: Tensor) -> Tensor:
    """Convert observations to float tensors shaped as ``(B, C, H, W)``."""

    if obs.dim() == 3:  # (H, W, C) or (C, H, W)
        obs = obs.unsqueeze(0)
    if obs.shape[1] != 3 and obs.shape[-1] == 3:
        obs = obs.permute(0, 3, 1, 2)
    obs = obs.float() / 255.0 if obs.dtype != torch.float32 else obs
    if obs.max().item() > 1.0:
        obs = obs / 255.0
    return obs


def _format_action_mask(mask: Optional[Tensor], batch_size: int) -> Optional[Tensor]:
    """Ensure the action mask is boolean with a batch dimension."""

    if mask is None:
        return None
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != batch_size:
        mask = mask.expand(batch_size, -1)
    return mask.to(dtype=torch.bool)


class ConvPolicy(nn.Module):
    """Feed-forward convolutional policy producing logits and state-values."""

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
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 96, 96)
            conv_out = self.encoder(dummy)
            self._flat_size = conv_out.shape[-1]
        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, hidden_size),
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
        del hx  # feed-forward policy ignores hidden state

        obs = _format_observation(obs)
        features = self.encoder(obs)
        hidden = self.fc(features)

        logits = self.policy_head(hidden)
        mask = _format_action_mask(action_mask, logits.shape[0])
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG_INF)

        value = self.value_head(hidden)
        return logits, value, None

    def to(self, device: torch.device | str) -> "ConvPolicy":  # type: ignore[override]
        super().to(device)
        return self


class ConvLSTMPolicy(nn.Module):
    """Convolutional encoder with an LSTM core for temporal credit assignment."""

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
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 96, 96)
            conv_out = self.encoder(dummy)
            self._flat_size = conv_out.shape[-1]
        self.proj = nn.Sequential(
            nn.Linear(self._flat_size, hidden_size),
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
        obs = _format_observation(obs)
        batch_size = obs.shape[0]
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        elif obs.dim() != 5:
            raise ValueError("Expected observation with shape (B,C,H,W) or (B,T,C,H,W)")

        b, t, c, h, w = obs.shape
        obs = obs.view(b * t, c, h, w)
        features = self.encoder(obs)
        projected = self.proj(features).view(b, t, -1)

        if hx is None:
            device = projected.device
            hx = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
            )

        lstm_out, next_hx = self.lstm(projected, hx)
        last_out = lstm_out[:, -1]
        logits = self.policy_head(last_out)
        mask = _format_action_mask(action_mask, logits.shape[0])
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG_INF)

        value = self.value_head(last_out)
        return logits, value, next_hx

    def to(self, device: torch.device | str) -> "ConvLSTMPolicy":  # type: ignore[override]
        super().to(device)
        return self
