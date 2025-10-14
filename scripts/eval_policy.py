"""Evaluate a trained PPO policy checkpoint."""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch

from agents.nets import ConvLSTMPolicy, ConvPolicy
from envs.crlite_env import CRLiteEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO policy checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run.",
    )
    return parser.parse_args()


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor / 255.0


def _mask_to_tensor(mask, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(mask, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=torch.bool)


def main() -> None:
    args = parse_args()
    device = _device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    use_lstm = bool(config.get("use_lstm", False))
    opponent = str(config.get("opponent", "greedy"))

    if use_lstm:
        policy: torch.nn.Module = ConvLSTMPolicy().to(device)
        hidden_size = policy.lstm.hidden_size  # type: ignore[attr-defined]
    else:
        policy = ConvPolicy().to(device)
        hidden_size = None
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()

    env = CRLiteEnv(opponent_policy=opponent, render_mode="rgb_array")

    returns: List[float] = []
    lengths: List[int] = []
    wins: List[bool] = []

    for _ in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_lstm:
            assert hidden_size is not None
            hx = (
                torch.zeros(1, 1, hidden_size, device=device),
                torch.zeros(1, 1, hidden_size, device=device),
            )

        while not done:
            obs_tensor = _obs_to_tensor(obs, device)
            mask_tensor = _mask_to_tensor(info["action_mask"], device)
            with torch.no_grad():
                logits, _, next_hx = policy(obs_tensor, action_mask=mask_tensor, hx=hx)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            done = bool(terminated or truncated)
            if use_lstm and next_hx is not None:
                hx = (next_hx[0].detach(), next_hx[1].detach())

        returns.append(episode_return)
        lengths.append(episode_length)
        wins.append(bool(info.get("win", False)))

    env.close()

    mean_return = sum(returns) / len(returns) if returns else 0.0
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    win_rate = sum(wins) / len(wins) if wins else 0.0

    print(
        f"Evaluation over {args.episodes} episodes | "
        f"avg return: {mean_return:.2f} | avg length: {mean_length:.1f} | win-rate: {win_rate:.2%}"
    )


if __name__ == "__main__":
    main()
