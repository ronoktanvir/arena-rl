"""Record a single rollout from a trained PPO policy to an MP4 file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from agents.nets import ConvLSTMPolicy, ConvPolicy
from envs.crlite_env import CRLiteEnv


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected observation with 3 dimensions, got {tensor.shape}")
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor / 255.0


def _mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(mask).to(device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=torch.bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a PPO policy rollout to an MP4 file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load.")
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/rollout.mp4",
        help="Output video file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    use_lstm = bool(config.get("use_lstm", False))

    if use_lstm:
        policy: torch.nn.Module = ConvLSTMPolicy()
        hidden_size = policy.lstm.hidden_size  # type: ignore[attr-defined]
    else:
        policy = ConvPolicy()
        hidden_size = None
    policy.load_state_dict(checkpoint["model_state"])
    policy = policy.to(device)
    policy.eval()

    env = CRLiteEnv(opponent_policy="greedy", render_mode="rgb_array")
    obs, info = env.reset()
    frames = []

    hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if use_lstm:
        assert hidden_size is not None
        hx = (
            torch.zeros(1, 1, hidden_size, device=device),
            torch.zeros(1, 1, hidden_size, device=device),
        )

    done = False
    while True:
        frames.append(np.array(obs, copy=True))
        obs_tensor = _obs_to_tensor(obs, device)
        mask_tensor = _mask_to_tensor(info["action_mask"], device)
        with torch.no_grad():
            logits, _, hx = policy(obs_tensor, mask_tensor, hx)
            action = int(torch.argmax(logits, dim=-1).item())
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if done:
            frames.append(np.array(obs, copy=True))
            break

    env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise RuntimeError("No frames captured during rollout.")

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (width, height),
    )

    for frame in frames:
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"Saved rollout to {out_path}")


if __name__ == "__main__":
    main()
