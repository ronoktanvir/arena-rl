"""Record a single CRLiteEnv rollout to an MP4 file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from agents.nets import ConvLSTMPolicy, ConvPolicy
from envs.crlite_env import CRLiteEnv

FRAME_SIZE = (96, 96)
FPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a PPO policy rollout")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/rollout.mp4",
        help="Output MP4 path (directories will be created).",
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
    obs, info = env.reset()

    frames: list[np.ndarray] = []
    hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if use_lstm:
        assert hidden_size is not None
        hx = (
            torch.zeros(1, 1, hidden_size, device=device),
            torch.zeros(1, 1, hidden_size, device=device),
        )

    done = False
    while not done:
        frames.append(np.array(obs, copy=True))
        obs_tensor = _obs_to_tensor(obs, device)
        mask_tensor = _mask_to_tensor(info["action_mask"], device)
        with torch.no_grad():
            logits, _, next_hx = policy(obs_tensor, action_mask=mask_tensor, hx=hx)
        action = int(torch.argmax(logits, dim=-1).item())
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if use_lstm and next_hx is not None:
            hx = (next_hx[0].detach(), next_hx[1].detach())

    env.close()

    if not frames:
        raise RuntimeError("No frames collected from the environment.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        FRAME_SIZE,
    )

    width, height = FRAME_SIZE
    for frame in frames:
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.shape[1] != width or rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"Saved rollout to {out_path}")


if __name__ == "__main__":
    main()
