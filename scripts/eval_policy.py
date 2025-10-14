"""Evaluate a trained PPO policy checkpoint."""

from __future__ import annotations

import argparse

from agents.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO policy checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOTrainer()
    trainer.evaluate(args.checkpoint, episodes=args.episodes)


if __name__ == "__main__":
    main()
