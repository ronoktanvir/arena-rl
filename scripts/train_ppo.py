"""CLI entry point for training PPO agents."""

from __future__ import annotations

import argparse

from agents.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_cnn.yaml",
        help="Path to a YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOTrainer()
    trainer.train(args.config)


if __name__ == "__main__":
    main()
