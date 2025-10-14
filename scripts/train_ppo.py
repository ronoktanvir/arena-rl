"""Command-line entry point for training PPO agents."""

from __future__ import annotations

import argparse

from agents.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on CRLiteEnv")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_cnn.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Number of updates between logging summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOTrainer()
    trainer.train(args.config, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
