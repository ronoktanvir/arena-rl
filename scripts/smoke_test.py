"""Quick smoke test exercising the CRLiteEnv API."""

from __future__ import annotations

import numpy as np

from envs.crlite_env import CRLiteEnv


def main() -> None:
    env = CRLiteEnv(opponent_policy="greedy", seed=0, render_mode="rgb_array")
    obs, info = env.reset()
    assert obs.shape == (96, 96, 3), f"Unexpected observation shape: {obs.shape}"
    for _ in range(50):
        mask = info["action_mask"]
        assert mask.shape == (9,), f"Unexpected action mask shape: {mask.shape}"
        valid_actions = np.flatnonzero(mask)
        assert valid_actions.size > 0, "No valid actions available"
        action = int(np.random.choice(valid_actions))
        obs, _, terminated, truncated, info = env.step(action)
        assert obs.shape == (96, 96, 3), f"Unexpected observation shape: {obs.shape}"
        if terminated or truncated:
            obs, info = env.reset()
            assert obs.shape == (96, 96, 3)
    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
