"""Quick smoke test for the CRLiteEnv dynamics and observation space."""

from __future__ import annotations

import numpy as np

from envs.crlite_env import CRLiteEnv


def run_smoke_test(steps: int = 60) -> None:
    env = CRLiteEnv(opponent_policy="greedy", render_mode="rgb_array", seed=123)
    obs, info = env.reset()
    assert obs.shape == (96, 96, 3), f"Unexpected observation shape: {obs.shape}"
    assert obs.dtype == np.uint8
    for step in range(steps):
        mask = info["action_mask"]
        assert len(mask) == 9
        valid_actions = np.flatnonzero(mask)
        action = int(np.random.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (96, 96, 3)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":  # pragma: no cover
    run_smoke_test()
