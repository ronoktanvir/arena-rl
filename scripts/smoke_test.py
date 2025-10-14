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
import numpy as np
from envs.crlite_env import CRLiteEnv

env = CRLiteEnv(seed=123)
obs, info = env.reset()
assert obs.shape == (96, 96, 3), obs.shape
mask = info.get("action_mask", np.ones(9, dtype=bool))
total = 0.0
for step in range(50):
    valid = np.flatnonzero(mask)
    a = int(np.random.choice(valid)) if len(valid) else 0
    obs, r, term, trunc, info = env.step(a)
    total += r
    mask = info.get("action_mask", np.ones(9, dtype=bool))
    if term or trunc:
        break
print(f"OK: steps={step+1}, return={total:.2f}, result={info.get('result')}")
