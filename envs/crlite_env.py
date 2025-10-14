from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CRLiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, opponent: str = "greedy", seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(9)  # 0=no-op; 1..8 = play card in lane
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.max_steps = 600
        self.player_hp = 1000
        self.enemy_hp = 1000
        self.elixir = 5.0

    def _frame(self) -> np.ndarray:
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[70:86, 42:54, :] = (0, 200, 0)   # player tower
        img[10:26, 42:54, :] = (200, 0, 0)   # enemy tower
        e = int(np.clip(self.elixir / 10.0 * 96, 0, 96))
        img[92:94, :e, :] = (120, 120, 255)  # elixir bar
        return img

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.elixir >= 2.0:
            mask[:] = True
        else:
            mask[0] = True
        return mask

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.player_hp = 1000
        self.enemy_hp = 1000
        self.elixir = 5.0
        obs = self._frame()
        info = {"action_mask": self._action_mask(), "win": False, "result": None}
        return obs, info

    def step(self, action: int):
        self.t += 1
        reward = 0.0
        if not self._action_mask()[action]:
            action = 0
            reward -= 0.01
        if action == 0:
            reward -= 0.01
        else:
            self.elixir -= 2.0
            if self.rng.random() < 0.6:
                dmg = int(self.rng.integers(10, 30))
                self.enemy_hp = max(0, self.enemy_hp - dmg)
                reward += dmg / 100.0
            else:
                dmg = int(self.rng.integers(10, 30))
                self.player_hp = max(0, self.player_hp - dmg)
                reward -= dmg / 100.0
        self.elixir = min(10.0, self.elixir + 0.05)

        terminated = False
        truncated = False
        result = None
        if self.enemy_hp <= 0:
            reward += 5.0
            terminated, result = True, "win"
        elif self.player_hp <= 0:
            reward -= 5.0
            terminated, result = True, "loss"
        if self.t >= self.max_steps:
            truncated = True

        obs = self._frame()
        info = {"action_mask": self._action_mask(), "win": result == "win", "result": result}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._frame()
