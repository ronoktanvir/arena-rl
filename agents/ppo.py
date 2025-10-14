"""Proximal Policy Optimisation trainer for Arena-RL policies.

This module provides a lean yet fully featured PPO implementation capable of
training either the feed-forward :class:`agents.nets.ConvPolicy` or the
recurrent :class:`agents.nets.ConvLSTMPolicy` on the ``CRLiteEnv`` environment.
It supports vectorised environment rollouts, Generalised Advantage Estimation,
and the clipped PPO objective together with entropy and value function losses.

The primary entry point is :class:`PPOTrainer`, which exposes two convenience
methods:

``train(config_path: str)``
    Load hyperparameters from a YAML configuration file and launch training.

``evaluate(ckpt_path: str, episodes: int)``
    Run evaluation episodes using a saved checkpoint.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.distributions import Categorical

from agents.nets import ConvLSTMPolicy, ConvPolicy
from envs.crlite_env import CRLiteEnv

__all__ = ["PPOTrainer"]


def _to_tensor(obs: np.ndarray, device: torch.device) -> Tensor:
    """Convert an RGB observation to a normalised ``torch.Tensor``."""

    tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    if tensor.ndim == 3:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 4:  # (B, H, W, C)
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected observation shape: {tensor.shape}")
    return tensor / 255.0


def _to_mask(mask: np.ndarray, device: torch.device) -> Tensor:
    """Convert a numpy action mask to a boolean tensor."""

    tensor = torch.from_numpy(mask).to(device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=torch.bool)


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class RolloutBatch:
    """Container keeping track of rollout tensors for PPO updates."""

    obs: Tensor
    action_masks: Tensor
    actions: Tensor
    logprobs: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    advantages: Optional[Tensor] = None
    returns: Optional[Tensor] = None
    h0: Optional[Tensor] = None
    c0: Optional[Tensor] = None

    @property
    def num_steps(self) -> int:
        return self.obs.shape[0]

    @property
    def num_envs(self) -> int:
        return self.obs.shape[1]

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return tuple(self.obs.shape[2:])

    @property
    def action_dim(self) -> int:
        return self.action_masks.shape[-1]

    def flatten(self) -> Tuple[Tensor, ...]:
        """Return feed-forward views where time and env dimensions are merged."""

        t, n = self.num_steps, self.num_envs
        view = lambda x: x.view(t * n, *x.shape[2:]) if x.ndim > 2 else x.view(t * n)
        return (
            view(self.obs),
            view(self.action_masks),
            view(self.actions),
            view(self.logprobs),
            view(self.advantages),
            view(self.returns),
            view(self.values),
        )


class PPOTrainer:
    """Utility class orchestrating PPO training and evaluation."""

    def __init__(self) -> None:
        self.device = _default_device()
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            raise ValueError("Configuration file must contain a mapping.")
        required = {
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_eps",
            "entropy_coef",
            "value_coef",
            "num_steps",
            "update_epochs",
            "minibatch_size",
            "num_envs",
            "use_lstm",
        }
        missing = sorted(required - cfg.keys())
        if missing:
            raise KeyError(f"Missing PPO config keys: {', '.join(missing)}")
        return cfg

    # ------------------------------------------------------------------
    # Environment factories
    # ------------------------------------------------------------------
    @staticmethod
    def _make_env(seed: Optional[int] = None) -> gym.Env:
        return CRLiteEnv(opponent_policy="greedy", seed=seed, render_mode="rgb_array")

    def _vector_env(self, num_envs: int, *, base_seed: Optional[int]) -> gym.vector.VectorEnv:
        def make_thunk(rank: int) -> callable:
            def _init() -> CRLiteEnv:
                seed = None if base_seed is None else base_seed + rank
                return self._make_env(seed)

            return _init

        thunks = [make_thunk(i) for i in range(num_envs)]
        return gym.vector.SyncVectorEnv(thunks)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _collect_rollouts(
        self,
        envs: gym.vector.VectorEnv,
        policy: torch.nn.Module,
        obs: np.ndarray,
        info: Dict[str, np.ndarray],
        running_returns: np.ndarray,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
        *,
        use_lstm: bool,
        lstm_hidden_size: Optional[int] = None,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[RolloutBatch, np.ndarray, Dict[str, np.ndarray], Optional[Tuple[Tensor, Tensor]], List[float], List[float]]:
        num_envs = envs.num_envs
        if use_lstm and hx is None:
            if lstm_hidden_size is None:
                raise ValueError("lstm_hidden_size must be provided when use_lstm is True")
            hx = (
                torch.zeros(1, num_envs, lstm_hidden_size, device=self.device),
                torch.zeros(1, num_envs, lstm_hidden_size, device=self.device),
            )

        obs_tensor = _to_tensor(obs, self.device)
        obs_shape = tuple(obs_tensor.shape[1:])
        batch = RolloutBatch(
            obs=torch.zeros(num_steps, num_envs, *obs_shape, dtype=torch.float32),
            action_masks=torch.zeros(num_steps, num_envs, 9, dtype=torch.bool),
            actions=torch.zeros(num_steps, num_envs, dtype=torch.long),
            logprobs=torch.zeros(num_steps, num_envs, dtype=torch.float32),
            rewards=torch.zeros(num_steps, num_envs, dtype=torch.float32),
            dones=torch.zeros(num_steps, num_envs, dtype=torch.float32),
            values=torch.zeros(num_steps, num_envs, dtype=torch.float32),
            h0=None,
            c0=None,
        )

        if use_lstm and hx is not None:
            batch.h0 = hx[0].detach().cpu()
            batch.c0 = hx[1].detach().cpu()

        episode_returns = running_returns.copy()
        completed_returns: List[float] = []
        completed_wins: List[float] = []

        for step in range(num_steps):
            obs_tensor = _to_tensor(obs, self.device)
            mask_tensor = _to_mask(info["action_mask"], self.device)
            with torch.no_grad():
                logits, value, next_hx = policy(obs_tensor, mask_tensor, hx)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprob = dist.log_prob(actions)

            batch.obs[step] = obs_tensor.cpu()
            batch.action_masks[step] = mask_tensor.cpu()
            batch.actions[step] = actions.cpu()
            batch.logprobs[step] = logprob.cpu()
            batch.values[step] = value.squeeze(-1).cpu()

            actions_np = actions.cpu().numpy()
            obs, reward, terminated, truncated, info = envs.step(actions_np)
            done = np.logical_or(terminated, truncated)

            batch.rewards[step] = torch.from_numpy(reward).to(dtype=torch.float32)
            batch.dones[step] = torch.from_numpy(done).to(dtype=torch.float32)

            episode_returns += reward
            win_array = np.zeros(num_envs, dtype=np.float32)
            if done.any():
                for idx in np.where(done)[0]:
                    env = envs.envs[idx]
                    win_flag = float(not env.opponent_towers["king"].alive)
                    completed_returns.append(float(episode_returns[idx]))
                    completed_wins.append(win_flag)
                    win_array[idx] = win_flag
                    episode_returns[idx] = 0.0
            info["win"] = win_array

            if use_lstm and next_hx is not None:
                hx = (next_hx[0].clone(), next_hx[1].clone())
                done_tensor = torch.from_numpy(done).to(self.device)
                if done_tensor.any():
                    mask = (1.0 - done_tensor.float()).view(1, -1, 1)
                    hx = (hx[0] * mask, hx[1] * mask)
            else:
                hx = None

        # Compute value of last observation for GAE.
        obs_tensor = _to_tensor(obs, self.device)
        mask_tensor = _to_mask(info["action_mask"], self.device)
        with torch.no_grad():
            last_value = policy(obs_tensor, mask_tensor, hx)[1].squeeze(-1)

        values = torch.cat([batch.values, last_value.cpu().unsqueeze(0)], dim=0)
        advantages = torch.zeros_like(batch.rewards)
        last_advantage = torch.zeros(num_envs, dtype=torch.float32)
        for t in reversed(range(num_steps)):
            next_non_terminal = 1.0 - batch.dones[t]
            delta = (
                batch.rewards[t]
                + gamma * values[t + 1] * next_non_terminal
                - values[t]
            )
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage
        returns = advantages + batch.values
        batch.advantages = advantages
        batch.returns = returns

        return batch, obs, info, hx, episode_returns, completed_returns, completed_wins

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def _feed_forward_minibatches(
        self,
        batch: RolloutBatch,
        minibatch_size: int,
        device: torch.device,
    ) -> Iterable[Tuple[Tensor, ...]]:
        obs, masks, actions, logprobs, advantages, returns, values = batch.flatten()
        batch_size = obs.shape[0]
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield (
                obs[idx].to(device),
                masks[idx].to(device),
                actions[idx].to(device),
                logprobs[idx].to(device),
                advantages[idx].to(device),
                returns[idx].to(device),
                values[idx].to(device),
            )

    def _recurrent_minibatches(
        self,
        batch: RolloutBatch,
        minibatch_size: int,
        device: torch.device,
    ) -> Iterable[Tuple[Tensor, ...]]:
        num_envs = batch.num_envs
        indices = torch.randperm(num_envs)
        for start in range(0, num_envs, minibatch_size):
            env_idx = indices[start : start + minibatch_size]
            obs = batch.obs[:, env_idx].to(device)
            masks = batch.action_masks[:, env_idx].to(device)
            actions = batch.actions[:, env_idx].to(device)
            logprobs = batch.logprobs[:, env_idx].to(device)
            advantages = batch.advantages[:, env_idx].to(device)
            returns = batch.returns[:, env_idx].to(device)
            values = batch.values[:, env_idx].to(device)
            dones = batch.dones[:, env_idx].to(device)
            if batch.h0 is None or batch.c0 is None:
                raise RuntimeError("Missing initial hidden state for recurrent batch.")
            h0 = batch.h0[:, env_idx].to(device)
            c0 = batch.c0[:, env_idx].to(device)
            yield obs, masks, actions, logprobs, advantages, returns, values, dones, (h0, c0)

    @staticmethod
    def _recurrent_forward(
        policy: ConvLSTMPolicy,
        obs: Tensor,
        masks: Tensor,
        dones: Tensor,
        hidden: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        outputs: List[Tensor] = []
        values: List[Tensor] = []
        hx = (hidden[0].clone(), hidden[1].clone())
        timesteps = obs.shape[0]
        for t in range(timesteps):
            logits, value, hx = policy(obs[t], masks[t], hx)
            outputs.append(logits)
            values.append(value)
            done_mask = (1.0 - dones[t]).view(1, -1, 1)
            hx = (hx[0] * done_mask, hx[1] * done_mask)
        return torch.stack(outputs), torch.stack(values)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, config_path: str, *, log_interval: int = 10) -> None:
        config = self._load_config(config_path)
        torch.manual_seed(config.get("seed", 0))
        np.random.seed(config.get("seed", 0))

        num_envs = int(config["num_envs"])
        envs = self._vector_env(num_envs, base_seed=config.get("seed"))
        obs, info = envs.reset(seed=config.get("seed"))

        use_lstm = bool(config["use_lstm"])
        if use_lstm:
            policy: torch.nn.Module = ConvLSTMPolicy()
            lstm_hidden_size = policy.lstm.hidden_size
        else:
            policy = ConvPolicy()
            lstm_hidden_size = None
        policy = policy.to(self.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(config["learning_rate"]))

        total_timesteps = config.get("total_timesteps")
        if total_timesteps is not None:
            timesteps_per_rollout = int(config["num_steps"]) * num_envs
            total_updates = max(1, total_timesteps // timesteps_per_rollout)
        else:
            total_updates = int(config.get("num_updates", 1000))

        best_win_rate = -math.inf
        returns_window: deque[float] = deque(maxlen=100)
        wins_window: deque[float] = deque(maxlen=100)

        hx: Optional[Tuple[Tensor, Tensor]]
        if use_lstm:
            hx = (
                torch.zeros(1, num_envs, lstm_hidden_size, device=self.device),
                torch.zeros(1, num_envs, lstm_hidden_size, device=self.device),
            )
        else:
            hx = None
        running_returns = np.zeros(num_envs, dtype=np.float32)
        for update in range(1, total_updates + 1):
            batch, obs, info, hx, running_returns, completed_returns, completed_wins = self._collect_rollouts(
                envs,
                policy,
                obs,
                info,
                running_returns,
                num_steps=int(config["num_steps"]),
                gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]),
                use_lstm=use_lstm,
                lstm_hidden_size=lstm_hidden_size,
                hx=hx,
            )

            advantages = batch.advantages
            if advantages is None:
                raise RuntimeError("Advantages were not computed during rollout collection.")
            adv_flat = advantages.view(-1)
            adv_mean = adv_flat.mean()
            adv_std = adv_flat.std(unbiased=False)
            batch.advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            for _ in range(int(config["update_epochs"])):
                if use_lstm:
                    for mb in self._recurrent_minibatches(batch, int(config["minibatch_size"]), self.device):
                        obs_mb, masks_mb, actions_mb, old_logprob_mb, adv_mb, ret_mb, _, dones_mb, hidden_mb = mb
                        logits_seq, values_seq = self._recurrent_forward(policy, obs_mb, masks_mb, dones_mb, hidden_mb)
                        logits = logits_seq.view(-1, logits_seq.shape[-1])
                        values = values_seq.view(-1)
                        dist = Categorical(logits=logits)
                        logprob = dist.log_prob(actions_mb.view(-1))
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(logprob - old_logprob_mb.view(-1))
                        surr1 = ratio * adv_mb.view(-1)
                        surr2 = torch.clamp(
                            ratio,
                            1.0 - float(config["clip_eps"]),
                            1.0 + float(config["clip_eps"]),
                        ) * adv_mb.view(-1)
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(values, ret_mb.view(-1))
                        loss = (
                            policy_loss
                            + float(config["value_coef"]) * value_loss
                            - float(config["entropy_coef"]) * entropy
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                        optimizer.step()
                else:
                    for mb in self._feed_forward_minibatches(batch, int(config["minibatch_size"]), self.device):
                        obs_mb, masks_mb, actions_mb, old_logprob_mb, adv_mb, ret_mb, _ = mb
                        logits, value, _ = policy(obs_mb, masks_mb)
                        dist = Categorical(logits=logits)
                        logprob = dist.log_prob(actions_mb)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(logprob - old_logprob_mb)
                        surr1 = ratio * adv_mb
                        surr2 = torch.clamp(
                            ratio,
                            1.0 - float(config["clip_eps"]),
                            1.0 + float(config["clip_eps"]),
                        ) * adv_mb
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(value.squeeze(-1), ret_mb)
                        loss = (
                            policy_loss
                            + float(config["value_coef"]) * value_loss
                            - float(config["entropy_coef"]) * entropy
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                        optimizer.step()

            # Update logging buffers based on completed episodes in this rollout.
            returns_window.extend(completed_returns)
            wins_window.extend(completed_wins)

            mean_return = float(np.mean(list(returns_window))) if returns_window else 0.0
            mean_win = float(np.mean(list(wins_window))) if wins_window else 0.0

            if update % log_interval == 0:
                print(
                    f"Update {update}/{total_updates}: mean_return={mean_return:.2f} "
                    f"win_rate={mean_win:.2f}"
                )

            checkpoint = {
                "model_state": policy.state_dict(),
                "config": config,
                "update": update,
            }
            torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
            if mean_win > best_win_rate:
                best_win_rate = mean_win
                torch.save(checkpoint, self.checkpoint_dir / "best.pt")

        envs.close()

    def evaluate(self, ckpt_path: str, episodes: int = 5) -> None:
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        config = checkpoint.get("config", {})
        use_lstm = bool(config.get("use_lstm", False))
        if use_lstm:
            policy: torch.nn.Module = ConvLSTMPolicy()
        else:
            policy = ConvPolicy()
        policy.load_state_dict(checkpoint["model_state"])
        policy = policy.to(self.device)
        policy.eval()

        env = CRLiteEnv(opponent_policy="greedy", render_mode="rgb_array")
        returns: List[float] = []
        wins = 0
        for episode in range(episodes):
            obs, info = env.reset()
            hx: Optional[Tuple[Tensor, Tensor]] = None
            if use_lstm:
                hidden_size = policy.lstm.hidden_size  # type: ignore[attr-defined]
                hx = (
                    torch.zeros(1, 1, hidden_size, device=self.device),
                    torch.zeros(1, 1, hidden_size, device=self.device),
                )
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = _to_tensor(obs, self.device)
                mask_tensor = _to_mask(info["action_mask"], self.device)
                with torch.no_grad():
                    logits, _, hx = policy(obs_tensor, mask_tensor, hx)
                    action = torch.argmax(logits, dim=-1)
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = bool(terminated or truncated)
                total_reward += float(reward)
                if done and terminated:
                    wins += int(not env.opponent_towers["king"].alive)
                if done and use_lstm:
                    hx = (
                        torch.zeros_like(hx[0]),
                        torch.zeros_like(hx[1]),
                    )
            returns.append(total_reward)

        mean_return = float(np.mean(returns)) if returns else 0.0
        win_rate = wins / max(1, episodes)
        print(
            f"Evaluation over {episodes} episodes: mean_return={mean_return:.2f}, "
            f"win_rate={win_rate:.2f}"
        )
        env.close()
