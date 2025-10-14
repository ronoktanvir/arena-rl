"""Minimal Proximal Policy Optimisation trainer used for CR-Lite."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
import yaml

from agents.nets import ConvLSTMPolicy, ConvPolicy
from envs.crlite_env import CRLiteEnv

__all__ = ["PPOTrainer"]


def _select_device() -> torch.device:
    """Choose the preferred training device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> Tensor:
    """Convert observations to ``(B, C, H, W)`` float tensors."""

    tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if tensor.ndim == 3:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 4:  # (B, H, W, C)
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported observation shape: {tensor.shape}")
    return tensor


def _mask_to_tensor(mask: np.ndarray, device: torch.device) -> Tensor:
    """Convert action masks to boolean tensors with a batch dimension."""

    tensor = torch.as_tensor(mask, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=torch.bool)


class PPOTrainer:
    """Lean PPO trainer with vectorised rollouts and optional LSTM support."""

    def __init__(self) -> None:
        self.device = _select_device()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> Dict[str, object]:
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a mapping of hyperparameters.")

        required_keys = {
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
            "opponent",
            "save_dir",
        }
        missing = required_keys - config.keys()
        if missing:
            raise KeyError(f"Missing PPO configuration keys: {sorted(missing)}")
        return config

    # ------------------------------------------------------------------
    # Environment factories
    # ------------------------------------------------------------------
    @staticmethod
    def _make_vector_env(num_envs: int, opponent: str, seed: Optional[int]) -> gym.vector.VectorEnv:
        def make_env(rank: int):
            def _init() -> CRLiteEnv:
                env_seed = None if seed is None else seed + rank
                return CRLiteEnv(opponent_policy=opponent, seed=env_seed, render_mode="rgb_array")

            return _init

        thunks = [make_env(i) for i in range(num_envs)]
        return gym.vector.SyncVectorEnv(thunks)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _collect_rollout(
        self,
        envs: gym.vector.VectorEnv,
        policy: torch.nn.Module,
        obs: np.ndarray,
        info: Dict[str, np.ndarray],
        *,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
        use_lstm: bool,
        lstm_hidden_size: Optional[int],
        episode_returns: np.ndarray,
        episode_lengths: np.ndarray,
        optimizer_device: torch.device,
        hx: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Dict[str, Tensor], np.ndarray, Dict[str, np.ndarray], Optional[Tuple[Tensor, Tensor]], List[float], List[int], List[bool]]:
        """Sample a rollout of ``num_steps`` transitions from the vectorised env."""

        num_envs = envs.num_envs
        obs_tensor = _obs_to_tensor(obs, optimizer_device)
        obs_shape = obs_tensor.shape[1:]
        action_dim = info["action_mask"].shape[-1]

        storage: Dict[str, Tensor] = {
            "obs": torch.zeros((num_steps, num_envs, *obs_shape), device=optimizer_device),
            "masks": torch.zeros((num_steps, num_envs, action_dim), device=optimizer_device, dtype=torch.bool),
            "actions": torch.zeros((num_steps, num_envs), device=optimizer_device, dtype=torch.long),
            "logprobs": torch.zeros((num_steps, num_envs), device=optimizer_device),
            "rewards": torch.zeros((num_steps, num_envs), device=optimizer_device),
            "dones": torch.zeros((num_steps, num_envs), device=optimizer_device),
            "values": torch.zeros((num_steps, num_envs), device=optimizer_device),
        }
        if use_lstm:
            if hx is None:
                if lstm_hidden_size is None:
                    raise ValueError("lstm_hidden_size must be provided when using an LSTM policy.")
                hx = (
                    torch.zeros(1, num_envs, lstm_hidden_size, device=optimizer_device),
                    torch.zeros(1, num_envs, lstm_hidden_size, device=optimizer_device),
                )
            storage["lstm_h"] = torch.zeros((num_steps, num_envs, hx[0].shape[-1]), device=optimizer_device)
            storage["lstm_c"] = torch.zeros((num_steps, num_envs, hx[1].shape[-1]), device=optimizer_device)

        finished_returns: List[float] = []
        finished_lengths: List[int] = []
        finished_wins: List[bool] = []

        for step in range(num_steps):
            obs_tensor = _obs_to_tensor(obs, optimizer_device)
            mask_tensor = _mask_to_tensor(info["action_mask"], optimizer_device)

            if use_lstm and hx is not None:
                storage["lstm_h"][step].copy_(hx[0].squeeze(0))
                storage["lstm_c"][step].copy_(hx[1].squeeze(0))

            storage["obs"][step].copy_(obs_tensor)
            storage["masks"][step].copy_(mask_tensor)

            logits, value, next_hx = policy(obs_tensor, action_mask=mask_tensor, hx=hx)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            storage["actions"][step] = action
            storage["logprobs"][step] = log_prob
            storage["values"][step] = value.squeeze(-1)

            next_obs, reward, terminated, truncated, next_info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            reward_tensor = torch.as_tensor(reward, device=optimizer_device, dtype=torch.float32)
            done_tensor = torch.as_tensor(done, device=optimizer_device, dtype=torch.float32)
            storage["rewards"][step] = reward_tensor
            storage["dones"][step] = done_tensor

            episode_returns += reward
            episode_lengths += 1

            final_info = next_info.get("final_info") if isinstance(next_info, dict) else None
            for env_index, finished in enumerate(done):
                if not finished:
                    continue
                finfo = None
                if final_info is not None:
                    finfo = final_info[env_index]
                win_flag = False
                if isinstance(finfo, dict):
                    win_flag = bool(finfo.get("win", False))
                elif isinstance(next_info, dict):
                    # Fall back to stacked info arrays
                    win_array = next_info.get("win")
                    if win_array is not None:
                        win_flag = bool(win_array[env_index])
                finished_returns.append(float(episode_returns[env_index]))
                finished_lengths.append(int(episode_lengths[env_index]))
                finished_wins.append(win_flag)
                episode_returns[env_index] = 0.0
                episode_lengths[env_index] = 0
                if use_lstm and hx is not None:
                    hx[0][:, env_index] = 0.0
                    hx[1][:, env_index] = 0.0

            hx = None if next_hx is None else (next_hx[0].detach(), next_hx[1].detach())

            obs = next_obs
            info = next_info

        # Bootstrap the final value
        last_obs_tensor = _obs_to_tensor(obs, optimizer_device)
        last_mask_tensor = _mask_to_tensor(info["action_mask"], optimizer_device)
        hx_for_value: Optional[Tuple[Tensor, Tensor]] = None
        if use_lstm and hx is not None:
            hx_for_value = (hx[0].clone(), hx[1].clone())
        with torch.no_grad():
            _, next_value, _ = policy(last_obs_tensor, action_mask=last_mask_tensor, hx=hx_for_value)
        storage["next_value"] = next_value.squeeze(-1)

        return storage, obs, info, hx, finished_returns, finished_lengths, finished_wins

    # ------------------------------------------------------------------
    # Advantage estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_gae(
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        num_steps, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(num_envs, device=rewards.device)
        next_val = next_value
        for step in reversed(range(num_steps)):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_val * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
            next_val = values[step]
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Optimisation step
    # ------------------------------------------------------------------
    def _ppo_update(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        storage: Dict[str, Tensor],
        *,
        num_epochs: int,
        minibatch_size: int,
        clip_eps: float,
        entropy_coef: float,
        value_coef: float,
        use_lstm: bool,
    ) -> Tuple[float, float, float]:
        policy.train()
        obs = storage["obs"].view(-1, *storage["obs"].shape[2:])
        masks = storage["masks"].view(-1, storage["masks"].shape[-1])
        actions = storage["actions"].view(-1)
        old_logprobs = storage["logprobs"].view(-1)
        advantages = storage["advantages"].view(-1)
        returns = storage["returns"].view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        if use_lstm:
            lstm_h = storage["lstm_h"].view(-1, storage["lstm_h"].shape[-1])
            lstm_c = storage["lstm_c"].view(-1, storage["lstm_c"].shape[-1])

        batch_size = obs.shape[0]
        indices = torch.randperm(batch_size, device=obs.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0

        for _ in range(num_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                obs_batch = obs[mb_idx]
                mask_batch = masks[mb_idx]
                action_batch = actions[mb_idx]
                old_logprob_batch = old_logprobs[mb_idx]
                advantage_batch = advantages[mb_idx]
                return_batch = returns[mb_idx]

                hx_batch: Optional[Tuple[Tensor, Tensor]] = None
                if use_lstm:
                    h_batch = lstm_h[mb_idx].unsqueeze(0)
                    c_batch = lstm_c[mb_idx].unsqueeze(0)
                    hx_batch = (h_batch, c_batch)

                logits, value, _ = policy(obs_batch, action_mask=mask_batch, hx=hx_batch)
                dist = Categorical(logits=logits)
                logprob = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()

                ratio = (logprob - old_logprob_batch).exp()
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value = value.squeeze(-1)
                value_loss = 0.5 * (return_batch - value).pow(2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                updates += 1

        if updates == 0:
            return 0.0, 0.0, 0.0
        return (
            total_policy_loss / updates,
            total_value_loss / updates,
            total_entropy / updates,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, config_path: str) -> None:
        config = self._load_config(config_path)
        seed = config.get("seed")
        if isinstance(seed, int):
            torch.manual_seed(seed)
            np.random.seed(seed)

        num_envs = int(config["num_envs"])
        save_dir = Path(str(config["save_dir"]))
        save_dir.mkdir(parents=True, exist_ok=True)

        envs = self._make_vector_env(num_envs, str(config["opponent"]), seed if isinstance(seed, int) else None)

        use_lstm = bool(config["use_lstm"])
        if use_lstm:
            policy = ConvLSTMPolicy().to(self.device)
            lstm_hidden_size = policy.lstm.hidden_size
        else:
            policy = ConvPolicy().to(self.device)
            lstm_hidden_size = None

        optimizer = torch.optim.Adam(policy.parameters(), lr=float(config["learning_rate"]))

        obs, info = envs.reset(seed=seed if isinstance(seed, int) else None)

        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        win_history: deque = deque(maxlen=100)
        best_win_rate = -float("inf")

        total_updates = int(config.get("total_updates", 1))

        hx: Optional[Tuple[Tensor, Tensor]] = None
        if use_lstm:
            hidden_size = lstm_hidden_size or 256
            hx = (
                torch.zeros(1, num_envs, hidden_size, device=self.device),
                torch.zeros(1, num_envs, hidden_size, device=self.device),
            )

        for update in range(total_updates):
            storage, obs, info, hx, finished_returns, finished_lengths, finished_wins = self._collect_rollout(
                envs,
                policy,
                obs,
                info,
                num_steps=int(config["num_steps"]),
                gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]),
                use_lstm=use_lstm,
                lstm_hidden_size=lstm_hidden_size,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
                optimizer_device=self.device,
                hx=hx,
            )

            win_history.extend(finished_wins)

            advantages, returns = self._compute_gae(
                storage["rewards"],
                storage["values"],
                storage["dones"],
                storage["next_value"],
                gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]),
            )
            storage["advantages"] = advantages
            storage["returns"] = returns

            policy_loss, value_loss, entropy = self._ppo_update(
                policy,
                optimizer,
                storage,
                num_epochs=int(config["update_epochs"]),
                minibatch_size=int(config["minibatch_size"]),
                clip_eps=float(config["clip_eps"]),
                entropy_coef=float(config["entropy_coef"]),
                value_coef=float(config["value_coef"]),
                use_lstm=use_lstm,
            )

            mean_return = float(np.mean(finished_returns)) if finished_returns else 0.0
            mean_length = float(np.mean(finished_lengths)) if finished_lengths else 0.0
            win_rate = (sum(finished_wins) / len(finished_wins)) if finished_wins else 0.0
            moving_win_rate = (sum(win_history) / len(win_history)) if win_history else 0.0

            print(
                f"Update {update + 1}/{total_updates} | "
                f"return: {mean_return:.2f} | length: {mean_length:.1f} | "
                f"win-rate: {win_rate:.2%} | moving win-rate: {moving_win_rate:.2%} | "
                f"policy loss: {policy_loss:.3f} | value loss: {value_loss:.3f} | entropy: {entropy:.3f}",
                flush=True,
            )

            checkpoint = {
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "update": update + 1,
                "moving_win_rate": moving_win_rate,
            }
            ckpt_path = save_dir / f"ckpt_{update + 1:04d}.pt"
            torch.save(checkpoint, ckpt_path)
            if moving_win_rate >= best_win_rate:
                best_win_rate = moving_win_rate
                torch.save(checkpoint, save_dir / "best.pt")

        envs.close()

    def evaluate(self, checkpoint_path: str, episodes: int = 10) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        use_lstm = bool(config.get("use_lstm", False))
        opponent = str(config.get("opponent", "greedy"))

        if use_lstm:
            policy = ConvLSTMPolicy().to(self.device)
        else:
            policy = ConvPolicy().to(self.device)
        policy.load_state_dict(checkpoint["model_state"])
        policy.eval()

        env = CRLiteEnv(opponent_policy=opponent, render_mode="rgb_array")

        returns: List[float] = []
        lengths: List[int] = []
        wins: List[bool] = []

        for _ in range(episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            hx: Optional[Tuple[Tensor, Tensor]] = None
            if use_lstm:
                hidden_size = policy.lstm.hidden_size
                hx = (
                    torch.zeros(1, 1, hidden_size, device=self.device),
                    torch.zeros(1, 1, hidden_size, device=self.device),
                )
            while not done:
                obs_tensor = _obs_to_tensor(obs, self.device)
                mask_tensor = _mask_to_tensor(info["action_mask"], self.device)
                logits, _, next_hx = policy(obs_tensor, action_mask=mask_tensor, hx=hx)
                action = torch.argmax(logits, dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

                if use_lstm and next_hx is not None:
                    hx = (next_hx[0].detach(), next_hx[1].detach())
                if done:
                    if use_lstm and hx is not None:
                        hx = (
                            torch.zeros_like(hx[0]),
                            torch.zeros_like(hx[1]),
                        )

            returns.append(episode_return)
            lengths.append(episode_length)
            wins.append(bool(info.get("win", False)))

        mean_return = sum(returns) / len(returns)
        mean_length = sum(lengths) / len(lengths)
        win_rate = sum(wins) / len(wins)
        print(
            f"Evaluation over {episodes} episodes | "
            f"return: {mean_return:.2f} | length: {mean_length:.1f} | win-rate: {win_rate:.2%}",
            flush=True,
        )
        env.close()
