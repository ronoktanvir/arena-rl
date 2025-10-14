"""Interactive Streamlit viewer for CRLiteEnv checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import streamlit as st
import torch
from torch import Tensor
from torch.distributions import Categorical

from agents.nets import ConvLSTMPolicy, ConvPolicy
from envs.crlite_env import CRLiteEnv, Tower

CHECKPOINT_PATH = Path("checkpoints/best.pt")
OPPONENT_OPTIONS = {
    "Greedy": "greedy",
    "Mirror": "mirror",
    "Idle": "idle",
}


@dataclass
class AppState:
    """State container stored inside ``st.session_state``."""

    env: CRLiteEnv
    policy: torch.nn.Module
    use_lstm: bool
    obs: np.ndarray
    info: Dict[str, np.ndarray]
    total_reward: float = 0.0
    done: bool = False
    hx: Optional[Tuple[Tensor, Tensor]] = None
    outcome: Optional[str] = None


def _prepare_observation(obs: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float()
    tensor /= 255.0
    return tensor


def _prepare_mask(mask: np.ndarray) -> Tensor:
    return torch.from_numpy(mask).unsqueeze(0).to(dtype=torch.bool)


def _load_policy(checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    use_lstm = bool(config.get("use_lstm", False))
    if use_lstm:
        policy: torch.nn.Module = ConvLSTMPolicy()
    else:
        policy = ConvPolicy()
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    return policy, {"use_lstm": use_lstm, "config": config}


def _init_state(opponent: str) -> None:
    try:
        policy, meta = _load_policy(CHECKPOINT_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    env = CRLiteEnv(opponent_policy=opponent, render_mode="rgb_array")
    obs, info = env.reset()
    st.session_state.app_state = AppState(
        env=env,
        policy=policy,
        use_lstm=bool(meta["use_lstm"]),
        obs=obs,
        info=info,
    )


def _reset_env(opponent: str) -> None:
    if "app_state" not in st.session_state:
        _init_state(opponent)
        return

    state: AppState = st.session_state.app_state
    env = CRLiteEnv(opponent_policy=opponent, render_mode="rgb_array")
    obs, info = env.reset()
    st.session_state.app_state = AppState(
        env=env,
        policy=state.policy,
        use_lstm=state.use_lstm,
        obs=obs,
        info=info,
    )


def _step_env(sample: bool = False) -> None:
    state: AppState = st.session_state.app_state
    if state.done:
        return

    obs_tensor = _prepare_observation(state.obs)
    mask_tensor = _prepare_mask(state.info["action_mask"])

    with torch.no_grad():
        logits, _value, next_hx = state.policy(obs_tensor, mask_tensor, state.hx)
        if sample:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        else:
            action = int(torch.argmax(logits, dim=-1).item())

    obs, reward, terminated, truncated, info = state.env.step(action)
    total_reward = state.total_reward + reward
    done = terminated or truncated
    outcome: Optional[str] = state.outcome
    if done:
        if terminated:
            if not state.env.opponent_towers["king"].alive:
                outcome = "Win"
            elif not state.env.player_towers["king"].alive:
                outcome = "Loss"
            else:
                outcome = "Finished"
        else:
            outcome = "Timeout"

    if state.use_lstm and next_hx is not None:
        next_hx = tuple(t.detach() for t in next_hx)

    st.session_state.app_state = AppState(
        env=state.env,
        policy=state.policy,
        use_lstm=state.use_lstm,
        obs=obs,
        info=info,
        total_reward=total_reward,
        done=done,
        hx=next_hx if state.use_lstm else None,
        outcome=outcome,
    )


def _tower_summary(label: str, towers: Dict[str, Tower]) -> str:
    parts = []
    for key in ("lane_0", "lane_1", "king"):
        tower = towers.get(key)
        if tower is None:
            continue
        parts.append(f"{key}:{tower.hp:.0f}/{tower.max_hp:.0f}")
    return f"{label} towers → " + ", ".join(parts)


def _render_frame(frame: np.ndarray) -> np.ndarray:
    scale = 4
    return frame.repeat(scale, axis=0).repeat(scale, axis=1)


def main() -> None:
    st.set_page_config(page_title="Arena-RL Viewer", layout="centered")
    st.title("Arena-RL Streamlit Viewer")

    opponent_label = st.sidebar.selectbox("Opponent", list(OPPONENT_OPTIONS.keys()), index=0)
    opponent = OPPONENT_OPTIONS[opponent_label]

    if "app_state" not in st.session_state:
        _init_state(opponent)
    else:
        # If opponent selection changed, rebuild the environment.
        current_env: CRLiteEnv = st.session_state.app_state.env
        if current_env._opponent_policy != opponent:
            _reset_env(opponent)

    state: AppState = st.session_state.app_state

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Observation")
        st.image(_render_frame(state.obs), caption="Current frame", clamp=True)
        st.write(f"Total reward: {state.total_reward:.2f}")
        st.write(f"Episode done: {'Yes' if state.done else 'No'}")
        if state.outcome:
            st.success(f"Outcome: {state.outcome}")

    with col2:
        st.subheader("State")
        st.write(_tower_summary("Player", state.env.player_towers))
        st.write(_tower_summary("Opponent", state.env.opponent_towers))
        st.write(f"Player elixir: {state.env.player_elixir:.1f}")
        st.write(f"Opponent elixir: {state.env.opponent_elixir:.1f}")
        st.write(f"Step: {state.env.elapsed_steps}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Step", disabled=state.done):
            _step_env(sample=False)
            st.experimental_rerun()
    with col_b:
        if st.button("Play Episode", disabled=state.done):
            for _ in range(600):
                if st.session_state.app_state.done:
                    break
                _step_env(sample=False)
            st.experimental_rerun()
    with col_c:
        if st.button("Reset"):
            _reset_env(opponent)
            st.experimental_rerun()

    st.caption("Actions are selected greedily from the policy logits. Use the sidebar to change opponents.")


if __name__ == "__main__":
    main()
