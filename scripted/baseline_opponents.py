"""Scripted baseline opponents for CRLiteEnv."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

Action = Optional[Tuple[int, int]]
OpponentFn = Callable[[Dict[str, object]], Action]


def idle_opponent(_: Dict[str, object]) -> Action:
    """A passive opponent that never plays a card."""

    return None


def greedy_spammer() -> OpponentFn:
    """Create a greedy opponent that spams the cheapest affordable card."""

    state_cache = {"last_lane": "R"}

    def _policy(state: Dict[str, object]) -> Action:
        step = int(state.get("step", 0))
        affordable: List[int] = list(state.get("affordable_idxs", []))
        if not affordable or step % 20 != 0:
            return None

        hand_costs: List[int] = list(state.get("hand_costs", []))
        cheapest_idx = min(affordable, key=lambda idx: hand_costs[idx] if idx < len(hand_costs) else 99)

        lane_load: Dict[str, int] = dict(state.get("lane_load", {"L": 0, "R": 0}))
        preferred_lane = min(lane_load.items(), key=lambda item: (item[1], item[0]))[0]
        if lane_load.get("L", 0) == lane_load.get("R", 0):
            preferred_lane = "R" if state_cache["last_lane"] == "L" else "L"
        state_cache["last_lane"] = preferred_lane
        lane_int = 0 if preferred_lane.upper().startswith("L") else 1
        return cheapest_idx, lane_int

    return _policy


def mirror_opponent() -> OpponentFn:
    """Create an opponent that mirrors the player's last lane when possible."""

    def _policy(state: Dict[str, object]) -> Action:
        affordable: List[int] = list(state.get("affordable_idxs", []))
        if not affordable:
            return None
        rng = state.get("rng")
        last_lane = state.get("last_player_lane")
        if last_lane is None:
            if hasattr(rng, "integers"):
                lane_int = int(rng.integers(0, 2))
            elif hasattr(rng, "choice"):
                lane_int = int(rng.choice([0, 1]))
            else:
                lane_int = 0
        else:
            lane_int = 0 if str(last_lane).upper().startswith("L") else 1
        if hasattr(rng, "choice"):
            hand_idx = int(rng.choice(affordable))
        elif hasattr(rng, "integers"):
            hand_idx = int(affordable[int(rng.integers(0, len(affordable)))])
        else:
            hand_idx = affordable[0]
        return hand_idx, lane_int

    return _policy


def get_opponent(name: str) -> OpponentFn:
    """Return a scripted opponent callable by name."""

    key = name.lower()
    if key == "idle":
        return idle_opponent
    if key == "greedy":
        return greedy_spammer()
    if key == "mirror":
        return mirror_opponent()
    raise ValueError(f"Unknown opponent policy: {name}")


__all__ = [
    "get_opponent",
    "idle_opponent",
    "greedy_spammer",
    "mirror_opponent",
]
