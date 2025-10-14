"""Simple scripted opponents for the CRLite environment."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

Lane = str
OpponentAction = Optional[Tuple[int, Lane]]
OpponentState = Mapping[str, Any]

LANES: Tuple[Lane, Lane] = ("L", "R")


def _normalize_affordable(indices: Iterable[int]) -> Tuple[int, ...]:
    """Return a sorted tuple of affordable hand indices."""

    return tuple(sorted(int(idx) for idx in indices))


def _cheapest_index(indices: Sequence[int], costs: Optional[Sequence[int]]) -> Optional[int]:
    """Select the index with the smallest associated cost."""

    if not indices:
        return None
    if not costs:
        return int(indices[0])
    return int(min(indices, key=lambda idx: costs[idx]))


def _alternate_lane(step: int) -> Lane:
    """Alternate lanes based on the 20 step spam window."""

    window = max(step, 0) // 20
    return LANES[window % len(LANES)]


def idle_opponent(state: OpponentState) -> OpponentAction:
    """Always return ``None`` to represent a no-op action."""

    return None


def greedy_spammer(state: OpponentState) -> OpponentAction:
    """Every 20 steps play the cheapest affordable card while alternating lanes."""

    step = int(state.get("step", 0))
    if step % 20 != 0:
        return None

    affordable = _normalize_affordable(state.get("affordable_idxs", ()))
    if not affordable:
        return None

    costs = state.get("hand_costs")
    cost_seq: Optional[Sequence[int]] = costs if isinstance(costs, Sequence) else None
    card_idx = _cheapest_index(affordable, cost_seq)
    if card_idx is None:
        return None

    lane = _alternate_lane(step)

    # If lane load information is available, prefer the lighter lane but still alternate on ties.
    lane_load_raw = state.get("lane_load")
    if isinstance(lane_load_raw, Mapping):
        load_left = int(lane_load_raw.get("L", 0))
        load_right = int(lane_load_raw.get("R", 0))
        if load_left != load_right:
            lane = "L" if load_left < load_right else "R"

    return card_idx, lane


def mirror_opponent(state: OpponentState) -> OpponentAction:
    """Play a random affordable card in the same lane as the player's last action."""

    affordable = _normalize_affordable(state.get("affordable_idxs", ()))
    if not affordable:
        return None

    last_lane_raw = state.get("last_player_lane")
    lane: Optional[Lane]
    if isinstance(last_lane_raw, str) and last_lane_raw.upper() in LANES:
        lane = last_lane_raw.upper()
    elif last_lane_raw in (0, 1):
        lane = LANES[int(last_lane_raw)]
    else:
        lane = None

    if lane not in LANES:
        return None

    rng = state.get("rng")
    if hasattr(rng, "choice"):
        try:
            card_idx = int(rng.choice(affordable))  # type: ignore[arg-type]
        except TypeError:
            card_idx = affordable[0]
    else:
        card_idx = affordable[0]

    return card_idx, lane


def get_opponent(name: str) -> Callable[[OpponentState], OpponentAction]:
    """Return the scripted opponent callable referenced by ``name``."""

    mapping: Dict[str, Callable[[OpponentState], OpponentAction]] = {
        "idle": idle_opponent,
        "greedy": greedy_spammer,
        "greedy_spammer": greedy_spammer,
        "mirror": mirror_opponent,
    }

    try:
        return mapping[name]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown opponent '{name}'. Valid options: {valid}.") from exc


__all__ = [
    "OpponentAction",
    "OpponentState",
    "get_opponent",
    "greedy_spammer",
    "idle_opponent",
    "mirror_opponent",
]
