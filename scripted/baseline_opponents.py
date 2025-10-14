"""Baseline scripted opponents for :mod:`envs.crlite_env`.

The environment expects opponents to expose a simple callable interface accepting a
plain ``state`` dictionary and returning either ``None`` (no action) or a tuple of
``(hand_index, lane)`` describing the card to play and the lane target (``"L"`` or
``"R"``).  The helpers below intentionally keep their logic lightweight and make as
few assumptions as possible about the environment internals.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

Lane = str
OpponentAction = Optional[Tuple[int, Lane]]
OpponentState = Mapping[str, object]

LANES: Tuple[Lane, Lane] = ("L", "R")


def _cheapest_affordable(affordable: Iterable[int], costs: Optional[Sequence[int]]) -> Optional[int]:
    """Return the index of the cheapest affordable card if available."""

    affordable_list = list(affordable)
    if not affordable_list:
        return None

    if not costs:
        # Without cost data, just take the first affordable index to remain robust.
        return affordable_list[0]

    return min(affordable_list, key=lambda idx: costs[idx])


def _lane_choice_from_load(step: int, lane_load: Mapping[Lane, int]) -> Lane:
    """Choose a lane preferring the lower load and alternating on ties."""

    load_left = int(lane_load.get("L", 0))
    load_right = int(lane_load.get("R", 0))

    if load_left < load_right:
        return "L"
    if load_right < load_left:
        return "R"

    # If loads are equal, alternate based on the number of spam windows elapsed.
    return LANES[(step // 20) % len(LANES)]


def idle_opponent(state: OpponentState) -> OpponentAction:
    """Never play a card."""

    return None


def greedy_spammer(state: OpponentState) -> OpponentAction:
    """Every 20 steps play the cheapest card, alternating lanes when loads tie."""

    step = int(state.get("step", 0))
    if step % 20 != 0:
        return None

    affordable = state.get("affordable_idxs", ())
    costs = state.get("hand_costs")
    costs_seq: Optional[Sequence[int]] = None
    if isinstance(costs, Sequence):
        costs_seq = costs  # type: ignore[assignment]

    card_idx = _cheapest_affordable(affordable, costs_seq)
    if card_idx is None:
        return None

    lane_load = state.get("lane_load", {})
    if not isinstance(lane_load, Mapping):
        lane_load = {}

    lane = _lane_choice_from_load(step, lane_load)
    return card_idx, lane


def mirror_opponent(state: OpponentState) -> OpponentAction:
    """Mirror the player's last lane by playing a random affordable card."""

    lane_raw = state.get("last_player_lane")
    if isinstance(lane_raw, str):
        lane = lane_raw.upper()
    elif lane_raw in (0, 1):
        lane = LANES[int(lane_raw)]
    else:
        lane = None
    affordable = list(state.get("affordable_idxs", ()))
    if lane not in LANES or not affordable:
        return None

    rng = state.get("rng")
    if hasattr(rng, "choice"):
        try:
            choice = rng.choice(affordable)  # type: ignore[call-arg]
        except TypeError:
            choice = affordable[0]
    else:
        # Fall back to deterministic behaviour for environments without RNG access.
        choice = affordable[0]

    return int(choice), lane  # type: ignore[arg-type]


def get_opponent(name: str) -> Callable[[OpponentState], OpponentAction]:
    """Return the scripted opponent callable associated with ``name``."""

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
