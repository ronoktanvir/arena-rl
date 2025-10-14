"""Simple scripted opponents used by :class:`envs.crlite_env.CRLiteEnv`.

The environment interacts with opponents through lightweight callables that
accept a minimal ``state`` dictionary describing the current match and return a
``(hand_index, lane)`` tuple indicating the card to play and the target lane.
Returning ``None`` represents a no-op for the step.  The helpers below
implement three baseline behaviours requested by the environment.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
OpponentState = Dict[str, object]
OpponentAction = Optional[Tuple[int, int]]


def _select_cheapest(affordable: Iterable[int], costs: Sequence[int]) -> Optional[int]:
    """Return the index of the cheapest affordable card, if any."""

    try:
        return min(affordable, key=lambda idx: costs[idx])
    except ValueError:
        return None


def idle_opponent(state: OpponentState) -> OpponentAction:
    """Never plays a card, always returning ``None``."""

    _ = state
    return None


def greedy_spammer(state: OpponentState) -> OpponentAction:
    """Play the cheapest affordable card every 20 steps, alternating lanes."""

    step = int(state.get("step", 0))
    affordable = tuple(state.get("affordable_indices", ()))
    costs = tuple(state.get("hand_costs", ()))

    if not affordable or step % 20 != 0:
        return None

    choice = _select_cheapest(affordable, costs)
    if choice is None:
        return None

    lane = (step // 20) % 2
    return choice, lane


def mirror_opponent(state: OpponentState) -> OpponentAction:
    """Mirror the player's last lane with a random affordable card."""

    lane = state.get("last_player_lane")
    affordable = tuple(state.get("affordable_indices", ()))
    rng = state.get("rng")

    if lane is None or not affordable:
        return None

    if rng is not None and hasattr(rng, "choice"):
        index = int(rng.choice(affordable))  # type: ignore[arg-type]
    else:
        # Fallback to deterministic choice without RNG access.
        index = affordable[0]

    return index, int(lane)


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
    "OpponentState",
    "OpponentAction",
    "idle_opponent",
    "greedy_spammer",
    "mirror_opponent",
    "get_opponent",
]
