"""Utility functions for tracking training metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


def exp_smooth(series: Iterable[float], alpha: float = 0.1) -> List[float]:
    """Return the exponential moving average of *series*.

    Args:
        series: Numeric samples to smooth.
        alpha: Smoothing factor in ``(0, 1]``. Higher values discount older
            observations faster.

    Returns:
        A list containing the smoothed values. The first value is the first
        element of *series* to preserve alignment.
    """

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    iterator = iter(series)
    try:
        first = next(iterator)
    except StopIteration:
        return []

    smoothed: List[float] = [float(first)]
    for value in iterator:
        next_value = alpha * float(value) + (1.0 - alpha) * smoothed[-1]
        smoothed.append(next_value)
    return smoothed


def win_rate_from_infos(infos: Sequence[Mapping[str, object]]) -> float:
    """Compute the mean win rate from a sequence of info dictionaries."""

    if not infos:
        return 0.0

    wins = 0
    for info in infos:
        win_value = info.get("win") if isinstance(info, Mapping) else None
        if isinstance(win_value, bool):
            wins += 1 if win_value else 0
        elif isinstance(win_value, (int, float)):
            wins += 1 if win_value else 0
    return wins / len(infos)


@dataclass
class _StubTensorBoardLogger:
    """A lightweight stand-in for a TensorBoard summary writer."""

    log_dir: Optional[Path]

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to the console."""
        location = f" ({self.log_dir})" if self.log_dir else ""
        print(f"[tb_logger{location}] step={step} {tag}={value:.4f}")


def tb_logger(log_dir: Optional[str] = None) -> _StubTensorBoardLogger:
    """Create a stub TensorBoard logger that prints scalar metrics.

    Args:
        log_dir: Optional directory to mimic TensorBoard log storage.

    Returns:
        An object exposing :meth:`log_scalar` that prints metrics.
    """

    path = Path(log_dir) if log_dir else None
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
    return _StubTensorBoardLogger(log_dir=path)


__all__ = ["exp_smooth", "win_rate_from_infos", "tb_logger"]
