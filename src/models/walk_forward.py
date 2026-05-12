from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def make_walk_forward_splits(
    n_rows: int,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[WalkForwardSplit]:
    splits: list[WalkForwardSplit] = []
    start = 0
    while start + train_bars + val_bars + test_bars <= n_rows:
        train_end = start + train_bars
        val_end = train_end + val_bars
        test_end = val_end + test_bars
        splits.append(WalkForwardSplit(start, train_end, train_end, val_end, val_end, test_end))
        start += step_bars
    if not splits and n_rows >= 100:
        train_end = int(n_rows * 0.6)
        val_end = int(n_rows * 0.8)
        splits.append(WalkForwardSplit(0, train_end, train_end, val_end, val_end, n_rows))
    return splits
