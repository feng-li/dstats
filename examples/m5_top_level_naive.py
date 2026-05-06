"""Evaluate a last-value baseline on prepared M5 top-level hierarchy data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.forecast.hierarchical import hierarchy_rmsse
from dstats.forecast.hierarchical import infer_time_columns


DEFAULT_DATA = Path("data/m5_top_levels.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if not args.path.exists():
        raise FileNotFoundError(args.path)

    hierarchy = pd.read_parquet(args.path)
    day_cols = infer_time_columns(hierarchy, prefixes=("d_",))
    if len(day_cols) <= args.horizon:
        raise ValueError("Not enough history for the requested horizon")

    train_cols = day_cols[: -args.horizon]
    actual_cols = day_cols[-args.horizon :]
    forecast_cols = [f"F{i}" for i in range(1, args.horizon + 1)]

    forecast = pd.DataFrame({"id_str": hierarchy["id_str"]})
    last_observed = hierarchy[train_cols[-1]]
    for col in forecast_cols:
        forecast[col] = last_observed

    scores = hierarchy_rmsse(
        hierarchy,
        forecast,
        train_cols=train_cols,
        actual_cols=actual_cols,
        forecast_cols=forecast_cols,
    )

    print(
        "M5 top-level naive baseline: "
        f"series={len(scores)}, horizon={args.horizon}, "
        f"mean_rmsse={scores['rmsse'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
