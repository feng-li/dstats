"""Evaluate StatsForecast AutoARIMA on prepared M5 top-level hierarchy data."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.forecast.hierarchical import hierarchy_rmsse
from dstats.forecast.hierarchical import infer_time_columns


DEFAULT_DATA = Path("data/m5_top_levels.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--season-length", type=int, default=7)
    parser.add_argument("--nmodels", type=int, default=30)
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args.season_length <= 0:
        raise ValueError("season-length must be positive")
    if args.nmodels <= 0:
        raise ValueError("nmodels must be positive")
    if not args.path.exists():
        raise FileNotFoundError(args.path)

    hierarchy = pd.read_parquet(args.path)
    day_cols = infer_time_columns(hierarchy, prefixes=("d_",))
    if len(day_cols) <= args.horizon:
        raise ValueError("Not enough history for the requested horizon")

    train_cols = day_cols[: -args.horizon]
    actual_cols = day_cols[-args.horizon :]
    forecast_cols = [f"F{i}" for i in range(1, args.horizon + 1)]

    forecast = _forecast_autoarima(
        hierarchy,
        train_cols=train_cols,
        forecast_cols=forecast_cols,
        season_length=args.season_length,
        nmodels=args.nmodels,
    )
    scores = hierarchy_rmsse(
        hierarchy,
        forecast,
        train_cols=train_cols,
        actual_cols=actual_cols,
        forecast_cols=forecast_cols,
    )

    print(
        "M5 top-level AutoARIMA baseline: "
        f"series={len(scores)}, horizon={args.horizon}, "
        f"season_length={args.season_length}, mean_rmsse={scores['rmsse'].mean():.4f}"
    )


def _forecast_autoarima(
    hierarchy: pd.DataFrame,
    *,
    train_cols: list[str],
    forecast_cols: list[str],
    season_length: int,
    nmodels: int,
) -> pd.DataFrame:
    rows: list[list[float | str]] = []
    for row in hierarchy.itertuples(index=False):
        series = pd.Series(row._asdict())
        y = series.loc[train_cols].to_numpy(dtype=float)
        model = AutoARIMA(
            season_length=season_length,
            seasonal=season_length > 1,
            stepwise=True,
            approximation=True,
            max_p=3,
            max_q=3,
            max_P=1,
            max_Q=1,
            max_order=5,
            nmodels=nmodels,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.fit(y).predict(h=len(forecast_cols))["mean"]
        rows.append([series["id_str"], *np.asarray(pred, dtype=float)])

    return pd.DataFrame(rows, columns=["id_str", *forecast_cols])


if __name__ == "__main__":
    main()
