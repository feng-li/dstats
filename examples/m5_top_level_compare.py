"""Compare M5 top-level baseline forecasts in one compact table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.forecast.hierarchical import hierarchy_rmsse
from dstats.forecast.hierarchical import infer_time_columns
from dstats.forecast.hierarchical import naive_hierarchy_forecast
from m5_top_level_autoarima import _forecast_autoarima


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

    rows = []
    rows.append(
        _score_method(
            "last",
            hierarchy,
            naive_hierarchy_forecast(
                hierarchy,
                train_cols=train_cols,
                horizon=args.horizon,
                method="last",
                season_length=args.season_length,
            ),
            train_cols,
            actual_cols,
            forecast_cols,
        )
    )
    rows.append(
        _score_method(
            "seasonal",
            hierarchy,
            naive_hierarchy_forecast(
                hierarchy,
                train_cols=train_cols,
                horizon=args.horizon,
                method="seasonal",
                season_length=args.season_length,
            ),
            train_cols,
            actual_cols,
            forecast_cols,
        )
    )
    rows.append(
        _score_method(
            "autoarima",
            hierarchy,
            _forecast_autoarima(
                hierarchy,
                train_cols=train_cols,
                forecast_cols=forecast_cols,
                season_length=args.season_length,
                nmodels=args.nmodels,
            ),
            train_cols,
            actual_cols,
            forecast_cols,
        )
    )

    print(pd.DataFrame(rows).to_string(index=False))


def _score_method(
    method: str,
    hierarchy: pd.DataFrame,
    forecast: pd.DataFrame,
    train_cols: list[str],
    actual_cols: list[str],
    forecast_cols: list[str],
) -> dict[str, float | int | str]:
    scores = hierarchy_rmsse(
        hierarchy,
        forecast,
        train_cols=train_cols,
        actual_cols=actual_cols,
        forecast_cols=forecast_cols,
    )
    return {
        "method": method,
        "series": len(scores),
        "horizon": len(forecast_cols),
        "mean_rmsse": round(float(scores["rmsse"].mean()), 4),
    }


if __name__ == "__main__":
    main()
