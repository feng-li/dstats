"""Prepare top-level M5 hierarchy aggregates from a sales CSV or Parquet file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.forecast.hierarchical import aggregate_m5_top_levels
from dstats.forecast.hierarchical import infer_time_columns


DEFAULT_OUTPUT = Path("data/m5_top_levels.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sales", type=Path, help="M5 sales_train_* CSV or Parquet file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.sales.exists():
        raise FileNotFoundError(args.sales)

    sales = _read_sales(args.sales)
    value_cols = infer_time_columns(sales, prefixes=("d_", "F"))
    aggregates = aggregate_m5_top_levels(sales, value_cols=value_cols)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".csv":
        aggregates.to_csv(args.out, index=False)
    else:
        aggregates.to_parquet(args.out, index=False)

    print(
        f"Wrote {len(aggregates)} top-level series and {len(value_cols)} time columns "
        f"to {args.out}"
    )


def _read_sales(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported sales file suffix: {path.suffix!r}")


if __name__ == "__main__":
    main()
