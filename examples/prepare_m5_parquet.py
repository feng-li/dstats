"""Convert local M5 CSV data files to Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("rawcode/m5-accuracy-competition/m5-data")
DEFAULT_OUTPUT = Path("data/m5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["error", "overwrite", "ignore"], default="overwrite")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        out_path = args.out_dir / f"{csv_path.stem}.parquet"
        if out_path.exists():
            if args.mode == "error":
                raise FileExistsError(out_path)
            if args.mode == "ignore":
                print(f"{out_path} already exists")
                continue
            out_path.unlink()

        df = pd.read_csv(csv_path, low_memory=False)
        df.to_parquet(out_path, index=False)
        print(f"Wrote {len(df)} rows, {len(df.columns)} columns to {out_path}")


if __name__ == "__main__":
    main()
