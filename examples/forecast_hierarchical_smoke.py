"""Small smoke check for M5-style hierarchical forecast helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.forecast.hierarchical import aggregate_m5_top_levels
from dstats.forecast.hierarchical import top_level_alignment_metrics


def main() -> None:
    bottom = pd.DataFrame(
        {
            "id": [
                "HOBBIES_1_001_CA_1_evaluation",
                "HOBBIES_1_002_CA_1_evaluation",
                "FOODS_2_001_TX_1_evaluation",
            ],
            "F1": [1.0, 3.0, 5.0],
            "F2": [2.0, 4.0, 6.0],
            "F3": [3.0, 5.0, 7.0],
        }
    )

    aggregates = aggregate_m5_top_levels(bottom)
    reference = aggregates.copy()
    reference.loc[reference["id_str"] == "all", "F1"] += 1.0
    metrics = top_level_alignment_metrics(bottom, reference)

    assert aggregates.loc[aggregates["id_str"] == "all", "F1"].iloc[0] == 9.0
    assert metrics["rmsse"].notna().all()
    print(
        "Forecast hierarchy smoke check passed: "
        f"aggregates={len(aggregates)}, mean_rmse={metrics['rmse'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
