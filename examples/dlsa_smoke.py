"""Small Spark 4 smoke check for the migrated DLSA path."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.dlsa import dlsa_fit
from dstats.dlsa import dlsa_mapreduce
from dstats.dlsa import fit_logistic_partitions
from dstats.dlsa import simulate_logistic
from dstats.spark import get_spark


def main() -> None:
    spark = get_spark(
        "dstats-dlsa-smoke",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        sample_size = 1_000
        n_features = 5
        partition_num = 4
        feature_cols = [f"x{i}" for i in range(n_features)]

        pdf = simulate_logistic(
            sample_size=sample_size,
            n_features=n_features,
            partition_num=partition_num,
            seed=7,
        )
        sdf = spark.createDataFrame(pdf)

        mapped = fit_logistic_partitions(
            sdf,
            label_col="label",
            feature_cols=feature_cols,
        )
        mapped_count = mapped.count()
        assert mapped_count == n_features * partition_num

        reduced = dlsa_mapreduce(mapped)
        assert reduced.shape == (n_features, n_features + 2)

        selected = dlsa_fit(
            reduced.iloc[:, 2:],
            reduced["beta_byOLS"],
            sample_size=sample_size,
        )
        assert selected.shape == (n_features, 2)
        assert selected.columns.tolist() == ["beta_byAIC", "beta_byBIC"]
    finally:
        spark.stop()

    print("DLSA smoke check passed")


if __name__ == "__main__":
    main()
