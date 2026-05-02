"""Small Spark 4 smoke check for the migrated DQR path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.dqr import dqr_fit
from dstats.dqr import simulate_quantile
from dstats.spark import get_spark


def main() -> None:
    spark = get_spark(
        "dstats-dqr-smoke",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        sample_size = 800
        n_features = 4
        partition_num = 4
        feature_cols = ["intercept", *[f"x{i}" for i in range(n_features)]]

        pdf = simulate_quantile(
            sample_size=sample_size,
            n_features=n_features,
            partition_num=partition_num,
            quantile=0.5,
            seed=19,
        )
        sdf = spark.createDataFrame(pdf)

        fitted = dqr_fit(
            sdf,
            label_col="label",
            feature_cols=feature_cols,
            quantile=0.5,
            pilot_fraction=0.35,
            pilot_seed=19,
        )
        assert fitted.shape == (len(feature_cols), 5)
        assert fitted.columns.tolist() == [
            "beta_dqr",
            "beta_pilot",
            "var_dqr",
            "se_dqr",
            "pvalue_dqr",
        ]
        assert np.isfinite(fitted["beta_dqr"]).all()
        assert np.isfinite(fitted["se_dqr"]).all()
    finally:
        spark.stop()

    print("DQR smoke check passed")


if __name__ == "__main__":
    main()
