from __future__ import annotations

import numpy as np

from dstats.dqr import dqr_fit
from dstats.dqr import fit_quantile_pilot
from dstats.dqr import simulate_quantile


def test_fit_quantile_pilot_on_synthetic_data():
    feature_cols = ["intercept", "x0", "x1"]
    pdf = simulate_quantile(
        sample_size=120,
        n_features=2,
        partition_num=3,
        seed=23,
    )

    pilot = fit_quantile_pilot(
        pdf,
        label_col="label",
        feature_cols=feature_cols,
        quantile=0.5,
    )

    assert len(pilot.params) == len(feature_cols)
    assert np.isfinite(pilot.params).all()


def test_dqr_synthetic_path(spark):
    sample_size = 260
    n_features = 3
    partition_num = 3
    feature_cols = ["intercept", *[f"x{i}" for i in range(n_features)]]
    pdf = simulate_quantile(
        sample_size=sample_size,
        n_features=n_features,
        partition_num=partition_num,
        seed=29,
    )
    sdf = spark.createDataFrame(pdf)

    fitted = dqr_fit(
        sdf,
        label_col="label",
        feature_cols=feature_cols,
        quantile=0.5,
        pilot_fraction=0.4,
        pilot_seed=29,
    )

    assert fitted.shape == (len(feature_cols), 5)
    assert fitted.index.tolist() == feature_cols
    assert np.isfinite(fitted["beta_dqr"]).all()
    assert np.isfinite(fitted["se_dqr"]).all()
