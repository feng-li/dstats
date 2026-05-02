from __future__ import annotations

import numpy as np

from dstats.dlsa import dlsa_fit
from dstats.dlsa import dlsa_mapreduce
from dstats.dlsa import fit_logistic_partitions
from dstats.dlsa import simulate_logistic


def test_dlsa_synthetic_path(spark):
    sample_size = 240
    partition_num = 3
    n_features = 4
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
    assert mapped.count() == n_features * partition_num

    reduced = dlsa_mapreduce(mapped)
    selected = dlsa_fit(
        reduced.iloc[:, 2:],
        reduced["beta_byOLS"],
        sample_size=sample_size,
    )

    assert reduced.shape == (n_features, n_features + 2)
    assert selected.shape == (n_features, 2)
    assert np.isfinite(selected.to_numpy()).all()
