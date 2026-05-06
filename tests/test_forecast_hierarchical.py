from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dstats.forecast.hierarchical import aggregate_m5_top_levels
from dstats.forecast.hierarchical import infer_time_columns
from dstats.forecast.hierarchical import parse_m5_id_columns
from dstats.forecast.hierarchical import rmsse
from dstats.forecast.hierarchical import top_level_alignment_metrics


def _bottom_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [
                "HOBBIES_1_001_CA_1_evaluation",
                "HOBBIES_1_002_CA_1_evaluation",
                "FOODS_2_001_TX_1_evaluation",
            ],
            "F1": [1.0, 3.0, 5.0],
            "F2": [2.0, 4.0, 6.0],
        }
    )


def test_infer_time_columns_orders_numeric_suffixes():
    df = pd.DataFrame(columns=["id", "F10", "F2", "F1"])

    assert infer_time_columns(df) == ["F1", "F2", "F10"]


def test_parse_m5_id_columns_adds_metadata():
    out = parse_m5_id_columns(_bottom_forecasts())

    first = out.iloc[0]
    assert first["cat_id"] == "HOBBIES"
    assert first["dept_id"] == "HOBBIES_1"
    assert first["item_id"] == "HOBBIES_1_001"
    assert first["state_id"] == "CA"
    assert first["store_id"] == "CA_1"


def test_aggregate_m5_top_levels_sums_expected_groups():
    out = aggregate_m5_top_levels(_bottom_forecasts())
    by_id = out.set_index("id_str")

    assert by_id.loc["all", ["F1", "F2"]].tolist() == [9.0, 12.0]
    assert by_id.loc["CA", ["F1", "F2"]].tolist() == [4.0, 6.0]
    assert by_id.loc["TX_1", ["F1", "F2"]].tolist() == [5.0, 6.0]
    assert by_id.loc["HOBBIES", ["F1", "F2"]].tolist() == [4.0, 6.0]
    assert by_id.loc["FOODS_2", ["F1", "F2"]].tolist() == [5.0, 6.0]


def test_rmsse_uses_training_scale():
    value = rmsse(
        actual=np.array([2.0, 4.0, 6.0]),
        predicted=np.array([2.0, 5.0, 7.0]),
        train=np.array([1.0, 2.0, 4.0, 7.0]),
    )

    assert value == pytest.approx(np.sqrt((2.0 / 3.0) / (14.0 / 3.0)))


def test_top_level_alignment_metrics_compares_bottom_up_to_reference():
    bottom = _bottom_forecasts()
    reference = aggregate_m5_top_levels(bottom)
    reference.loc[reference["id_str"] == "all", "F1"] += 1.0

    metrics = top_level_alignment_metrics(bottom, reference)
    by_id = metrics.set_index("id_str")

    assert by_id.loc["all", "mean_error"] == pytest.approx(-0.5)
    assert by_id.loc["all", "rmse"] == pytest.approx(np.sqrt(0.5))
    assert np.isfinite(metrics["rmsse"]).all()
