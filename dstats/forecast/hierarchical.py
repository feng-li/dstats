"""Small hierarchical forecast utilities migrated from the M5 notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import re

import numpy as np
import pandas as pd


M5_METADATA_COLUMNS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
M5_TOP_LEVEL_GROUPS = (
    ("state_id",),
    ("store_id",),
    ("cat_id",),
    ("dept_id",),
)
M5_ALL_LEVEL_GROUPS = (
    *M5_TOP_LEVEL_GROUPS,
    ("state_id", "cat_id"),
    ("state_id", "dept_id"),
    ("store_id", "cat_id"),
    ("store_id", "dept_id"),
    ("item_id",),
    ("item_id", "state_id"),
    ("id",),
)


def infer_time_columns(
    df: pd.DataFrame,
    prefixes: Sequence[str] = ("F", "d_"),
) -> list[str]:
    """Return forecast/history columns such as ``F1`` or ``d_1`` in numeric order."""

    matched: list[tuple[str, int, str]] = []
    for col in df.columns:
        name = str(col)
        for prefix in prefixes:
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
            if match:
                matched.append((prefix, int(match.group(1)), col))
                break
    return [col for _, _, col in sorted(matched, key=lambda item: (item[0], item[1]))]


def parse_m5_id_columns(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
) -> pd.DataFrame:
    """Add M5 metadata columns parsed from bottom-level item IDs."""

    if id_col not in df.columns:
        raise ValueError(f"{id_col!r} is required to parse M5 ids")

    out = df.copy()
    parts = out[id_col].astype(str).str.split("_", expand=True)
    if parts.shape[1] < 5:
        raise ValueError("M5 ids must look like CAT_DEPT_ITEM_STATE_STORE[_split]")

    out["cat_id"] = parts[0]
    out["dept_id"] = parts[0] + "_" + parts[1]
    out["item_id"] = parts[0] + "_" + parts[1] + "_" + parts[2]
    out["state_id"] = parts[3]
    out["store_id"] = parts[3] + "_" + parts[4]
    return out


def aggregate_m5_top_levels(
    df: pd.DataFrame,
    *,
    value_cols: Sequence[str] | None = None,
    id_col: str = "id",
    metadata_cols: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate bottom-level M5 forecasts to the notebook's top levels 1-5."""

    return aggregate_m5_levels(
        df,
        levels="top",
        value_cols=value_cols,
        id_col=id_col,
        metadata_cols=metadata_cols,
    )


def aggregate_m5_levels(
    df: pd.DataFrame,
    *,
    levels: str = "top",
    value_cols: Sequence[str] | None = None,
    id_col: str = "id",
    metadata_cols: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate bottom-level M5 forecasts to top or full hierarchy levels."""

    data = df.copy()
    value_cols = _value_columns(data, value_cols)
    group_specs = _m5_group_specs(levels, id_col=id_col)

    if metadata_cols:
        data = data.rename(columns={source: target for target, source in metadata_cols.items()})

    missing_metadata = [col for col in M5_METADATA_COLUMNS if col not in data.columns]
    if missing_metadata:
        data = parse_m5_id_columns(data, id_col=id_col)
        missing_metadata = [col for col in M5_METADATA_COLUMNS if col not in data.columns]
    if missing_metadata:
        raise ValueError(f"Missing M5 metadata columns: {missing_metadata}")

    aggregates = [_total_aggregate(data, value_cols)]
    for group_cols in group_specs:
        aggregates.append(_group_aggregate(data, group_cols, value_cols))

    return pd.concat(aggregates, ignore_index=True).loc[:, ["id_str", *value_cols]]


def rmsse(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    train: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | float:
    """Compute root mean squared scaled error along the last axis."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")

    train_array = actual_array if train is None else np.asarray(train, dtype=float)
    if train_array.shape[-1] < 2:
        raise ValueError("train must contain at least two time points")

    numerator = np.mean((predicted_array - actual_array) ** 2, axis=-1)
    denominator = np.mean(np.diff(train_array, axis=-1) ** 2, axis=-1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("RMSSE denominator must be positive")

    out = np.sqrt(numerator / denominator)
    return float(out) if np.ndim(out) == 0 else out


def hierarchy_rmsse(
    actual_hierarchy: pd.DataFrame,
    forecast_hierarchy: pd.DataFrame,
    *,
    train_cols: Sequence[str],
    actual_cols: Sequence[str],
    forecast_cols: Sequence[str] | None = None,
    id_col: str = "id_str",
) -> pd.DataFrame:
    """Compute RMSSE for each hierarchy row."""

    forecast_cols = list(actual_cols) if forecast_cols is None else list(forecast_cols)
    actual_cols = list(actual_cols)
    train_cols = list(train_cols)
    if len(actual_cols) != len(forecast_cols):
        raise ValueError("actual_cols and forecast_cols must have the same length")
    if id_col not in actual_hierarchy.columns or id_col not in forecast_hierarchy.columns:
        raise ValueError(f"Both inputs must include {id_col!r}")

    _require_columns(actual_hierarchy, [id_col, *train_cols, *actual_cols], "actual_hierarchy")
    _require_columns(forecast_hierarchy, [id_col, *forecast_cols], "forecast_hierarchy")

    aligned = actual_hierarchy.loc[:, [id_col, *train_cols, *actual_cols]].merge(
        forecast_hierarchy.loc[:, [id_col, *forecast_cols]],
        on=id_col,
        how="inner",
    )
    if aligned.empty:
        raise ValueError("No matching hierarchy ids found")

    scores = rmsse(
        aligned.loc[:, actual_cols].to_numpy(dtype=float),
        aligned.loc[:, forecast_cols].to_numpy(dtype=float),
        train=aligned.loc[:, train_cols].to_numpy(dtype=float),
    )
    return pd.DataFrame({id_col: aligned[id_col], "rmsse": scores})


def naive_hierarchy_forecast(
    hierarchy: pd.DataFrame,
    *,
    train_cols: Sequence[str],
    horizon: int,
    method: str = "seasonal",
    season_length: int = 7,
    id_col: str = "id_str",
    forecast_prefix: str = "F",
) -> pd.DataFrame:
    """Build a simple last-value or seasonal-naive hierarchy forecast."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    train_cols = list(train_cols)
    _require_columns(hierarchy, [id_col, *train_cols], "hierarchy")

    forecast = pd.DataFrame({id_col: hierarchy[id_col]})
    if method == "last":
        source_cols = [train_cols[-1]] * horizon
    elif method == "seasonal":
        if season_length <= 0:
            raise ValueError("season_length must be positive")
        if len(train_cols) < season_length:
            raise ValueError("train_cols must contain at least season_length columns")
        season_cols = train_cols[-season_length:]
        source_cols = [season_cols[idx % season_length] for idx in range(horizon)]
    else:
        raise ValueError("method must be one of: last, seasonal")

    for idx, source_col in enumerate(source_cols, start=1):
        forecast[f"{forecast_prefix}{idx}"] = hierarchy[source_col]
    return forecast


def top_level_alignment_metrics(
    bottom_forecasts: pd.DataFrame,
    reference_forecasts: pd.DataFrame,
    *,
    value_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare bottom-up top-level aggregates with reference top-level forecasts."""

    value_cols = _value_columns(bottom_forecasts, value_cols)
    bottom_agg = aggregate_m5_top_levels(bottom_forecasts, value_cols=value_cols)
    if "id_str" not in reference_forecasts.columns:
        raise ValueError("reference_forecasts must include an 'id_str' column")

    reference = reference_forecasts.loc[:, ["id_str", *value_cols]].copy()
    aligned = bottom_agg.merge(
        reference,
        on="id_str",
        how="inner",
        suffixes=("_bottom", "_reference"),
    )
    if aligned.empty:
        raise ValueError("No matching top-level ids found")

    bottom_values = aligned[[f"{col}_bottom" for col in value_cols]].to_numpy(dtype=float)
    reference_values = aligned[[f"{col}_reference" for col in value_cols]].to_numpy(dtype=float)
    error = bottom_values - reference_values

    return pd.DataFrame(
        {
            "id_str": aligned["id_str"],
            "mean_error": error.mean(axis=1),
            "mean_abs_error": np.abs(error).mean(axis=1),
            "rmse": np.sqrt(np.mean(error**2, axis=1)),
            "rmsse": rmsse(reference_values, bottom_values),
        }
    )


def _value_columns(df: pd.DataFrame, value_cols: Sequence[str] | None) -> list[str]:
    cols = list(value_cols) if value_cols is not None else infer_time_columns(df)
    if not cols:
        raise ValueError("No forecast/time columns were provided or inferred")
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing forecast/time columns: {missing}")
    return cols


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _total_aggregate(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame([df.loc[:, value_cols].sum(numeric_only=True)])
    out.insert(0, "id_str", "all")
    return out


def _group_aggregate(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    group_cols = list(group_cols)
    out = df.groupby(group_cols, as_index=False, sort=True)[list(value_cols)].sum()
    out.insert(0, "id_str", out.loc[:, group_cols].astype(str).agg("_".join, axis=1))
    return out.drop(columns=group_cols)


def _m5_group_specs(levels: str, *, id_col: str) -> tuple[tuple[str, ...], ...]:
    if levels == "top":
        return M5_TOP_LEVEL_GROUPS
    if levels == "all":
        return tuple((id_col,) if spec == ("id",) else spec for spec in M5_ALL_LEVEL_GROUPS)
    raise ValueError("levels must be one of: top, all")


__all__ = [
    "aggregate_m5_levels",
    "aggregate_m5_top_levels",
    "hierarchy_rmsse",
    "infer_time_columns",
    "naive_hierarchy_forecast",
    "parse_m5_id_columns",
    "rmsse",
    "top_level_alignment_metrics",
]
