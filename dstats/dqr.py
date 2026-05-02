"""First DQR migration slice.

This module keeps the first Spark 4 path small: fit a pilot quantile
regression, compute partition-level one-step components with ``applyInPandas``,
and reduce those components on the driver. Project-specific cleaning, dummy
encoding, and communication-cost helpers remain in the old package for now.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType
from scipy import stats
from scipy.stats import norm
from statsmodels.regression.quantile_regression import QuantReg


def simulate_quantile(
    sample_size: int,
    n_features: int,
    partition_num: int,
    *,
    quantile: float = 0.5,
    seed: int | None = 123,
    fit_intercept: bool = True,
    label_col: str = "label",
    partition_col: str = "partition_id",
) -> pd.DataFrame:
    """Create a dense numeric quantile-regression fixture."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if partition_num <= 0:
        raise ValueError("partition_num must be positive")
    _validate_quantile(quantile)

    rng = np.random.default_rng(seed)
    beta = np.zeros(n_features)
    beta[: max(1, n_features // 2)] = np.linspace(1.0, -0.7, max(1, n_features // 2))

    x = rng.normal(size=(sample_size, n_features))
    shifted_noise = rng.normal(size=sample_size) - norm.ppf(quantile)
    y = x @ beta + shifted_noise

    feature_cols = [f"x{i}" for i in range(n_features)]
    pdf = pd.DataFrame(x, columns=feature_cols)
    if fit_intercept:
        pdf.insert(0, "intercept", 1.0)
        y = y + 0.5
    pdf.insert(0, label_col, y)
    pdf.insert(0, partition_col, np.arange(sample_size, dtype=np.int64) % partition_num)
    return pdf


def fit_quantile_pilot(
    pdf: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
    quantile: float = 0.5,
    max_iter: int = 1_000,
):
    """Fit the driver-side pilot quantile regression."""

    _validate_quantile(quantile)
    missing = [col for col in [label_col, *feature_cols] if col not in pdf.columns]
    if missing:
        raise ValueError(f"Missing columns in pilot data: {missing}")

    clean_pdf = pdf.loc[:, [*feature_cols, label_col]].dropna()
    if len(clean_pdf) <= len(feature_cols):
        raise ValueError("Pilot sample is too small for the requested feature set")

    x = clean_pdf.loc[:, list(feature_cols)].astype(float)
    y = clean_pdf[label_col].astype(float)
    return QuantReg(endog=y, exog=x).fit(q=quantile, max_iter=max_iter)


def quantile_component_schema(
    feature_cols: Sequence[str],
    *,
    partition_col: str = "partition_id",
    include_score: bool = True,
    include_xtx: bool = True,
) -> StructType:
    """Schema returned by one DQR partition component calculation."""

    fields = [StructField(partition_col, LongType(), False)]
    if include_xtx:
        fields.extend(StructField(col, DoubleType(), False) for col in _xtx_columns(feature_cols))
    if include_score:
        fields.extend(StructField(col, DoubleType(), False) for col in _score_columns(feature_cols))
    fields.append(StructField("kernel_sum", DoubleType(), False))
    return StructType(fields)


def qr_asymptotic_components(
    pdf: pd.DataFrame,
    *,
    beta0: Sequence[float] | np.ndarray | pd.Series,
    bandwidth: float,
    label_col: str,
    feature_cols: Sequence[str],
    partition_col: str = "partition_id",
    quantile: float = 0.5,
    include_score: bool = True,
    include_xtx: bool = True,
) -> pd.DataFrame:
    """Compute one partition's DQR one-step components."""

    _validate_quantile(quantile)
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    missing = [col for col in [label_col, *feature_cols] if col not in pdf.columns]
    if missing:
        raise ValueError(f"Missing columns in partition data: {missing}")

    clean_pdf = pdf.loc[:, [*( [partition_col] if partition_col in pdf.columns else []), *feature_cols, label_col]].dropna()
    if clean_pdf.empty:
        raise ValueError("Partition has no complete rows")

    x = clean_pdf.loc[:, list(feature_cols)].to_numpy(dtype=float)
    y = clean_pdf[label_col].to_numpy(dtype=float)
    beta = np.asarray(beta0, dtype=float).reshape(-1)
    if beta.shape[0] != x.shape[1]:
        raise ValueError("beta0 length does not match feature columns")

    error = y - x @ beta
    kernel_sum = float(np.sum(norm.pdf(error / bandwidth)))

    if partition_col in clean_pdf.columns:
        partition_value = int(clean_pdf[partition_col].iloc[0])
    else:
        partition_value = 0

    row: dict[str, float | int] = {partition_col: partition_value}
    if include_xtx:
        xtx = x.T @ x
        for col, value in zip(_xtx_columns(feature_cols), xtx[np.tril_indices(x.shape[1])]):
            row[col] = float(value)
    if include_score:
        indicator = (error < 0).astype(float)
        score = x.T @ (quantile - indicator)
        for col, value in zip(_score_columns(feature_cols), score):
            row[col] = float(value)
    row["kernel_sum"] = kernel_sum
    return pd.DataFrame([row], columns=[field.name for field in quantile_component_schema(
        feature_cols,
        partition_col=partition_col,
        include_score=include_score,
        include_xtx=include_xtx,
    )])


def fit_quantile_partitions(
    sdf: DataFrame,
    *,
    beta0: Sequence[float] | np.ndarray | pd.Series,
    bandwidth: float,
    label_col: str,
    feature_cols: Sequence[str],
    partition_col: str = "partition_id",
    quantile: float = 0.5,
    include_score: bool = True,
    include_xtx: bool = True,
) -> DataFrame:
    """Compute DQR partition components with Spark 4 ``applyInPandas``."""

    schema = quantile_component_schema(
        feature_cols,
        partition_col=partition_col,
        include_score=include_score,
        include_xtx=include_xtx,
    )
    beta = np.asarray(beta0, dtype=float).reshape(-1)

    def mapper(pdf: pd.DataFrame) -> pd.DataFrame:
        return qr_asymptotic_components(
            pdf,
            beta0=beta,
            bandwidth=bandwidth,
            label_col=label_col,
            feature_cols=feature_cols,
            partition_col=partition_col,
            quantile=quantile,
            include_score=include_score,
            include_xtx=include_xtx,
        )

    return sdf.groupBy(partition_col).applyInPandas(mapper, schema=schema)


def dqr_reduce(
    mapped_sdf: DataFrame,
    *,
    beta0: Sequence[float] | np.ndarray | pd.Series,
    sample_size: int,
    bandwidth: float,
    quantile: float,
    feature_cols: Sequence[str],
    density_sdf: DataFrame | None = None,
) -> pd.DataFrame:
    """Reduce mapped DQR components into one-step estimates and standard errors."""

    _validate_quantile(quantile)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    feature_cols = list(feature_cols)
    beta = np.asarray(beta0, dtype=float).reshape(-1)
    if beta.shape[0] != len(feature_cols):
        raise ValueError("beta0 length does not match feature columns")

    mapped_pdf = mapped_sdf.toPandas()
    if mapped_pdf.empty:
        raise ValueError("No mapped DQR components were available to reduce")

    xtx_cols = _xtx_columns(feature_cols)
    score_cols = _score_columns(feature_cols)
    missing = [col for col in [*xtx_cols, *score_cols, "kernel_sum"] if col not in mapped_pdf.columns]
    if missing:
        raise ValueError(f"mapped_sdf is missing required columns: {missing}")

    xtx = _symmetric_from_lower(mapped_pdf.loc[:, xtx_cols].sum(axis=0).to_numpy(dtype=float), len(feature_cols))
    xtx_inv = np.linalg.pinv(xtx)

    score = mapped_pdf.loc[:, score_cols].sum(axis=0).to_numpy(dtype=float)
    kernel0 = float(mapped_pdf["kernel_sum"].sum())
    if kernel0 <= 0:
        raise ValueError("Mapped DQR components produced a non-positive kernel sum")

    f0_inv = sample_size * bandwidth / kernel0
    beta_dqr = beta + f0_inv * xtx_inv @ score

    density_pdf = density_sdf.toPandas() if density_sdf is not None else mapped_pdf
    if "kernel_sum" not in density_pdf.columns or density_pdf.empty:
        raise ValueError("density_sdf must include kernel_sum values")
    kernel1 = float(density_pdf["kernel_sum"].sum())
    if kernel1 <= 0:
        raise ValueError("Density components produced a non-positive kernel sum")

    f1_inv = sample_size * bandwidth / kernel1
    cov = xtx_inv * sample_size * quantile * (1.0 - quantile) * f1_inv**2
    var = np.diag(cov)
    se = np.sqrt(np.maximum(var, 0.0))
    dof = max(sample_size - len(feature_cols), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(beta_dqr / se), dof))

    return pd.DataFrame(
        {
            "beta_dqr": beta_dqr,
            "beta_pilot": beta,
            "var_dqr": var,
            "se_dqr": se,
            "pvalue_dqr": pvalues,
        },
        index=pd.Index(feature_cols, name="term"),
    )


def dqr_fit(
    sdf: DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
    partition_col: str = "partition_id",
    quantile: float = 0.5,
    pilot_fraction: float = 0.1,
    pilot_seed: int | None = 123,
    max_iter: int = 1_000,
) -> pd.DataFrame:
    """Run the first migrated DQR one-step estimator path."""

    _validate_quantile(quantile)
    if not 0 < pilot_fraction <= 1:
        raise ValueError("pilot_fraction must be in (0, 1]")

    feature_cols = list(feature_cols)
    columns = [partition_col, *feature_cols, label_col]
    clean_sdf = sdf.select(*columns).dropna()
    sample_size = clean_sdf.count()
    if sample_size <= len(feature_cols):
        raise ValueError("Input data is too small for the requested feature set")

    pilot_sdf = clean_sdf.sample(
        withReplacement=False,
        fraction=pilot_fraction,
        seed=pilot_seed,
    )
    pilot_pdf = pilot_sdf.select(*feature_cols, label_col).toPandas()
    min_pilot_size = min(sample_size, max(20, 3 * len(feature_cols)))
    if len(pilot_pdf) < min_pilot_size:
        pilot_pdf = (
            clean_sdf.orderBy(F.rand(seed=pilot_seed))
            .limit(min_pilot_size)
            .select(*feature_cols, label_col)
            .toPandas()
        )

    pilot = fit_quantile_pilot(
        pilot_pdf,
        label_col=label_col,
        feature_cols=feature_cols,
        quantile=quantile,
        max_iter=max_iter,
    )
    beta0 = pd.Series(np.asarray(pilot.params, dtype=float), index=feature_cols)
    bandwidth = float(pilot.bandwidth)

    mapped = fit_quantile_partitions(
        clean_sdf,
        beta0=beta0,
        bandwidth=bandwidth,
        label_col=label_col,
        feature_cols=feature_cols,
        partition_col=partition_col,
        quantile=quantile,
    )
    first_pass = dqr_reduce(
        mapped,
        beta0=beta0,
        sample_size=sample_size,
        bandwidth=bandwidth,
        quantile=quantile,
        feature_cols=feature_cols,
    )
    density = fit_quantile_partitions(
        clean_sdf,
        beta0=first_pass["beta_dqr"],
        bandwidth=bandwidth,
        label_col=label_col,
        feature_cols=feature_cols,
        partition_col=partition_col,
        quantile=quantile,
        include_score=False,
        include_xtx=False,
    )
    out = dqr_reduce(
        mapped,
        beta0=beta0,
        sample_size=sample_size,
        bandwidth=bandwidth,
        quantile=quantile,
        feature_cols=feature_cols,
        density_sdf=density,
    )
    out.attrs["bandwidth"] = bandwidth
    out.attrs["pilot_size"] = len(pilot_pdf)
    out.attrs["sample_size"] = sample_size
    out.attrs["quantile"] = quantile
    return out


def _xtx_columns(feature_cols: Sequence[str]) -> list[str]:
    p = len(feature_cols)
    return [f"xtx_{i}_{j}" for i in range(p) for j in range(i + 1)]


def _score_columns(feature_cols: Sequence[str]) -> list[str]:
    return [f"score_{i}" for i, _ in enumerate(feature_cols)]


def _symmetric_from_lower(values: np.ndarray, size: int) -> np.ndarray:
    expected = size * (size + 1) // 2
    if values.shape[0] != expected:
        raise ValueError(f"Expected {expected} lower-triangular values, got {values.shape[0]}")
    out = np.zeros((size, size), dtype=float)
    lower = np.tril_indices(size)
    out[lower] = values
    out = out + out.T - np.diag(np.diag(out))
    return out


def _validate_quantile(quantile: float) -> None:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")


dqr = dqr_fit

__all__ = [
    "dqr",
    "dqr_fit",
    "dqr_reduce",
    "fit_quantile_partitions",
    "fit_quantile_pilot",
    "qr_asymptotic_components",
    "quantile_component_schema",
    "simulate_quantile",
]
