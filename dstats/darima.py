"""Python-native DARIMA migration slice.

This module translates the small R-backed DARIMA path into Python. Automatic
ARIMA selection is handled by statsforecast's AutoARIMA.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType
from statsforecast.models import AutoARIMA


def ar_coefficients(
    *,
    ar: Sequence[float] | np.ndarray = (),
    d: int = 0,
    ma: Sequence[float] | np.ndarray = (),
    sar: Sequence[float] | np.ndarray = (),
    seasonal_d: int = 0,
    sma: Sequence[float] | np.ndarray = (),
    mean: float = 0.0,
    drift: float = 0.0,
    period: int = 1,
    tol: int = 500,
) -> np.ndarray:
    """Convert SARIMA coefficients to the approximating AR representation."""

    if tol <= 0:
        raise ValueError("tol must be positive")
    if period <= 0:
        raise ValueError("period must be positive")

    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)
    sar = np.asarray(sar, dtype=float)
    sma = np.asarray(sma, dtype=float)

    ar_poly = np.concatenate([[1.0], -ar])
    ar_poly = _poly_power(ar_poly, d)

    if period > 1 and len(sar):
        seasonal_ar = np.zeros(period * len(sar) + 1)
        seasonal_ar[0] = 1.0
        for idx, value in enumerate(sar, start=1):
            seasonal_ar[period * idx] = -value
    else:
        seasonal_ar = np.array([1.0])
    if period > 1 and seasonal_d:
        seasonal_diff = np.zeros(period + 1)
        seasonal_diff[0] = 1.0
        seasonal_diff[period] = -1.0
        seasonal_ar = np.polynomial.polynomial.polymul(
            seasonal_ar,
            _poly_power(seasonal_diff, seasonal_d),
        )
    ar_total = _poly_trim(np.polynomial.polynomial.polymul(ar_poly, seasonal_ar))

    ma_poly = np.concatenate([[1.0], ma])
    if period > 1 and len(sma):
        seasonal_ma = np.zeros(period * len(sma) + 1)
        seasonal_ma[0] = 1.0
        for idx, value in enumerate(sma, start=1):
            seasonal_ma[period * idx] = value
    else:
        seasonal_ma = np.array([1.0])
    ma_total = _poly_trim(np.polynomial.polynomial.polymul(ma_poly, seasonal_ma))

    theta = -ma_total[1:]
    if len(theta) == 0:
        theta = np.array([0.0])

    phi = -np.concatenate([ar_total[1:], np.zeros(tol)])
    q = len(theta)
    pie_work = np.zeros(q + 1 + tol)
    pie_work[q] = 1.0
    for j in range(1, tol + 1):
        history = np.array([pie_work[j + q - 1 - idx] for idx in range(q)])
        pie_work[j + q] = -phi[j - 1] + float(theta @ history)

    pie = pie_work[q : q + tol + 1]
    pie = -pie[1 : tol + 1]

    time_weights = np.arange(1, tol + 1, dtype=float)
    c0 = mean * (1.0 - pie.sum()) + drift * float(time_weights @ pie)
    c1 = drift * (1.0 - pie.sum())
    return np.concatenate([[c0, c1], pie])


def darima_result_schema(tol: int) -> StructType:
    """Schema returned by one DARIMA partition fit."""

    return StructType(
        [
            StructField("par_id", LongType(), False),
            StructField("Sig_inv_value", DoubleType(), False),
            *[StructField(col, DoubleType(), False) for col in _coef_columns(tol)],
        ]
    )


def fit_darima_partition(
    pdf: pd.DataFrame,
    *,
    value_col: str,
    partition_col: str = "partition_id",
    time_col: str | None = None,
    period: int = 1,
    tol: int = 50,
    d: int | None = None,
    seasonal_d: int | None = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    max_d: int = 2,
    max_D: int = 1,
    stepwise: bool = True,
    approximation: bool | None = False,
    allowmean: bool = True,
    allowdrift: bool = True,
    nmodels: int = 94,
) -> pd.DataFrame:
    """Fit one partition and return weighted AR-representation coefficients."""

    if time_col:
        pdf = pdf.sort_values(time_col)
    x = pdf[value_col].dropna().to_numpy(dtype=float)
    if len(x) <= max(10, 2 * period):
        raise ValueError("DARIMA partition is too small for the requested model")

    model = AutoARIMA(
        d=d,
        D=seasonal_d,
        max_p=max_p,
        max_q=max_q,
        max_P=max_P,
        max_Q=max_Q,
        max_order=max_order,
        max_d=max_d,
        max_D=max_D,
        seasonal=period > 1,
        season_length=period,
        stepwise=stepwise,
        approximation=approximation,
        allowmean=allowmean,
        allowdrift=allowdrift,
        nmodels=nmodels,
    ).fit(x)
    fitted = model.model_
    coef_dict = fitted.get("coef", {})
    arma = fitted.get("arma", (0, 0, 0, 0, period, 0, 0))
    model_period = int(arma[4]) if len(arma) >= 5 else period
    model_d = int(arma[5]) if len(arma) >= 6 else 0
    model_D = int(arma[6]) if len(arma) >= 7 else 0

    sigma2 = float(fitted.get("sigma2", np.nanvar(fitted.get("residuals", x), ddof=1)))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("DARIMA partition produced a non-positive innovation variance")

    mean = float(coef_dict.get("intercept", coef_dict.get("mean", 0.0)))
    drift = float(coef_dict.get("drift", 0.0))
    coef = ar_coefficients(
        ar=_indexed_coefficients(coef_dict, "ar"),
        d=model_d,
        ma=_indexed_coefficients(coef_dict, "ma"),
        sar=_indexed_coefficients(coef_dict, "sar"),
        seasonal_d=model_D,
        sma=_indexed_coefficients(coef_dict, "sma"),
        mean=mean,
        drift=drift,
        period=model_period,
        tol=tol,
    )

    sig_inv_value = len(x) / sigma2
    par_id = int(pdf[partition_col].iloc[0]) if partition_col in pdf.columns else 0
    return pd.DataFrame(
        [[par_id, sig_inv_value, *(sig_inv_value * coef)]],
        columns=["par_id", "Sig_inv_value", *_coef_columns(tol)],
    )


def fit_darima_partitions(
    sdf: DataFrame,
    *,
    value_col: str,
    partition_col: str = "partition_id",
    time_col: str | None = None,
    period: int = 1,
    tol: int = 50,
    d: int | None = None,
    seasonal_d: int | None = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    max_d: int = 2,
    max_D: int = 1,
    stepwise: bool = True,
    approximation: bool | None = False,
    allowmean: bool = True,
    allowdrift: bool = True,
    nmodels: int = 94,
) -> DataFrame:
    """Fit DARIMA partition models with Spark 4 ``applyInPandas``."""

    schema = darima_result_schema(tol)

    def mapper(pdf: pd.DataFrame) -> pd.DataFrame:
        return fit_darima_partition(
            pdf,
            value_col=value_col,
            partition_col=partition_col,
            time_col=time_col,
            period=period,
            tol=tol,
            d=d,
            seasonal_d=seasonal_d,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            max_order=max_order,
            max_d=max_d,
            max_D=max_D,
            stepwise=stepwise,
            approximation=approximation,
            allowmean=allowmean,
            allowdrift=allowdrift,
            nmodels=nmodels,
        )

    return sdf.groupBy(partition_col).applyInPandas(mapper, schema=schema)


def darima_mapreduce(model_mapped_sdf: DataFrame, sample_size: int) -> pd.DataFrame:
    """Combine partition DARIMA estimates."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    columns = model_mapped_sdf.columns
    required = {"par_id", "Sig_inv_value"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"model_mapped_sdf is missing required columns: {sorted(missing)}")

    coef_cols = [col for col in columns if col not in required]
    summed = model_mapped_sdf.agg(
        F.sum("Sig_inv_value").alias("Sig_inv_value"),
        *[F.sum(col).alias(col) for col in coef_cols],
    ).toPandas()

    sig_inv_sum = float(summed["Sig_inv_value"].iloc[0])
    sig_inv_m_coef = summed.loc[0, coef_cols].to_numpy(dtype=float)
    theta = sig_inv_m_coef / sig_inv_sum
    sigma = (sample_size / sig_inv_sum) * np.eye(len(coef_cols))

    out = pd.DataFrame(sigma, columns=coef_cols)
    out.insert(0, "Theta_tilde", theta)
    return out


def darima_forecast(
    theta: pd.Series | np.ndarray,
    sigma: pd.DataFrame | np.ndarray,
    x: pd.Series | np.ndarray,
    *,
    period: int = 1,
    h: int = 1,
    level: float = 95,
) -> pd.DataFrame:
    """Forecast from combined DARIMA coefficients."""

    if h <= 0:
        raise ValueError("h must be positive")
    theta_array = np.asarray(theta, dtype=float).reshape(-1)
    sigma_array = np.asarray(sigma, dtype=float)
    x_array = np.asarray(x, dtype=float).reshape(-1)
    if theta_array.size < 3:
        raise ValueError("theta must include c0, c1, and at least one AR coefficient")

    sigma2 = float(np.trace(sigma_array) / sigma_array.shape[0])
    p = theta_array.size - 2
    ar = theta_array[2:]

    y = np.concatenate([x_array.copy(), np.zeros(h)])
    n = len(x_array)
    for step in range(1, h + 1):
        lag_values = y[n + step - 1 - np.arange(1, p + 1)]
        y[n + step - 1] = float(theta_array @ np.concatenate([[1.0, n + step], lag_values]))

    pred = y[n : n + h]
    psi = _ar_to_ma(ar, max(0, h - 1))
    variances = np.cumsum(np.concatenate([[1.0], psi**2])) * sigma2
    se = np.sqrt(variances[:h])
    q = norm.ppf(0.5 * (1.0 + level / 100.0))
    return pd.DataFrame(
        {
            "pred": pred,
            "lower": pred - q * se,
            "upper": pred + q * se,
        }
    )


def model_eval(
    x: pd.Series | np.ndarray,
    xx: pd.Series | np.ndarray,
    *,
    period: int,
    pred: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
    level: float = 95,
) -> pd.DataFrame:
    """Calculate MASE, sMAPE, and MSIS for forecasts."""

    x = np.asarray(x, dtype=float)
    xx = np.asarray(xx, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    diffs = np.abs(x[period:] - x[:-period]) if len(x) > period else np.abs(np.diff(x))
    scaling = float(np.mean(diffs))
    if not np.isfinite(scaling) or scaling == 0:
        raise ValueError("Cannot evaluate forecasts with zero seasonal scale")

    mase = np.abs(xx - pred) / scaling
    smape = (np.abs(xx - pred) * 200.0) / (np.abs(xx) + np.abs(pred))
    alpha = (100.0 - level) / 100.0
    msis = (
        upper
        - lower
        + (2.0 / alpha) * (lower - xx) * (lower > xx)
        + (2.0 / alpha) * (xx - upper) * (upper < xx)
    ) / scaling
    return pd.DataFrame({"mase": mase, "smape": smape, "msis": msis})


def simulate_ar1(
    sample_size: int = 240,
    *,
    phi: float = 0.6,
    sigma: float = 0.5,
    seed: int = 123,
    partition_num: int = 4,
) -> pd.DataFrame:
    """Create a deterministic AR(1) fixture with contiguous partitions."""

    rng = np.random.default_rng(seed)
    y = np.zeros(sample_size)
    for idx in range(1, sample_size):
        y[idx] = phi * y[idx - 1] + rng.normal(scale=sigma)
    partition_size = int(np.ceil(sample_size / partition_num))
    return pd.DataFrame(
        {
            "time": np.arange(sample_size, dtype=np.int64),
            "value": y,
            "partition_id": np.minimum(
                np.arange(sample_size, dtype=np.int64) // partition_size,
                partition_num - 1,
            ),
        }
    )


def _coef_columns(tol: int) -> list[str]:
    return ["c0", "c1", *[f"pi{i}" for i in range(1, tol + 1)]]


def _poly_power(poly: np.ndarray, power: int) -> np.ndarray:
    out = np.array([1.0])
    for _ in range(power):
        out = np.polynomial.polynomial.polymul(out, poly)
    return _poly_trim(out)


def _poly_trim(poly: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    if len(poly) == 0:
        return np.array([1.0])
    last = len(poly) - 1
    while last > 0 and abs(poly[last]) < tol:
        last -= 1
    return poly[: last + 1]


def _ar_to_ma(ar: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return np.zeros(0)
    psi_full = np.zeros(steps + 1)
    psi_full[0] = 1.0
    for idx in range(1, steps + 1):
        upto = min(len(ar), idx)
        psi_full[idx] = sum(ar[j - 1] * psi_full[idx - j] for j in range(1, upto + 1))
    return psi_full[1:]


def _indexed_coefficients(coef_dict: dict, prefix: str) -> np.ndarray:
    values = []
    idx = 1
    while f"{prefix}{idx}" in coef_dict:
        values.append(float(coef_dict[f"{prefix}{idx}"]))
        idx += 1
    return np.asarray(values, dtype=float)


dlsa_mapreduce = darima_mapreduce
darima_forec = darima_forecast


__all__ = [
    "ar_coefficients",
    "darima_forec",
    "darima_forecast",
    "darima_mapreduce",
    "darima_result_schema",
    "dlsa_mapreduce",
    "fit_darima_partition",
    "fit_darima_partitions",
    "model_eval",
    "simulate_ar1",
]
