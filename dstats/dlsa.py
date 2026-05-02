"""First DLSA migration slice.

This module intentionally keeps only the core path needed for a small Spark 4
run: simulate data, fit local logistic models, reduce local estimators, and run
the DLSA selector. Airline-data cleaning and dummy handling remain in the old
package for now.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log, sqrt

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType
from sklearn.linear_model import LogisticRegression


def simulate_logistic(
    sample_size: int,
    n_features: int,
    partition_num: int,
    *,
    seed: int | None = 123,
    label_col: str = "label",
    partition_col: str = "partition_id",
) -> pd.DataFrame:
    """Create a small logistic-regression fixture with systematic partitions."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if partition_num <= 0:
        raise ValueError("partition_num must be positive")

    rng = np.random.default_rng(seed)
    n_active = max(1, int(n_features * 0.4))
    beta = np.zeros(n_features)
    beta[:n_active] = 1.0

    x = rng.random((sample_size, n_features)) - 0.5
    prob = 1.0 / (1.0 + np.exp(-(x @ beta)))
    y = rng.binomial(1, prob)

    feature_cols = [f"x{i}" for i in range(n_features)]
    pdf = pd.DataFrame(x, columns=feature_cols)
    pdf.insert(0, label_col, y.astype(np.int64))
    pdf.insert(0, partition_col, np.arange(sample_size, dtype=np.int64) % partition_num)
    return pdf


def logistic_result_schema(
    feature_cols: Sequence[str],
    *,
    fit_intercept: bool = False,
) -> StructType:
    """Schema returned by one local logistic model fit."""

    coef_cols = _coefficient_columns(feature_cols, fit_intercept)
    return StructType(
        [
            StructField("par_id", LongType(), False),
            StructField("coef", DoubleType(), False),
            StructField("Sig_invMcoef", DoubleType(), False),
            *[StructField(col, DoubleType(), False) for col in coef_cols],
        ]
    )


def fit_logistic_partition(
    pdf: pd.DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
    partition_col: str = "partition_id",
    fit_intercept: bool = False,
    max_iter: int = 500,
) -> pd.DataFrame:
    """Fit one local logistic model and return DLSA reducer components."""

    missing = [col for col in [label_col, *feature_cols] if col not in pdf.columns]
    if missing:
        raise ValueError(f"Missing columns in partition data: {missing}")

    x_pdf = pdf.loc[:, list(feature_cols)].astype(float)
    y = pdf[label_col].to_numpy(dtype=np.int64)
    classes = np.unique(y)
    if len(classes) < 2:
        partition = pdf[partition_col].iloc[0] if partition_col in pdf.columns and len(pdf) else "unknown"
        raise ValueError(f"Partition {partition!r} has only one response class")

    model = LogisticRegression(
        solver="newton-cg",
        C=1e12,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
    )
    model.fit(x_pdf, y)

    prob = model.predict_proba(x_pdf)[:, 1]
    x = x_pdf.to_numpy(dtype=float)
    coef_cols = _coefficient_columns(feature_cols, fit_intercept)

    if fit_intercept:
        x = np.column_stack([np.ones(x.shape[0]), x])
        coef = np.concatenate([model.intercept_, model.coef_.ravel()])
    else:
        coef = model.coef_.ravel()

    weight = prob * (1.0 - prob)
    sig_inv = x.T @ (weight[:, None] * x)
    sig_inv_m_coef = sig_inv @ coef

    out = pd.DataFrame(sig_inv, columns=coef_cols)
    out.insert(0, "Sig_invMcoef", sig_inv_m_coef)
    out.insert(0, "coef", coef)
    out.insert(0, "par_id", np.arange(len(coef_cols), dtype=np.int64))
    return out


def fit_logistic_partitions(
    sdf: DataFrame,
    *,
    label_col: str,
    feature_cols: Sequence[str],
    partition_col: str = "partition_id",
    fit_intercept: bool = False,
    max_iter: int = 500,
) -> DataFrame:
    """Fit local logistic models with Spark 4 ``applyInPandas``."""

    schema = logistic_result_schema(feature_cols, fit_intercept=fit_intercept)

    def mapper(pdf: pd.DataFrame) -> pd.DataFrame:
        return fit_logistic_partition(
            pdf,
            label_col=label_col,
            feature_cols=feature_cols,
            partition_col=partition_col,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )

    return sdf.groupBy(partition_col).applyInPandas(mapper, schema=schema)


def dlsa_mapreduce(model_mapped_sdf: DataFrame) -> pd.DataFrame:
    """Aggregate local model components into global DLSA inputs."""

    columns = model_mapped_sdf.columns
    required = {"par_id", "coef", "Sig_invMcoef"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"model_mapped_sdf is missing required columns: {sorted(missing)}")

    matrix_cols = [col for col in columns if col not in required]
    if not matrix_cols:
        raise ValueError("model_mapped_sdf must include Hessian/precision matrix columns")

    agg_exprs = [
        F.sum(F.col("coef")).alias("coef"),
        F.sum(F.col("Sig_invMcoef")).alias("Sig_invMcoef"),
        F.count(F.col("coef")).alias("_n_models"),
        *[F.sum(F.col(col)).alias(col) for col in matrix_cols],
    ]
    summed = model_mapped_sdf.groupBy("par_id").agg(*agg_exprs).orderBy("par_id")
    summed_pdf = summed.toPandas()

    if summed_pdf.empty:
        raise ValueError("No mapped model rows were available to reduce")

    sig_inv = summed_pdf.loc[:, matrix_cols].to_numpy(dtype=float)
    sig_inv_m_coef = summed_pdf["Sig_invMcoef"].to_numpy(dtype=float)
    beta_by_ols = np.linalg.lstsq(sig_inv, sig_inv_m_coef, rcond=None)[0]
    beta_by_oneshot = (
        summed_pdf["coef"].to_numpy(dtype=float)
        / summed_pdf["_n_models"].to_numpy(dtype=float)
    )

    out = pd.DataFrame(sig_inv, columns=matrix_cols)
    out.insert(0, "beta_byONESHOT", beta_by_oneshot)
    out.insert(0, "beta_byOLS", beta_by_ols)
    return out


def dlsa_fit(
    sig_inv: pd.DataFrame | np.ndarray,
    beta: pd.Series | np.ndarray,
    sample_size: int,
    *,
    fit_intercept: bool = False,
) -> pd.DataFrame:
    """Run the DLSA AIC/BIC selector on reduced components."""

    sig_inv_array = np.asarray(sig_inv, dtype=float)
    beta_array = np.asarray(beta, dtype=float).reshape(-1)

    if sig_inv_array.ndim != 2 or sig_inv_array.shape[0] != sig_inv_array.shape[1]:
        raise ValueError("sig_inv must be a square matrix")
    if sig_inv_array.shape[0] != beta_array.shape[0]:
        raise ValueError("sig_inv and beta dimensions do not match")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    fitted = _lars_lsa(
        sig_inv_array,
        beta_array,
        intercept=fit_intercept,
        sample_size=sample_size,
    )

    aic_idx = int(np.argmin(fitted["AIC"]))
    bic_idx = int(np.argmin(fitted["BIC"]))
    beta_path = fitted["beta"]

    if fit_intercept:
        beta0 = fitted["beta0"]
        beta_by_aic = np.concatenate([[beta0[aic_idx]], beta_path[aic_idx, :]])
        beta_by_bic = np.concatenate([[beta0[bic_idx]], beta_path[bic_idx, :]])
    else:
        beta_by_aic = beta_path[aic_idx, :]
        beta_by_bic = beta_path[bic_idx, :]

    return pd.DataFrame({"beta_byAIC": beta_by_aic, "beta_byBIC": beta_by_bic})


def _coefficient_columns(feature_cols: Sequence[str], fit_intercept: bool) -> list[str]:
    cols = list(feature_cols)
    if fit_intercept:
        cols.insert(0, "intercept")
    return cols


def _backsolvet(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.triu(r, 0).T, x)


def _update_r(
    xnew: float,
    xold: np.ndarray,
    r: np.ndarray | None = None,
    eps: float = np.finfo(float).eps,
) -> tuple[np.ndarray, int]:
    norm_xnew = sqrt(float(xnew))
    if r is None:
        return np.array([[norm_xnew]], dtype=float), 1

    solved = _backsolvet(r, xold)
    rpp = norm_xnew**2 - float(np.sum(solved**2))
    rank = r.shape[1]
    if rpp <= eps:
        rpp = eps
    else:
        rpp = sqrt(rpp)
        rank += 1

    next_r = np.column_stack(
        (
            np.row_stack((r, np.zeros(r.shape[1]))),
            np.append(solved, rpp),
        )
    )
    return next_r, rank


def _delcol(r: np.ndarray, z: np.ndarray, k: int) -> np.ndarray:
    p = r.shape[0]
    r = np.delete(r, k, axis=1)
    z = np.matrix(z).T
    nz = z.shape[1]
    p1 = p - 1
    i = k + 1

    while i < p:
        a = r[i - 1, i - 1]
        b = r[i, i - 1]
        if b != 0:
            if abs(b) <= abs(a):
                tau = -b / a
                c = 1 / sqrt(1 + tau * tau)
                s = c * tau
            else:
                tau = -a / b
                s = 1 / sqrt(1 + tau * tau)
                c = s * tau

            r[i - 1, i - 1] = c * a - s * b
            r[i, i - 1] = s * a + c * b

            j = i + 1
            while j <= p1:
                a = r[i - 1, j - 1]
                b = r[i, j - 1]
                r[i - 1, j - 1] = c * a - s * b
                r[i, j - 1] = s * a + c * b
                j += 1

            j = 1
            while j <= nz:
                a = z[i - 1, j - 1]
                b = z[i, j - 1]
                z[i - 1, j - 1] = c * a - s * b
                z[i, j - 1] = s * a + c * b
                j += 1
        i += 1

    return r


def _downdate_r(r: np.ndarray, k: int) -> np.ndarray | None:
    p = r.shape[1]
    if p == 1:
        return None
    return np.delete(_delcol(r, np.ones(p), k), p - 1, axis=0)


def _lars_lsa(
    sigma0: np.ndarray,
    b0: np.ndarray,
    *,
    intercept: bool,
    sample_size: int,
    method: str = "lar",
    eps: float = np.finfo(float).eps,
    max_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Least-angle path used by the original DLSA implementation."""

    if method not in {"lar", "lasso"}:
        raise ValueError("method must be 'lar' or 'lasso'")

    p_total = sigma0.shape[0]
    if intercept:
        a11 = sigma0[0, 0]
        a12 = sigma0[1:, 0]
        a22 = sigma0[1:, 1:]
        sigma = a22 - np.outer(a12, a12) / a11
        b = b0[1:].copy()
        beta0_base = float(a12 @ b / a11)
    else:
        a11 = 1.0
        a12 = np.zeros(0)
        sigma = sigma0.copy()
        b = b0.copy()
        beta0_base = 0.0

    sigma = np.diag(np.abs(b)) @ sigma @ np.diag(np.abs(b))
    b = np.sign(b)
    m = sigma.shape[1]
    inactive_all = np.arange(1, m + 1)

    cvec = b @ sigma
    if max_steps is None:
        max_steps = 8 * m

    beta_path = np.zeros((max_steps + 1, m))
    first = np.zeros(m)
    active = np.array([], dtype=int)
    sign = np.array([], dtype=float)
    r = None
    rank = 0
    drops: np.ndarray | bool = False
    ignores = np.array([], dtype=int)
    k = 0

    while k < max_steps and len(active) < m:
        k += 1
        inactive = np.delete(inactive_all, active - 1)
        c = cvec[inactive - 1]
        cmax = max(abs(c))

        if not np.asarray(drops).any():
            new = inactive[abs(c) >= cmax - eps]
            c = c[abs(c) < cmax - eps]
            for inew in new:
                xold = np.asarray(sigma[inew - 1, active - 1]).reshape(-1)
                r, rank = _update_r(sigma[inew - 1, inew - 1], xold, r, eps=eps)
                if rank == len(active):
                    if r.shape[0] > 1 and r.shape[1] > 1:
                        r = r[:-1, :-1]
                    ignores = np.append(ignores, inew).astype(int)
                else:
                    if first[inew - 1] == 0:
                        first[inew - 1] = k
                    active = np.append(active, inew).astype(int)
                    sign = np.append(sign, np.sign(cvec[inew - 1]))

        if r is None or len(active) == 0:
            break

        gi1 = np.linalg.solve(np.triu(r, 0), _backsolvet(r, sign))
        aa = 1 / sqrt(float(np.sum(gi1 * sign)))
        w = aa * gi1

        if len(active) >= m:
            gamhat = cmax / aa
        else:
            drop_cols = (np.append(active, ignores) - 1).astype(int)
            remaining_sigma = np.delete(sigma, drop_cols, axis=1)
            a = w @ remaining_sigma[active - 1, :]
            gam = np.append((cmax - c) / (aa - a), (cmax + c) / (aa + a))
            gamhat = min(np.append(gam[gam > eps], cmax / aa))

        if method == "lasso":
            b1 = beta_path[k - 1, active - 1]
            z1 = -b1 / w
            zmin = np.min(np.append(z1[z1 > eps], gamhat))
            if zmin < gamhat:
                gamhat = zmin
                drops = z1 == zmin
            else:
                drops = False

        beta_path[k, :] = beta_path[k - 1, :]
        beta_path[k, active - 1] = beta_path[k, active - 1] + gamhat * w
        cvec = cvec - gamhat * (sigma[:, active - 1] @ w)

        if method == "lasso" and np.asarray(drops).any():
            dropid = np.where(drops)[0]
            for item in dropid[::-1]:
                r = _downdate_r(r, int(item))
            beta_path[k, active[drops] - 1] = 0
            active = active[~drops]
            sign = sign[~drops]

    beta_path = beta_path[: k + 1, :]
    dff = b.reshape(-1, 1) - beta_path.T
    rss = np.diag(dff.T @ sigma @ dff)

    if intercept:
        beta_path = (np.abs(b0[1:p_total]).reshape(-1, 1) * beta_path.T).T
        beta0 = beta0_base - (a12 @ beta_path.T) / a11
    else:
        beta_path = (np.abs(b0).reshape(-1, 1) * beta_path.T).T
        beta0 = np.zeros(k + 1)

    dof = (np.abs(beta_path) > eps).sum(axis=1)
    bic = rss + log(sample_size) * dof
    aic = rss + 2 * dof
    return {"AIC": aic, "BIC": bic, "beta": beta_path, "beta0": beta0}


dlsa_mapred = dlsa_mapreduce
dlsa = dlsa_fit


__all__ = [
    "dlsa",
    "dlsa_fit",
    "dlsa_mapred",
    "dlsa_mapreduce",
    "fit_logistic_partition",
    "fit_logistic_partitions",
    "logistic_result_schema",
    "simulate_logistic",
]
