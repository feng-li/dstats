# Migration Plan

## Goal

Migrate the archived distributed statistics code into one small Python package named
`dstats`, using Spark 4 and the current PySpark interface. Keep the initial package
flat. Do not introduce internal subpackages until the migrated code proves that the
extra structure is needed.

Current target layout:

```text
dstats/
  __init__.py
  _version.py
  spark.py
  dlsa.py
  darima.py
  dqr.py
```

The original package directories should remain as references during migration.
Move behavior gradually and keep runnable checks after each step.

## General Rules

- Start with direct migrations, not redesigns.
- Keep statistical code separable from Spark orchestration where practical.
- Prefer Spark 4 APIs such as `DataFrame.groupBy(...).applyInPandas(...)`.
- Avoid the old `PandasUDFType.GROUPED_MAP` pattern in new code.
- Avoid hardcoded cluster paths, Python paths, and `findspark`.
- Do not add `rpy2` to the new package. Translate R-backed routines to Python
  before moving them into `dstats`.
- Use `.envrc` for local Spark/PySpark environment settings.
- Preserve current outputs first; improve APIs only after behavior is covered by checks.
- Add small local datasets or synthetic fixtures before migrating each method.

## Environment

The local Spark 4 environment is:

```text
/home/fli/.virtualenvs/py3.12-spark4/
```

The repository `.envrc` should activate that environment and set:

```sh
SPARK_LOCAL_HOSTNAME=localhost
PYSPARK_PYTHON=/home/fli/.virtualenvs/py3.12-spark4/bin/python
PYSPARK_DRIVER_PYTHON=/home/fli/.virtualenvs/py3.12-spark4/bin/python
```

Basic verification command:

```sh
source .envrc
python -c "import dstats; from dstats.spark import get_spark; print(dstats.__version__)"
```

## Phase 1: Shared Spark Foundation

Status: started.

- Keep `dstats/spark.py` small.
- Provide one helper for creating a Spark session.
- Enable Arrow through modern Spark config keys.
- Keep local settings outside code in `.envrc`.
- Add only shared helpers that are needed by the first migrated package.

Avoid building a large framework before the first migration is complete.

## Phase 2: Migrate `dlsa` First

Source reference:

```text
dlsa/dlsa/
dlsa/projects/logistic_dlsa.py
```

Target:

```text
dstats/dlsa.py
```

Initial scope:

- Move the core DLSA aggregation behavior from `dlsa/dlsa/dlsa.py`.
- Move the logistic model helpers needed by the existing Spark example.
- Convert grouped map pandas UDF usage to `groupBy(...).applyInPandas(...)`.
- Keep dummy-variable and airline-data preprocessing minimal at first.
- Replace script globals with function parameters only where needed.

Suggested first public functions:

```python
dlsa_mapreduce(...)
dlsa_fit(...)
fit_logistic_partitions(...)
```

Validation:

- Add a tiny synthetic logistic dataset.
- Confirm the migrated Spark path runs under `local[2]`.
- Compare output shape and coefficient columns with the old implementation.
- Use numeric tolerances, not exact equality, for model coefficients.

Exit criteria:

- A local Spark 4 DLSA demo runs from the new `dstats.dlsa` module.
- No new code depends on `PandasUDFType`.
- No hardcoded user data paths are required for the demo.

## Phase 3: Migrate `darima` Second

Source reference:

```text
darima/darima/
darima/run_darima.py
darima/darima/R/
```

Target:

```text
dstats/darima.py
```

Initial scope:

- Move the partition-level ARIMA fitting wrapper.
- Move the DLSA-style ARIMA coefficient aggregation.
- Move forecasting and evaluation wrappers.
- Translate the existing R helper logic to Python.
- Use `statsforecast.models.AutoARIMA` for automatic ARIMA selection and
  NumPy/SciPy for AR conversion, forecast intervals, and evaluation.

Important constraint:

Use StatsForecast `AutoARIMA` as the replacement for R `forecast::auto.arima`.

Validation:

- Use a short built-in or synthetic seasonal time series.
- Verify partition fitting returns the expected schema.
- Verify aggregation returns `Theta_tilde` and coefficient columns.
- Verify forecast output has prediction and interval columns.

Exit criteria:

- A local Spark 4 DARIMA demo runs from `dstats.darima`.
- The old `run_darima.py` workflow has a minimal equivalent in the new module.

## Phase 4: Migrate `dqr` Third

Source reference:

```text
dqr/dqr/
dqr/projects/dqr_spark.py
```

Target:

```text
dstats/dqr.py
```

Initial scope:

- Move quantile-regression component calculations.
- Move Spark dummy-variable helpers only if needed for the first migrated demo.
- Move communication-cost helpers only after the core estimator path works.
- Convert any grouped pandas logic to `applyInPandas`.

Suggested first public functions:

```python
qr_asymptotic_components(...)
fit_quantile_partitions(...)
```

Validation:

- Use a tiny synthetic regression dataset.
- Check pilot estimator, one-step component output, and final coefficient shape.
- Compare migrated output with the old code on the same fixture.

Exit criteria:

- A local Spark 4 DQR demo runs from `dstats.dqr`.
- The estimator path no longer requires the old `dqr` package.
- Optional plotting and project scripts remain outside the initial migration.

## Deferred: `dts`

`dts` is not complete yet. Keep it later.

Do not migrate `dts` until:

- `dlsa`, `darima`, and `dqr` have working Spark 4 paths.
- The expected `dts` API and model workflow are clearer.
- Small deterministic checks exist for the MCMC and mapper behavior.

For now, use `dts/` only as a reference.

## Not In Initial Scope

- Large internal package hierarchy.
- Full CLI design.
- Full documentation site.
- Performance tuning for a real cluster.
- Migrating every old project script.

These can be added after the core migration paths are working.

## Working Checklist

- [x] Create Spark 4 virtualenv.
- [x] Store local Spark environment settings in `.envrc`.
- [x] Create flat `dstats` package skeleton.
- [x] Migrate shared Spark helper needed by DLSA.
- [x] Migrate first DLSA core slice.
- [x] Add a committed DLSA local demo/check.
- [x] Add a small DLSA check against Spark-written `airdelay_small.parquet`
      with `nominal_delay` and `real_delay` labels.
- [x] Migrate Python-native DARIMA core and local Spark demo.
- [x] Add a DARIMA electricity-data example using StatsForecast AutoARIMA.
- [x] Merge bundled electricity series into `data/electricity.parquet`.
- [ ] Migrate DQR core and local Spark demo.
- [ ] Revisit DTS once the first three migrations are stable.
