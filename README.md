# `dstats`

Distributed statistical modelling utilities for Spark 4 and modern PySpark.

This repository is consolidating the older standalone distributed-statistics
projects into one small Python package named `dstats`. The current package is
kept small and pragmatic:

```text
dstats/
  spark.py
  dlsa.py
  darima.py
  dqr.py
  forecast/
    hierarchical.py
```

The old project directories are kept as historical references during migration.
New code should use the `dstats.*` modules.

## Current Status

The first Spark 4 migration slices are available for:

- `dstats.dlsa`: distributed least-squares approximation for logistic examples.
- `dstats.darima`: distributed ARIMA aggregation and forecasting with
  `statsforecast.models.AutoARIMA`.
- `dstats.dqr`: distributed quantile regression by pilot sampling and one-step
  updating for dense numeric features.
- `dstats.forecast.hierarchical`: small M5-style hierarchical forecast
  aggregation and alignment helpers.

`dts` is deferred until the first three migrated paths are stable.

The new package does not use `rpy2`. R-backed routines from the old projects are
being translated to Python as they are migrated.

## Environment

The local development environment used for this migration is:

```text
/home/fli/.virtualenvs/py3.12-spark4/
```

Load it with:

```sh
source .envrc
```

The `.envrc` file sets `SPARK_LOCAL_HOSTNAME`, `PYSPARK_PYTHON`, and
`PYSPARK_DRIVER_PYTHON` for local Spark runs.

Install the package in editable mode when needed:

```sh
python -m pip install -e .
```

Quick import check:

```sh
python -c "import dstats; from dstats.spark import get_spark; print(dstats.__version__)"
```

## Tests

Install the test extra and run the compact synthetic suite:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

The tests cover shared Spark helpers and small synthetic paths for DLSA, DARIMA,
and DQR. Large prepared-data checks remain under `examples/`.

## Examples

Synthetic smoke checks:

```sh
python examples/dlsa_smoke.py
python examples/darima_smoke.py
python examples/dqr_smoke.py
python examples/forecast_hierarchical_smoke.py
```

Prepared-data examples:

```sh
python examples/dlsa_airdelay_small.py --path data/airdelay_small.parquet --nrows 5000 --partitions 4 --label-col nominal_delay
python examples/dlsa_airdelay_small.py --path data/airdelay_small.parquet --nrows 5000 --partitions 4 --label-col real_delay
python examples/darima_electricity.py
python examples/dqr_used_cars.py --nrows 3000 --partitions 4 --pilot-fraction 0.4
```

The compact `data/airdelay_small.parquet` artifact contains raw `ArrDelay` plus
the 10 standardized feature columns. `examples/dlsa_airdelay_small.py` derives
`nominal_delay` or `real_delay` from `ArrDelay` before modelling.

## Data Preparation

The example datasets are prepared as Parquet so Spark can read them efficiently:

```sh
python examples/prepare_airdelay_parquet.py --out data/airdelay_small.parquet --mode overwrite
python examples/prepare_electricity_parquet.py --out data/electricity.parquet --mode overwrite
python examples/prepare_used_cars_parquet.py --out data/used_cars.parquet --mode overwrite --coalesce 1
python examples/prepare_m5_hierarchy.py /path/to/sales_train_evaluation.csv --out data/m5_top_levels.parquet
python examples/prepare_m5_parquet.py --input-dir rawcode/m5-accuracy-competition/m5-data --out-dir data/m5 --mode overwrite
```

Current local prepared datasets:

- `data/airdelay_small.parquet`: compact airline-delay data with raw
  `ArrDelay` plus 10 standardized feature columns: `Year`, `Month`,
  `DayofMonth`, `DayOfWeek`, `DepTime`, `CRSDepTime`, `CRSArrTime`,
  `ActualElapsedTime`, `Distance`, and `DepDelay`.
- `data/electricity.parquet`: merged DARIMA electricity series in long format.
- `data/used_cars.parquet`: raw numeric used-car columns for DQR. The DQR
  example creates `log_price`, `intercept`, and standardized `z_*` features
  before modelling.
- `data/m5_top_levels.parquet`: optional M5 top-level hierarchy aggregates
  prepared from a local M5 sales CSV.
- `data/m5/`: ignored local Parquet conversion of the raw M5 CSV files.

Large generated data files may be better kept outside normal Git history or
managed with Git LFS.

## API Snapshot

DLSA:

```py
from dstats.dlsa import dlsa_fit
from dstats.dlsa import dlsa_mapreduce
from dstats.dlsa import fit_logistic_partitions
```

DARIMA:

```py
from dstats.darima import darima_forecast
from dstats.darima import darima_mapreduce
from dstats.darima import fit_darima_partitions
from dstats.darima import model_eval
```

DQR:

```py
from dstats.dqr import dqr_fit
from dstats.dqr import fit_quantile_partitions
from dstats.dqr import qr_asymptotic_components
```

Shared Spark helpers:

```py
from dstats.spark import get_spark
from dstats.spark import standardize_columns
from dstats.spark import with_partition_id
from dstats.spark import write_single_parquet
```

Forecast hierarchy helpers:

```py
from dstats.forecast.hierarchical import aggregate_m5_top_levels
from dstats.forecast.hierarchical import top_level_alignment_metrics
```

## Migration Notes

This is still a migration repository, not a finalized public API. The first goal
is to preserve working statistical paths under Spark 4 with small local checks.
Project-specific preprocessing, cluster tuning, plotting scripts, and full old
script parity are intentionally left outside the first package slice.

See [PLAN.md](PLAN.md) for the active migration plan and checklist.
