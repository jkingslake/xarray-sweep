# xarray-sweep

Small utility to run a function over a Cartesian product of parameters and return results as an unstacked `xarray` object.

## Install

```bash
pip install xarray-sweep
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Usage

```python
import xarray as xr
from xarray_sweep import xarray_sweep


def model(a: float, b: float):
    return a + b

out = xarray_sweep(model, a=[0.1, 0.2], b=[1.0, 2.0])
print(out)
```

`xarray_sweep` runs sequentially by default. To run parameter combinations in parallel using threads (useful for I/O-bound or numpy/xarray workloads that release the GIL), use `n_jobs`:

```python
# Use 4 worker threads
out = xarray_sweep(model, n_jobs=4, a=[0.1, 0.2], b=[1.0, 2.0])

# Use all available CPUs
out = xarray_sweep(model, n_jobs=-1, a=[0.1, 0.2], b=[1.0, 2.0])
```

To execute combinations with Dask:

```python
out = xarray_sweep(model, use_dask=True, a=[0.1, 0.2], b=[1.0, 2.0])
```

To build a delayed computation (for distributed workflows), use `compute=False`:

```python
delayed_out = xarray_sweep(model, use_dask=True, compute=False, a=[0.1, 0.2], b=[1.0, 2.0])
```

Then you can compute this later with

```python
out = delayed_out.compute()
```
 or if you like you can use a progress bar with:

 ```python
from dask.diagnostics import ProgressBar
with ProgressBar():
    out = delayed_out.compute()
```


## Run tests

```bash
pytest
```

## Build distribution artifacts

```bash
python -m build
```

## Publish to PyPI

1. Create a PyPI account and API token.
2. Build artifacts (`python -m build`).
3. Upload with Twine:

```bash
twine upload dist/*
```

For TestPyPI first:

```bash
twine upload --repository testpypi dist/*
```

## GitHub Actions CI

A workflow is included at `.github/workflows/ci.yml` that runs tests on pull requests and pushes to `main`.
