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
from xarray_sweep import grid_search


def model(a: float, b: float):
    return a + b

out = grid_search(model, a=[0.1, 0.2], b=[1.0, 2.0])
print(out)
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
