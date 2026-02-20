from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os
from typing import Any

from dask import delayed
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm


def _evaluate_combination(
    function: Callable[..., Any], inputs: dict[str, Any]
) -> xr.Dataset | xr.DataArray:
    result = function(**inputs)

    if isinstance(result, xr.Dataset):
        return result.assign_coords(inputs)
    return xr.DataArray(result, coords=inputs)


def _assemble_outputs(
    outputs: list[xr.Dataset | xr.DataArray], index: pd.MultiIndex
) -> xr.Dataset | xr.DataArray:
    stacked = xr.concat(outputs, dim="stacked_dim")
    stacked = stacked.assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, "stacked_dim")
    )
    return stacked.unstack("stacked_dim")


def xarray_sweep(
    function: Callable[..., Any],
    *,
    show_progress: bool = True,
    use_dask: bool = False,
    compute: bool = True,
    n_jobs: int = 1,
    **params: Iterable[Any],
) -> xr.Dataset | xr.DataArray | Delayed:
    """Run a function over the Cartesian product of parameter values.

    Parameters
    ----------
    function:
        Callable executed once per parameter combination.
    show_progress:
        Whether to show a tqdm progress bar.
    use_dask:
        Whether to execute parameter combinations with dask.
    compute:
        Whether to evaluate immediately. If False, returns a delayed object.
    n_jobs:
        Number of worker threads for parallel execution. Use ``1`` (default)
        for sequential execution, ``-1`` to use all available CPUs, or any
        positive integer to set the exact number of workers. Ignored when
        ``use_dask=True``.
    **params:
        Parameter name -> iterable of values to sweep.

    Returns
    -------
    xr.Dataset | xr.DataArray | dask.delayed.Delayed
        Unstacked xarray object with one dimension per swept parameter, or
        a delayed computation when compute=False.
    """
    if not params:
        raise ValueError("At least one parameter sweep must be provided.")

    param_names = list(params.keys())
    param_values = [list(values) for values in params.values()]

    if any(len(values) == 0 for values in param_values):
        raise ValueError("Parameter sweeps must contain at least one value each.")

    combinations = list(product(*param_values))
    index = pd.MultiIndex.from_tuples(combinations, names=param_names)

    if use_dask:
        tasks = [
            delayed(_evaluate_combination)(
                function, dict(zip(param_names, combo, strict=True))
            )
            for combo in combinations
        ]
        result = delayed(_assemble_outputs)(tasks, index)
        if not compute:
            return result
        if show_progress:
            with ProgressBar():
                return result.compute()
        return result.compute()

    if not compute:
        raise ValueError("compute=False requires use_dask=True.")

    all_inputs = [dict(zip(param_names, combo, strict=True)) for combo in combinations]

    if n_jobs != 1:
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_evaluate_combination, function, inputs)
                for inputs in all_inputs
            ]
            iterator = tqdm(futures, disable=not show_progress)
            outputs = [f.result() for f in iterator]
    else:
        outputs: list[xr.Dataset | xr.DataArray] = []
        iterator = tqdm(all_inputs, disable=not show_progress)
        for inputs in iterator:
            outputs.append(_evaluate_combination(function, inputs))

    return _assemble_outputs(outputs, index)


# Backward-compatible aliases for previous public names.
def grid_search(function: Callable[..., Any], **kwargs: Iterable[Any]) -> xr.Dataset | xr.DataArray:
    return xarray_sweep(function, **kwargs)


def gridSearch(function: Callable[..., Any], **kwargs: Iterable[Any]) -> xr.Dataset | xr.DataArray:
    return xarray_sweep(function, **kwargs)
