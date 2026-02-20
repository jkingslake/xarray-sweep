from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

from dask import compute, delayed
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


def xarray_sweep(
    function: Callable[..., Any],
    *,
    show_progress: bool = True,
    use_dask: bool = True,
    **params: Iterable[Any],
) -> xr.Dataset | xr.DataArray:
    """Run a function over the Cartesian product of parameter values.

    Parameters
    ----------
    function:
        Callable executed once per parameter combination.
    show_progress:
        Whether to show a tqdm progress bar.
    use_dask:
        Whether to execute parameter combinations with dask.
    **params:
        Parameter name -> iterable of values to sweep.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Unstacked xarray object with one dimension per swept parameter.
    """
    if not params:
        raise ValueError("At least one parameter sweep must be provided.")

    param_names = list(params.keys())
    param_values = [list(values) for values in params.values()]

    if any(len(values) == 0 for values in param_values):
        raise ValueError("Parameter sweeps must contain at least one value each.")

    combinations = list(product(*param_values))
    index = pd.MultiIndex.from_tuples(combinations, names=param_names)

    outputs: list[xr.Dataset | xr.DataArray]
    if use_dask:
        tasks = [
            delayed(_evaluate_combination)(
                function, dict(zip(param_names, combo, strict=True))
            )
            for combo in combinations
        ]
        if show_progress:
            with ProgressBar():
                outputs = list(compute(*tasks))
        else:
            outputs = list(compute(*tasks))
    else:
        outputs = []
        iterator = tqdm(index, disable=not show_progress)
        for combo in iterator:
            inputs = dict(zip(param_names, combo, strict=True))
            outputs.append(_evaluate_combination(function, inputs))

    stacked = xr.concat(outputs, dim="stacked_dim")
    stacked = stacked.assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, "stacked_dim")
    )
    return stacked.unstack("stacked_dim")


# Backward-compatible aliases for previous public names.
def grid_search(function: Callable[..., Any], **kwargs: Iterable[Any]) -> xr.Dataset | xr.DataArray:
    return xarray_sweep(function, **kwargs)


def gridSearch(function: Callable[..., Any], **kwargs: Iterable[Any]) -> xr.Dataset | xr.DataArray:
    return xarray_sweep(function, **kwargs)
