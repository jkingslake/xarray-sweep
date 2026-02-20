import xarray as xr

from xarray_sweep import grid_search, gridSearch, xarray_sweep


def test_xarray_sweep_with_scalar_output():
    def fn(a: int, b: int) -> int:
        return a + b

    out = xarray_sweep(fn, show_progress=False, a=[1, 2], b=[10, 20])

    assert isinstance(out, xr.DataArray)
    assert set(out.dims) == {"a", "b"}
    assert out.sel(a=1, b=10).item() == 11
    assert out.sel(a=2, b=20).item() == 22


def test_xarray_sweep_with_dataset_output():
    def fn(a: int, b: int) -> xr.Dataset:
        return xr.Dataset({"value": xr.DataArray(a * b)})

    out = xarray_sweep(fn, show_progress=False, a=[2, 3], b=[4, 5])

    assert isinstance(out, xr.Dataset)
    assert set(out.dims) == {"a", "b"}
    assert out["value"].sel(a=2, b=4).item() == 8
    assert out["value"].sel(a=3, b=5).item() == 15


def test_xarray_sweep_without_dask_still_works():
    def fn(a: int, b: int) -> int:
        return a * b

    out = xarray_sweep(fn, show_progress=False, use_dask=False, a=[2, 3], b=[4, 5])
    assert out.sel(a=2, b=4).item() == 8
    assert out.sel(a=3, b=5).item() == 15


def test_legacy_name_still_works():
    def fn(a: int) -> int:
        return a

    out = gridSearch(fn, a=[1, 2])
    assert set(out.dims) == {"a"}


def test_grid_search_alias_still_works():
    def fn(a: int) -> int:
        return a

    out = grid_search(fn, a=[1, 2])
    assert set(out.dims) == {"a"}


def test_empty_param_sweep_raises_value_error():
    def fn(a: int) -> int:
        return a

    try:
        xarray_sweep(fn, show_progress=False, a=[])
    except ValueError as exc:
        assert "at least one value" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty parameter list")
