from xarray_sweep import xarray_sweep


def model(a: float, b: float) -> float:
    return a + b


if __name__ == "__main__":
    out = xarray_sweep(model, a=[0.1, 0.2], b=[1.0, 2.0])
    print(out)
