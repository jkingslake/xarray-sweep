from xarray_sweep import grid_search


def model(a: float, b: float) -> float:
    return a + b


if __name__ == "__main__":
    out = grid_search(model, a=[0.1, 0.2], b=[1.0, 2.0])
    print(out)
