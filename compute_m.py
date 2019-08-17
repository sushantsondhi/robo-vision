"""Compute the Camera Matrix(M) of a given camera."""
import numpy as np


def gen_data_set(
    start: np.ndarray,
    dim_x: float = 40.0,
    dim_y: float = 40.0,
    dist_x: float = 20.0,
    dist_y: float = 20.0,
    m: int = 5,
    n: int = 5,
) -> list:
    """Documentation for generating data set.

    function to generate a data set of 3D co-ordinates as defined by a 'start'
    3-tuple, rectangles of size ('dim_x', 'dim_y') mxn in number spaced apart
    by 'dist_x' and 'dist_y'.
    """
    data_set = []
    for r in range(m):
        for c in range(n):
            p1 = np.array(
                [
                    start[0] + c * (dim_x + dist_x),
                    start[1] + r * (dim_y + dist_y),
                    start[2],
                ]
            )
            p2 = p1 + np.array([dist_x, 0.0, 0.0])
            p3 = p1 + np.array([dist_x, dist_y, 0.0])
            p4 = p1 + np.array([0.0, dist_y, 0.0])
            data_set.extend([p1, p2, p3, p4])

    return data_set


if __name__ == "__main__":
    start = np.array([500, 0, 0]).astype(np.float64)
    data_set = gen_data_set(start)
    print("Length of set:", len(data_set))
    for id, elem in enumerate(data_set):
        print("Element {}: ({}, {}, {})".format(id, elem[0], elem[1], elem[2]))
