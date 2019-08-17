"""Compute the Camera Matrix(M) from given image."""
import sys
import pickle
import numpy as np


def get_m(data_set: list) -> np.ndarray:
    """Documentation for a function.

    Given a set of 3D to 2D point correspondences compute the camera matrix
    using the Direct Linear Transform Method Cm = p.
    """

    def gen_row(point: tuple, id: int) -> list:
        """Documentation for a function.

        Given a 5-tuple (X, Y, Z, x, y) return the co-efficients of the linear
        system used to solve M. 'id' determines the row returned.
        """
        x_or_y = -1.0 * elem[3 + id]
        co_efficients = []
        if (id == 0):
            co_efficients.extend(point[0:3] + [1])
            co_efficients.extend([0, 0, 0, 0])
            co_efficients.extend([point[0] * x_or_y, point[1] * x_or_y, point[2] * x_or_y])
        elif (id == 1):
            co_efficients.extend([0, 0, 0, 0])
            co_efficients.extend(point[0:3] + [1])
            co_efficients.extend([point[0] * x_or_y, point[1] * x_or_y, point[2] * x_or_y])
        return co_efficients

    # Build the co-efficient and constant matrix from the data set
    data_size = len(data_set) * 2
    p = np.zeros(data_size)
    C = np.zeros((data_size, 11))
    for id, elem in enumerate(data_set):
        C[2 * id] = gen_row(elem, 0)
        C[2 * id + 1] = gen_row(elem, 1)
        p[2 * id] = elem[3]
        p[2 * id + 1] = elem[4]
    # Compute the co-efficients of M using Moore-Penrose inverse
    m = np.matmul(np.linalg.pinv(C), p)

    return m


if __name__ == "__main__":
    data_file = sys.argv[1]
    with open(data_file, "rb") as df:
        data_obj = pickle.load(df)
        data_set = data_obj.tuple_list

    co_effs = get_m(data_set)
    print(co_effs)
