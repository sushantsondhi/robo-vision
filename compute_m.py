"""Compute the Camera Matrix(M) from given image."""
import sys
import pickle
import numpy as np
from marker import List_5_tuple


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
            co_efficients.extend(point[0:3] + (1,))
            co_efficients.extend([0, 0, 0, 0])
            co_efficients.extend([point[0] * x_or_y, point[1] * x_or_y, point[2] * x_or_y])
        elif (id == 1):
            co_efficients.extend([0, 0, 0, 0])
            co_efficients.extend(point[0:3] + (1,))
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

    #resize the matrix to 3x4
    M = np.append(m, [1]).reshape((3,4))
    scl = np.linalg.norm(M[2,:3])
# to do normalise the last row of M
    return M / scl


if __name__ == "__main__":
    with open("left2.16mm_1.pkl", 'rb') as f:
        data_obj = pickle.load(f)
        data_set = data_obj.tuple_list
        print(data_set)

    co_effs = get_m(data_set)
    print(co_effs)
