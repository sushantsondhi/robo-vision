"""Compute the Camera Parameters(A, R, T) from given image."""
import sys
import pickle
import numpy as np
import math
from typing import Tuple
from marker import List_5_tuple

def get_camera_params_from_m(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.empty((3, 3))
    R = np.empty((3, 3))
    T = np.empty((3, 1))

    T[2,0] = M[2,3]
    R[2,:] = M[2,:3]

    m1, m2, m3 = M[0,:3], M[1,:3], M[2,:3]
    A[0,2] = np.dot(m1, m3)
    A[1,2] = np.dot(m2, m3)

    A[1,1] = math.sqrt(np.dot(m2, m2)-A[1,2]*A[1,2])

    R[0,:] = np.cross(m2, m3)/A[1,1]
    R[1,:] = (m2 - A[1,2]*R[2,:])/A[1,1]

    A[0,0] = np.dot(m1, R[0,:])
    A[0,1] = np.dot(m1, R[1,:])

    T[1,0] = (M[1,3] - A[1,2]*T[2,0])/A[1,1]
    T[0,0] = (M[0,3] - A[0,2]*T[2,0] - A[0,1]*T[1,0])/A[0,0]

    A[1,0], A[2,0], A[2,1], A[2,2] = 0, 0, 0, 1

    return A, R, T


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
