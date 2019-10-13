"""Compute the Camera Parameters(A, R, T) from given image."""
import sys
import math
import pickle
import numpy as np
from typing import Tuple
from marker import List_5_tuple


def __test(M: np.ndarray, A: np.ndarray, R: np.ndarray, T: np.ndarray, data_set: list, scl: float = 8.8 / 384):
    err_2d = err_3d = 0
    for idx, (X, Y, Z, x, y) in enumerate(data_set):
        Pw = np.array([X, Y, Z, 1])
        p = np.array([x, y])
        p_m = np.matmul(M, Pw)
        p_m = p_m[:2] / p_m[-1]
        # compute the 2D error
        err_2d += math.pow(np.linalg.norm(p_m - p * scl), 2)
        # compute the 3D error
        Pw_dir = np.matmul(np.linalg.inv(A), np.append(p, [1]))
        Pw_dir = Pw_dir / np.linalg.norm(Pw_dir)
        Pw_transformed = np.matmul(R, Pw[:-1]) + T
        p_dot = np.dot(Pw_dir, Pw_transformed)
        Pw_normal = Pw_transformed - p_dot * Pw_dir
        err_3d += np.linalg.norm(Pw_normal)

    return (math.sqrt(err_2d) / (idx + 1), err_3d / (idx + 1))


def get_data_set(pkl_file):
    """Documentation for get_data_set.

    Read the pickle file and return a list of 5-tuples of the form
    (X, Y, Z, x, y)
    """
    with open(pkl_file, 'rb') as f:
        data_obj = pickle.load(f)
        data_set = data_obj.tuple_list
    return data_set


def get_camera_params_from_m(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADD DOCSTRING."""
    A = np.empty((3, 3))
    R = np.empty((3, 3))
    T = np.empty((3, 1))

    T[2, 0] = M[2, 3]
    R[2, :] = M[2, :3]

    m1, m2, m3 = M[0, :3], M[1, :3], M[2, :3]
    A[0, 2] = np.dot(m1, m3)
    A[1, 2] = np.dot(m2, m3)

    A[1, 1] = math.sqrt(np.dot(m2, m2)-A[1, 2]*A[1, 2])

    R[0, :] = np.cross(m2, m3)/A[1, 1]
    R[1, :] = (m2 - A[1, 2]*R[2, :])/A[1, 1]

    A[0, 0] = np.dot(m1, R[0, :])
    A[0, 1] = np.dot(m1, R[1, :])

    T[1, 0] = (M[1, 3] - A[1, 2]*T[2, 0])/A[1, 1]
    T[0, 0] = (M[0, 3] - A[0, 2]*T[2, 0] - A[0, 1]*T[1, 0])/A[0, 0]

    A[1, 0], A[2, 0], A[2, 1], A[2, 2] = 0, 0, 0, 1

    return A, R, T


def get_m_svd(data_set: list) -> np.ndarray:
    """Documentation for get_m_svd.

    Given a set of 3D to 2D point correspondences compute the camera matrix
    solving Cm = 0, |m| = 1.
    """
    def gen_row(point: tuple, idx: int, scl: float = 8.8 / 384) -> list:
        """Documentation for gen_row.

        Given a 5-tuple (X, Y, Z, x, y) return the co-efficients of the linear
        system used to solve M. 'idx' determines the row returned.
        """
        x_or_y = -scl * point[3 + idx]
        co_efficients = []
        if (idx == 0):
            co_efficients.extend(point[0:3] + (1,))
            co_efficients.extend([0, 0, 0, 0])
        elif (idx == 1):
            co_efficients.extend([0, 0, 0, 0])
            co_efficients.extend(point[0:3] + (1,))
        co_efficients.extend([p * x_or_y for p in point[0:3]] + [x_or_y])
        return co_efficients

    # Build the co-efficient and constant matrix from the data set
    data_size = len(data_set) * 2
    C = np.zeros((data_size, 12))
    for idx, elem in enumerate(data_set):
        C[2 * idx] = gen_row(elem, 0)
        C[2 * idx + 1] = gen_row(elem, 1)
    # find the co-efficients using SVD decomposition
    _, _, vh = np.linalg.svd(C)
    m = (vh.T)[:, -1]
    M = m.reshape(3, 4)
    np.set_printoptions(precision=3, suppress=True)
    # normalize the last row
    scl = np.linalg.norm(M[2, :3])
    return M / scl


def get_m_dlt(data_set: list) -> np.ndarray:
    """Documentation for get_m_dlt.

    Given a set of 3D to 2D point correspondences compute the camera matrix
    using the Direct Linear Transform Method Cm = p.
    """
    def gen_row(point: tuple, idx: int, scl: float = 8.8 / 384) -> list:
        """Documentation for a function.

        Given a 5-tuple (X, Y, Z, x, y) return the co-efficients of the linear
        system used to solve M. 'idx' determines the row returned.
        """
        x_or_y = -scl * point[3 + idx]
        co_efficients = []
        if (idx == 0):
            co_efficients.extend(point[0:3] + (1,))
            co_efficients.extend([0, 0, 0, 0])
        elif (idx == 1):
            co_efficients.extend([0, 0, 0, 0])
            co_efficients.extend(point[0:3] + (1,))
        co_efficients.extend([p * x_or_y for p in point[0:3]])
        return co_efficients

    # Build the co-efficient and constant matrix from the data set
    data_size = len(data_set) * 2
    p = np.zeros(data_size)
    C = np.zeros((data_size, 11))
    for idx, elem in enumerate(data_set):
        C[2 * idx] = gen_row(elem, 0)
        C[2 * idx + 1] = gen_row(elem, 1)
        p[2 * idx] = elem[3]
        p[2 * idx + 1] = elem[4]
    # find the co-efficients using SVD decomposition
    # _, _, vh = np.linalg.svd(C)
    # m = (vh.T)[:, -1]

    # Compute the co-efficients of M using Moore-Penrose inverse
    m = np.matmul(np.linalg.pinv(C), p)
    m = np.append(m, [1])
    # reshape to 3x4 matrix
    M = m.reshape(3, 4)
    # normalize the last row
    scl = np.linalg.norm(M[2, :3])
    return M / scl


def compute_camera_params(data_set: list, get_m, disp: bool=True):
    """Documnetation for compute_camera_params.

    Given a data set of 5-tuples (Xw, Yw, Zw, x, y) and a function to compute the camera parameters
    report camera parameters M, A, R, T and 2D, 3D error.
    """
    M = get_m(data_set)
    np.set_printoptions(precision=3, suppress=True)
    if disp:
        print("M = \n", M, end="\n\n")
    A, R, T = get_camera_params_from_m(M)
    if disp:
        print("A = \n", A, end="\n\n")
        print("R = \n", R, end="\n\n")
        print("T = \n", T, end="\n\n")
    err_2d, err_3d = __test(M, A, R, T, data_set)
    if disp:
        print("2D Error: ",  err_2d)
        print("3D Error: ",  err_3d)

    return M, A, R, T


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 2:
        data_file = sys.argv[1:]
    else:
        raise TypeError(
            "Expected Format: python3 calibrate.py <data_file> <data_file>")
    data_set = get_data_set(data_file[0]) + get_data_set(data_file[1])

    compute_camera_params(data_set, get_m_dlt)
