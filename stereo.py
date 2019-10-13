import sys
import cv2
import math
import pickle
import numpy as np
import pygame
import pygame.locals as pl
import OpenGL.GL as gl
import OpenGL.GLU as glu
from typing import Tuple, List
from part1_marker import tuple_2D_points


def get_data_set(file_name: str) -> list:
    """Documentation for get_data_set.

    Get point correspondences from stereo images in (x, y, x', y') from
    pickle file
    """
    with open(file_name, "rb") as df:
        data_obj = pickle.load(df)
        point_list = data_obj.final_list
    return point_list


def get_all_matrices(
    u: int, v: int, s: int, f: int,
    Tx: int, Ty: int, Tz: int,
    Rx: int, Ry: int, Rz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Documentation for get_all_matrices.

    Build M from values and decompose to A, R, T
    """
    def M(a, r, t): return np.matmul(a, np.hstack((r, t)))

    T = np.array([[Tx], [Ty], [Tz]])
    A = np.array([[f, s, u], [0, f, v], [0, 0, 1]])

    sx, cx = math.sin(Rx), math.cos(Rx)
    sy, cy = math.sin(Ry), math.cos(Ry)
    sz, cz = math.sin(Rz), math.cos(Rz)

    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = np.matmul(R_z, np.matmul(R_y, R_x))

    return M(A, R, T), A, R, T


def find_F_from_parameters(
    Al: np.ndarray, Ar: np.ndarray,
    Rl: np.ndarray, Rr: np.ndarray,
    Tl: np.ndarray, Tr: np.ndarray
) -> np.ndarray:
    """Documentation for find_F_from_parameters.

    Calculate F from camera parameters A, R, T
    """
    R = np.matmul(Rr, np.linalg.inv(Rl))
    T = Tr - np.matmul(R, Tl)
    x, y, z = T[0, 0], T[1, 0], T[2, 0]
    T_cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    E = np.matmul(T_cross, R)
    F = np.matmul(np.matmul(np.linalg.inv(Ar).T, E), np.linalg.inv(Al))
    return F


def find_F_from_points(lis: list) -> np.ndarray:
    """Documentation for find_F_from_points.

    Given a set of 2d point correspondences (x, y, x', y') find the
    Fundamental matrix.
    """
    def get_part1_2d_points(lis: list) -> np.ndarray:
        """ Documentation for get_part1_2d_points.

        Generate the co-efficient matrix to solve for F
        """
        A = np.zeros((len(lis), 9))

        for k in range(len(lis)):
            i = lis[k]
            A[k] = np.array([
                i[0] * i[2], i[1] * i[2], i[2],
                i[0] * i[3], i[1] * i[3], i[3],
                i[0], i[1], 1]
            )
        return A

    A = get_part1_2d_points(lis)
    _, _, vh = np.linalg.svd(A)
    return (vh.T)[:, -1].reshape((3, 3))


def draw_epipolar_lines(
    point_list: list, img: np.ndarray,
    Al: np.ndarray, Rl: np.ndarray, Tl: np.ndarray,
    Mr: np.ndarray, l=3, disp=False
) -> np.ndarray:
    """Documenatation for draw_epipolar_lines.

    Given the camera parameters for and 2D point correspondences draw the
    epipolar lines in the right image.
    """
    def __dir(p: np.ndarray):
        RtAp = np.vstack(
            (np.matmul(Rl.T, np.matmul(np.linalg.inv(Al), p)), [1]))
        lA = np.matmul(Mr, RtAp)
        return lA / lA[-1, 0]
    lB = -np.matmul(Mr, np.vstack((np.matmul(Rl.T, Tl), [1])))
    B = lB / lB[-1, 0]

    for (x0, y0, x1, y1) in point_list:
        p = np.array([[x0], [y0], [1]])
        P = l * __dir(p) + B
        _ = cv2.line(
            img, (x1, y1), (int(P[0, 0]), int(P[1, 0])), (255, 0, 0), 1, cv2.LINE_AA)

    if disp:
        cv2.imshow("epipolar", img)
        cv2.waitKey(0)
    return img


def get_3d_points(
    point_list: list, Al: np.ndarray, Ar: np.ndarray,
    Rl: np.ndarray, Rr: np.ndarray, Tl: np.ndarray, Tr: np.ndarray,
) -> List[np.ndarray]:
    """Documentation for get_3d_points.

    Given 2D stereo point correspondences as (x, y, x', y') find the
    corresponding 3D points.
    """
    def calcPw_stupid(
        Al: np.ndarray, Ar: np.ndarray, Rl: np.ndarray, Rr: np.ndarray,
        Tl: np.ndarray, Tr: np.ndarray, pl: np.ndarray, pr: np.ndarray
    ) -> np.ndarray:
        """Documentation for calcPw.

        Calculate 3D points by finding the closest points on skew lines
        generated using point correspondences in stereo images.
        """
        # parameters of both the lines
        dir0 = np.matmul(Rl.T, np.matmul(-np.linalg.inv(Al), pl))
        org0 = np.matmul(Rl.T, Tl).reshape((3, 1))
        dir1 = np.matmul(Rr.T, np.matmul(np.linalg.inv(Ar), pr))
        org1 = np.matmul(Rr.T, Tr).reshape((3, 1))
        # solve the equations to find the points on lines
        normal_dir = np.cross(dir0.T, dir1.T).T
        B = org0 - org1
        C = np.hstack((-dir0, dir1, normal_dir)).reshape((3, 3)).T
        L = np.matmul(np.linalg.inv(C), B)

        return ((L[0, 0] * dir0 + org0) + (L[1, 0] * dir1 + org1)) / 2

    def calcPw(
        Al: np.ndarray, Ar: np.ndarray, Rl: np.ndarray, Rr: np.ndarray,
        Tl: np.ndarray, Tr: np.ndarray, pl: np.ndarray, pr: np.ndarray
    ) -> np.ndarray:
        """Documentation for calcPw.

        Generate 3D points by finding the intersection between the lines
        generated using point correspondences in stereo images
        """
        dir0 = np.matmul(Rl, np.matmul(np.linalg.inv(Al), pl))
        dir1 = np.matmul(Rr, np.matmul(np.linalg.inv(Ar), pr))
        org0, org1 = Tl, Tr
        C = np.hstack((dir0, -dir1))
        lam = np.matmul(np.linalg.pinv(C[:-1, :]), (org1 - org0)[:-1, :])
        return ((lam[0, 0] * dir0 + org0) + (lam[1, 0] * dir1 + org1)) / 2

    def get_3d(a, b): return calcPw_stupid(Al, Ar, Rl, Rr, Tl, Tr, a, b)
    points_3d = []
    # for all points
    for (x0, y0, x1, y1) in point_list:
        p0 = np.array([[x0], [y0], [1]])
        p1 = np.array([[x1], [y1], [1]])
        points_3d.append(get_3d(p0, p1))

    return points_3d


def draw_3d_points(point_list: list, disp: tuple = (1280, 720)):
    """Documentation for draw_3d_points.

    Given a set of 3D points draw and display the points.
    """
    def drawQuad(verts: list, edges: list):
        gl.glBegin(gl.GL_LINES)
        for (a, b) in edges:
            gl.glVertex3fv(verts[a])
            gl.glVertex3fv(verts[b])
        gl.glEnd()

    pygame.init()
    pygame.display.set_mode(disp, pl.DOUBLEBUF | pl.OPENGL)
    # world to model transforms
    glu.gluPerspective(45, (disp[0] / disp[1]), 0.1, 100)
    # gl.glTranslatef(0.0, -0.05, -5.0)
    gl.glTranslatef(1.0, -0.0, -1.5)
    # information for rendering
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    plist = point_list
    verts = (plist[0], plist[1], plist[3], plist[2])

    # event loop
    while True:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    verts = (plist[0], plist[1], plist[3], plist[2])
                if event.key == pygame.K_2:
                    verts = (plist[2], plist[3], plist[5], plist[4])
                if event.key == pygame.K_3:
                    verts = (plist[4], plist[5], plist[9], plist[8])
                if event.key == pygame.K_4:
                    verts = (plist[0], plist[2], plist[7], plist[6])
                if event.key == pygame.K_5:
                    verts = (plist[2], plist[4], plist[8], plist[7])
                if event.key == pygame.K_6:
                    verts = (plist[10], plist[11], plist[13], plist[12])
                if event.key == pygame.K_7:
                    verts = (plist[14], plist[15], plist[16], plist[17])

        # clear screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # render objects
        drawQuad((plist[0], plist[1], plist[3], plist[2]), edges)
        drawQuad((plist[2], plist[3], plist[5], plist[4]), edges)
        drawQuad((plist[4], plist[5], plist[9], plist[8]), edges)
        drawQuad((plist[0], plist[2], plist[7], plist[6]), edges)
        drawQuad((plist[2], plist[4], plist[8], plist[7]), edges)
        drawQuad((plist[10], plist[11], plist[13], plist[12]), edges)
        drawQuad((plist[14], plist[15], plist[16], plist[17]), edges)

        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 3:
        data_file = sys.argv[1:]
    else:
        raise TypeError(
            "Expected Format: \
                python3 stereo.py <left_image> <right_image> <data_file>")

    img_lft = cv2.imread(sys.argv[1])
    img_rht = cv2.imread(sys.argv[2])
    # find camera parameters from data
    Ml, Al, Rl, Tl = get_all_matrices(
        186.12, 164.15, 1.01, 16.55,
        -621.06, -58.07, 984.56,
        0.15, 0.28, 0.02)
    Mr, Ar, Rr, Tr = get_all_matrices(
        193.89, 144.43, 1.01, 16.84,
        -659.19, -76.57, 1055.80,
        0.16, 0.36, 0.03)
    point_list = get_data_set(sys.argv[3])

    # numpy print options
    np.set_printoptions(precision=3, suppress=True)

    # # 1.1.1 Compute Fundamental matrices
    F1 = find_F_from_parameters(Al, Ar, Rl, Rr, Tl, Tr)
    F2 = find_F_from_points(point_list)
    print("F1 = \n", F1, end="\n\n")
    print("F2 = \n", F2, end="\n\n")

    # 1.1.2 Draw epipolar lines
    img_epi = draw_epipolar_lines(point_list, img_rht, Al, Rl, Tl, Mr)
    cv2.imwrite("./Output/epipolar.png", img_epi)

    # 1.2 Find 3D points from stereo
    def to_tuple(a: np.ndarray): return [a[0, 0], a[1, 0], a[2, 0]]

    points_3d = get_3d_points(point_list,  Al, Ar, Rl, Rr, Tl, Tr,)
    point_list_3d = [to_tuple(p / 1000) for p in points_3d]
    for idx, point in enumerate(points_3d):
        print(str(idx + 1) + ":", point.T)
    draw_3d_points(point_list_3d)
