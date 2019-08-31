import numpy as np
from numpy.linalg import inv
from numpy.linalg import svd
import math
from typing import Tuple
from part1_marker import tuple_2D_points
import pickle
import cv2


def get_all_matrices(u: int, v: int, s: int, f: int, Tx: int, Ty: int, Tz: int, Rx: int, Ry: int, Rz: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    T = np.array([[Tx], [Ty], [Tz]])
    A = np.array([[f, s, u], [0, f, v], [0, 0, 1]])
    sx, cx = math.sin(Rx), math.cos(Rx)
    sy, cy = math.sin(Ry), math.cos(Ry)
    sz, cz = math.sin(Rz), math.cos(Rz)
    R_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    R = np.matmul(np.matmul(R_x, R_y), R_z)
    return A, R, T


def getMFromART(A: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    P = np.hstack((R, T))
    return np.matmul(A, P)


def findF(Al: np.ndarray, Ar: np.ndarray, Rl: np.ndarray, Rr: np.ndarray, Tl: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    R = np.matmul(Rr, inv(Rl))
    T = Tr - np.matmul(R, Tl)
    x, y, z = T[0, 0], T[1, 0], T[2, 0]
    T_cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    E = np.matmul(T_cross, R)
    F = np.matmul(np.matmul(inv(Ar).T, E), inv(Al))
    return F


# %%

def calcPw(Rl: np.ndarray, Rr: np.ndarray, pl: np.ndarray, pr: np.ndarray, Tl: np.ndarray,
           Tr: np.ndarray) -> np.ndarray:
    A = np.matmul(Rl, pl)
    B = np.matmul(Rr, pr)
    C = np.hstack((A, -B))
    lam = np.matmul(np.linalg.pinv(C), Tr - Tl)
    # print (lam[1,0]*np.matmul(Rr, pr) + Tr)
    return lam[0, 0] * np.matmul(Rl, pl) + Tl


# %%

def findF2(A: np.ndarray) -> np.ndarray:
    _, _, vh = svd(A)
    return (vh.T)[:, -1].reshape((3, 3))


# %%

def calcEpipolarLine(Mr: np.ndarray, R: np.ndarray, T: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scl = 384 / 8.8
    A = np.matmul(Mr, np.vstack((np.matmul(R, p), [1])))
    B = np.matmul(Mr, np.vstack((T, [1])))
    return (A).astype(int), (B ).astype(int)


def get_part1_2d_points(pickle_file_name: str):
    with open(pickle_file_name, "rb") as f:
        lis = pickle.load(f)
    lis = lis.final_list
    A = np.zeros((len(lis), 9))

    for k in range(len(lis)):
        i = lis[k]
        A[k] = np.array([i[0]*i[2],i[1]*i[2],i[2],i[0]*i[3],i[1]*i[3],i[3],i[0],i[1],1])
    print(A.shape)
    return A

def draw_epipolar(img, A:np.ndarray, B:np.ndarray):
    _= cv2.line(img, (int(B[0][0]), int(B[1,0])),(int(164.66*A[0][0]+B[0][0]), int(164.66*A[1][0]+B[1][0])),(255,0,0),5)
    print((int(B[0][0]), int(B[1,0])),(int(164.66*A[0][0]+164*B[0][0]), int(164.66*A[1][0]+164*B[1][0])))
    cv2.imshow("final",img)
    cv2.waitKey(0)


img=cv2.imread("right.jpg")
Ar, Rr, Tr= get_all_matrices(193.89675221, 144.43431051, 1.0116374294, 16.842326127,-659.19737229,-76.572279751, 1055.8014876, 0.16112722935, 0.36219027236, 0.026911763000)
Al, Rl, Tl= get_all_matrices(186.11619191, 164.15264850, 1.0166343583, 16.551086572, -621.06754176, -58.069551431, 984.55520522, 0.15540547317, 0.27888534145, 0.017528059127)
Ml = getMFromART(Al, Rl, Tl)
Mr= getMFromART(Ar, Rr, Tr)
R= np.matmul(Rr, Rl.T)
T= Tr- np.matmul(R, Tl)
with open("part1.pkl", "rb") as f:
    lis = pickle.load(f)
lis = lis.final_list
p_img= np.array([[lis[0][0]],[lis[0][1]],[1]])
p_world = np.matmul(np.linalg.pinv(Ml), p_img)
# A,B= calcEpipolarLine(Mr, R, T, p_world[:3] / p_world[3])
# draw_epipolar(img, A, B)
# ab= get_part1_2d_points("part1.pkl")
# print(findF2(ab))
p_dash= np.array([[lis[0][2]],[lis[0][3]],[1]])
# print(p_dash-B/A[0][0])
A= np.matmul(Mr, np.vstack((np.matmul(Rl.T,np.matmul(np.linalg.inv(Al),p_img)),[1])))
A= A/A[-1][0]
B= np.matmul(Mr, np.vstack((np.matmul(Rl.T, Tl),[1])))
B= B/B[-1][0]
_= cv2.line(img, tuple([int(i) for i in lis[0][2:]]), tuple([int(i) for i in B[:-1][0]]),(255,0,0),5)
# print((int(B[0][0]), int(B[1,0])),(int(164.66*A[0][0]+164*B[0][0]), int(164.66*A[1][0]+164*B[1][0])))
cv2.imshow("final",img)
cv2.waitKey(0)