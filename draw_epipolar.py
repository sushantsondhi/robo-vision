import cv2
import numpy as np
import pickle

class draw_epipolar_lines:
    def find_epipole_in_Cr(self,Mr: np.ndarray, R_it:np.ndarray, T_it:np.ndarray):
        image_of_cl= np.matmul(Mr, T_it)
        return ((image_of_cl[0,0]/image_of_cl[3,0]),(image_of_cl[1,0]/image_of_cl[3,0]))

    def draw_epipolar(self,epipole, point):

