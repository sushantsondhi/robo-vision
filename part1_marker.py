import sys
import cv2
import numpy as np
import pickle


# def mark(img: np.ndarray):
#     filtered = cv2.Canny(img, 25, 150)
#     return filtered
#
#
# if __name__ == "__main__":
#     img_name = "left.jpg"
#     if len(sys.argv) > 1:
#         img_name = sys.argv[1]
#     img = cv2.imread(img_name)
#     result = mark(img)
#     cv2.imshow("Result", result)
#     cv2.waitKey(0)

class Marker:
    def __init__(self):
        self.vertex_list=[]
        self.image_copy=None

    def manual_points_marker(self, img: np.ndarray, pickle_file_name:str):
        image_copy= img.copy()

        def select_faces(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.vertex_list.append((x,y))
                _ = cv2.circle(image_copy, (self.vertex_list[-1][0], self.vertex_list[-1][1]), 1, (255, 0, 0))
        cv2.namedWindow("Marking Points")
        cv2.setMouseCallback("Marking Points",select_faces)
        while(1):
            cv2.imshow("Marking Points",image_copy)
            k = cv2.waitKey(1)
            if k != -1:
                break
        self.image_copy = image_copy
        cv2.destroyWindow("Marking Points")
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(self,f)
        return self.vertex_list


class tuple_2D_points:
    def __init__(self):
        self.final_list=[]

    def merge(self, pickle1:str, pickle2:str, final_pickle:str):
        with open("left.pkl", 'rb') as f:
            mynewlist1 = pickle.load(f)
        with open("right.pkl", 'rb') as f:
            mynewlist2 = pickle.load(f)
        self.final_list=[p1 + p2 for p1, p2 in zip(mynewlist1.vertex_list, mynewlist2.vertex_list)]
        with open(final_pickle, 'wb') as f:
            pickle.dump(self,f)


# if __name__ == "__main__":
#   # img = cv2.imread("right.jpg")
#   # r_list = Marker().manual_points_marker(img, "right.pkl")
#   with open("right.pkl", 'rb') as f:
#       mynewlist = pickle.load(f)
#
#   print(mynewlist.vertex_list)
#   print(len(mynewlist.vertex_list))
#   cv2.imshow("final_image", mynewlist.image_copy)
#   cv2.waitKey(0)
#   # print(r_list)


tuple_2D_points().merge("left.pkl","right.pkl","part1.pkl")
with open("part1.pkl", 'rb') as f:
  mynewlist = pickle.load(f)
print(mynewlist.final_list)
print(len(mynewlist.final_list))