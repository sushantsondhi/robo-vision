import numpy as np
import cv2
import pickle

class Marker:
    def __init__(self):
        self.vertex_list = []
        self.image_copy=None


    def manual_face_detector(self,img: np.ndarray, pickle_file_name: str) -> list:
        img_copy = img.copy()

        def select_faces(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.vertex_list.append((x,y))
                _ = cv2.circle(img_copy, (self.vertex_list[-1][0], self.vertex_list[-1][1]), 1, (255, 0, 0))
                # if(len(vertex_list)%2 == 0):
                #     a = vertex_list[-2]
                #     b = vertex_list[-1]
                #     _ = cv2.rectangle(img_copy,(a[0],a[1]),(b[0],b[1]),(255,0,0),2)
                # else:
                #     _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
        cv2.namedWindow("Face Selection")
        cv2.setMouseCallback("Face Selection",select_faces)
        while(1):
            cv2.imshow("Face Selection",img_copy)
            k = cv2.waitKey(1)
            if k == 122 and len(self.vertex_list) != 0:
                self.vertex_list.pop()
                img_copy = img.copy()
                # for x,y in zip(vertex_list[0::2],vertex_list[1::2]):
                #     _ = cv2.rectangle(img_copy,x,y,(255,0,0),2)
                for a in self.vertex_list:
                    _ = cv2.circle(img_copy,(a[0],a[1]),1,(255,0,0))
                continue
            if k != -1:
                break
        self.image_copy = img_copy
        cv2.destroyWindow("Face Selection")
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(self,f)
        # rectangle_list = [(x[0],x[1],y[0]-x[0],y[1]-x[1]) for x,y in zip(vertex_list[0::2],vertex_list[1::2])]
        return self.vertex_list

class List_5_tuple:
    def __init__(self):
        self.tuple_list= []
        self.image_copy= None

    def save_list(self, tuple_list, image_copy, pickle_file_name:str):
        self.tuple_list = tuple_list
        self.image_copy = image_copy
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(self,f)

# def manual_face_detector( img: np.ndarray, pickle_file_name: str) -> list:
#     img_copy = img.copy()
#     vertex_list=[]
#     def select_faces(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             vertex_list.append((x, y))
#             _ = cv2.circle(img_copy, (vertex_list[-1][0], vertex_list[-1][1]), 1, (255, 0, 0))
#             # if(len(vertex_list)%2 == 0):
#             #     a = vertex_list[-2]
#             #     b = vertex_list[-1]
#             #     _ = cv2.rectangle(img_copy,(a[0],a[1]),(b[0],b[1]),(255,0,0),2)
#             # else:
#             #     _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
#
#     cv2.namedWindow("Face Selection")
#     cv2.setMouseCallback("Face Selection", select_faces)
#     while (1):
#         cv2.imshow("Face Selection", img_copy)
#         k = cv2.waitKey(1)
#         if k == 122 and len(vertex_list) != 0:
#             vertex_list.pop()
#             img_copy = img.copy()
#             # for x,y in zip(vertex_list[0::2],vertex_list[1::2]):
#             #     _ = cv2.rectangle(img_copy,x,y,(255,0,0),2)
#             for a in vertex_list:
#                 _ = cv2.circle(img_copy, (a[0], a[1]), 1, (255, 0, 0))
#             continue
#         if k != -1:
#             break
#
#     cv2.destroyWindow("Face Selection")
#     with open(pickle_file_name, 'wb') as f:
#         pickle.dump(vertex_list, f)
#     # rectangle_list = [(x[0],x[1],y[0]-x[0],y[1]-x[1]) for x,y in zip(vertex_list[0::2],vertex_list[1::2])]
#     return vertex_list

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
            p2 = p1 + np.array([dim_x, 0.0, 0.0])
            p3 = p1 + np.array([dim_x, dim_y, 0.0])
            p4 = p1 + np.array([0.0, dim_y, 0.0])
            data_set.extend([p1, p2, p3, p4])

    return data_set

# if __name__ == "__main__":
#   # img = cv2.imread("./left2.16mm.jpg")
#   # r_list = Marker().manual_face_detector(img, "left2.16mm.pkl");
#   with open("left1.16mm.pkl", 'rb') as f:
#       mynewlist = pickle.load(f)
#
#   print(mynewlist.vertex_list)
#   print(len(mynewlist.vertex_list))
#   cv2.imshow("final_image", mynewlist.image_copy)
#   cv2.waitKey(0)
#   # print(r_list)


# if __name__ == "__main__":
#   with open("left2.16mm.pkl", 'rb') as f:
#       mynewlist = pickle.load(f)
#   list_pickle= mynewlist.vertex_list
#   img= mynewlist.image_copy
#   data_3d = gen_data_set(np.array([500, 0, 200]))
#   data_set = []
#   for p_3d, p_2d in zip(data_3d, list_pickle):
#       data_set.append(tuple(p_3d) + tuple(p_2d))
#
#   List_5_tuple().save_list(data_set, img, "left2.16mm_1.pkl")

# if __name__ == "__main__":
#   with open("left1.16mm_1.pkl", 'rb') as f:
#       mynewlist = pickle.load(f)
#
#   print(mynewlist.tuple_list)
#   print(len(mynewlist.tuple_list))
#   cv2.imshow("final_image", mynewlist.image_copy)
#   cv2.waitKey(0)
