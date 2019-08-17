import numpy as np
import cv2

def manual_face_detector(img: np.ndarray) -> list:
    img_copy = img.copy()
    vertex_list = []
    def select_faces(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            vertex_list.append((x,y))
            if(len(vertex_list)%2 == 0):
                a = vertex_list[-2]
                b = vertex_list[-1]
                _ = cv2.rectangle(img_copy,(a[0],a[1]),(b[0],b[1]),(255,0,0),2)
            else:
                _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
    cv2.namedWindow("Face Selection")
    cv2.setMouseCallback("Face Selection",select_faces)
    while(1):
        cv2.imshow("Face Selection",img_copy)
        k = cv2.waitKey(1)
        if k == 122 and len(vertex_list) != 0:
            vertex_list.pop()
            img_copy = img.copy()
            for x,y in zip(vertex_list[0::2],vertex_list[1::2]):
                _ = cv2.rectangle(img_copy,x,y,(255,0,0),2)
            if len(vertex_list)%2 == 1:
                _ = cv2.circle(img_copy,(vertex_list[-1][0],vertex_list[-1][1]),1,(255,0,0))
            continue
        if k != -1:
            break
    cv2.destroyWindow("Face Selection")
    rectangle_list = [(x[0],x[1],y[0]-x[0],y[1]-x[1]) for x,y in zip(vertex_list[0::2],vertex_list[1::2])]
    return rectangle_list

if __name__ == "__main__":
  img = cv2.imread("../Input/Calibration/left1.16mm.jpg")
  r_list = manual_face_detector(img);
  print(r_list)

