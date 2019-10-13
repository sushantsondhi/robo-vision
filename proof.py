from marker import List_5_tuple
from stereo import get_3d_points
from calibrate import compute_camera_params
from calibrate import get_data_set, get_m_svd

if __name__ == "__main__":
    data_file1 = "./Input/left1.16mm.pkl"
    data_file2 = "./Input/left2.16mm.pkl"

    data_set = get_data_set(data_file1) + get_data_set(data_file2)
    M, A, R, T = compute_camera_params(data_set, get_m_svd, False)

    point_list = [
        (x0, y0, x1, y1)
        for (_, _, _, x0, y0), (_, _, _, x1, y1) in zip(data_set[:100], data_set[100:])
        ]
    get_3d_points(point_list, A, A, R, R, T, T)
    
