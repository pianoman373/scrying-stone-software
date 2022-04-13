import socket
import sys
import threading
import cv2
import numpy as np

import camera
import disparity
import undistort
import utils
import json

output_points = []
output_colors = []

def server_thread():
    h_name = socket.gethostname()
    h_ip = socket.gethostbyname(h_name)

    if len(sys.argv) > 1:
        h_ip = sys.argv[1]

    HOST, PORT = h_ip, 7777

    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("server running on {}:{}".format(HOST, PORT))

    server_address = (HOST, PORT)

    tcp_socket.bind(server_address)

    tcp_socket.listen(1)

    while True:
        print('waiting for connection...')
        connection, client_address = tcp_socket.accept()
        print('connection from', client_address)

        while True:
            try:
                print('waiting for data...')
                data = connection.recv(1024)
                print('received data:', data)

                j = json.dumps({
                    'length': len(output_points),
                    'points': output_points.astype(np.int32).flatten().tolist(),
                    'colors': output_colors.astype(np.int32).flatten().tolist()
                }, separators=(',', ':'))

                print(j)

                connection.sendall(bytes(j+'\n', 'utf-8'))
            except:
                break

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

def generate_pointcloud(disparity_map, color_map, K, D, Q):
    global output_points
    global output_colors

    h, w = disparity_map.shape[:2]
    # Load focal length.
    focal_length = 10.76

    # Perspective transformation matrix
    # This transformation matrix is from the openCV documentation, didn't seem to work for me.
    # Q = np.float32([[1, 0, 0, -w / 2.0],
    #                 [0, -1, 0, h / 2.0],
    #                 [0, 0, 0, -focal_length],
    #                 [0, 0, 1, 0]])

    K = intrinsic_from_fov(h, w, 90)
    K_inv = np.linalg.inv(K)

    # This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
    # Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
    Q2 = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, focal_length*0.05, 0],  # Focal length multiplication obtained experimentally.
                     [0, 0, 0, 1]])

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    # Get color points
    colors = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > 0
    # Mask colors and points.
    output_points = points_3D[mask_map]

    # for i in range(len(output_points)):
    #     point = output_points[i]
    #     output_points[i] = [point[0], point[1] + h, 2000 - point[2]]
    #     output_points[i] = K_inv[:3, :3] @ [point[0], point[1], 1] * (point[2])

    output_colors = colors[mask_map]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points)
    pcd.colors = o3d.utility.Vector3dVector(output_colors / 255.0)
    points = [
        [0, 0, 0],
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 100]
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd, line_set])

if __name__ == '__main__':
    cam0 = camera.CameraFeed(0)
    cam1 = camera.CameraFeed(1)

    #t1 = threading.Thread(target=server_thread, args=())
    #t1.start()

    disparity.reconstruct_init()

    K0 = np.load("camera_params/K0.npy")
    K1 = np.load("camera_params/K1.npy")
    D0 = np.load("camera_params/D0.npy")
    D1 = np.load("camera_params/D1.npy")
    R0 = np.load("camera_params/R0.npy")
    R1 = np.load("camera_params/R1.npy")
    Q = np.load("camera_params/Q.npy")

    while True:
        img0 = cam0.read(False)
        img1 = cam1.read(False)
        img0 = cv2.imread('calibration_images/old/left/image26.png')
        img1 = cv2.imread('calibration_images/old/right/image26.png')

        img0 = undistort.undistort_pinhole(img0, K0, D0, R0)
        img1 = undistort.undistort_pinhole(img1, K1, D1, R1)
        #
        # img0 = cv2.imread('im0.png')
        # img1 = cv2.imread('im1.png')

        cv2.imshow("cam0", img0)
        cv2.imshow("cam1", img1)

        #img0 = utils.downsample_image(img0, 1)
        #img1 = utils.downsample_image(img1, 1)

        disp = disparity.reconstruct(img0, img1)
        #
        #
        # disp = utils.downsample_image(disp, 2)
        # img0 = utils.downsample_image(img0, 2)

        disp_color = utils.colormap_depth(disp)
        cv2.imshow('disparity', disp_color)

        generate_pointcloud(disp, img0, K0, D0, Q)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break