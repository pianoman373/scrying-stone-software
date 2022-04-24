import argparse
import glob
import socket
import sys
import threading
import cv2
import numpy as np
import camera
import disparity
import odometry
import undistort
import utils
import json
import open3d as o3d

output_points = []
output_colors = []

color_image = None
depth_image = None

send_packet = False

def server_thread(h_ip):

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

                # h, w = color_image.shape[:2]
                #
                # print(color_image.astype(np.uint8).flatten())
                # color_bytes = color_image.astype(np.uint8).flatten().tobytes()
                # depth_bytes = (depth_image*1000).astype(np.uint32).flatten().tobytes()
                # header = np.array([w, h], dtype=np.uint32).tobytes()
                #
                # packet = header + color_bytes + depth_bytes
                #
                # print(w, h)
                # print(header)
                # connection.sendall(packet)

                j = json.dumps({
                    'length': len(output_points),
                    'points': (output_points * 1000).astype(np.int32).flatten().tolist(),
                    'colors': (output_colors * 255).astype(np.uint8).flatten().tolist()
                }, separators=(',', ':'))

                connection.sendall(bytes(j+'\n', 'utf-8'))
            except Exception as e:
                print(e)
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Odometry')
    parser.add_argument('--calibration', type=str, required=True, help='Folder containing calibration data')
    parser.add_argument('--mode', type=str, required=False,
                        help='Camera mode. Either gstreamer_fisheye, gstreamer, webcam, or webcam_offset')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset')
    parser.add_argument('--ip', type=str, required=True, help='IP Address to host on')
    args = parser.parse_args()

    t1 = threading.Thread(target=server_thread, args=(args.ip,))
    t1.start()

    P0 = np.load(args.calibration + "/P0.npy")
    P1 = np.load(args.calibration + "/P1.npy")
    K0 = np.load(args.calibration + "/K0.npy")
    K1 = np.load(args.calibration + "/K1.npy")
    D0 = np.load(args.calibration + "/D0.npy")
    D1 = np.load(args.calibration + "/D1.npy")
    R0 = np.load(args.calibration + "/R0.npy")
    R1 = np.load(args.calibration + "/R1.npy")

    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left = odometry.decompose_projection_matrix(P0)
    k_left_inv = np.linalg.inv(k_left)

    id = 1

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
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
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(line_set)
    vis.add_geometry(pcd)

    cam0 = None
    cam1 = None
    old_frame = None
    current_frame = None
    images0 = None
    images1 = None

    if args.dataset:
        images0 = glob.glob(args.dataset + "/left/*.png")
        images1 = glob.glob(args.dataset + "/right/*png")
        utils.sort_files(images0)
        utils.sort_files(images1)
        current_frame = cv2.imread(images0[0])
        current_frame = undistort.undistort_pinhole(current_frame, K0, D0, R0, P0)
    else:
        cam0 = camera.CameraFeed(0, args.mode)
        cam1 = camera.CameraFeed(1, args.mode)
        current_frame = cam0.read()
        current_frame = undistort.undistort_pinhole(current_frame, K0, D0, R0, P0)

    cv2.waitKey(10)

    T_tot = np.eye(4)

    position = np.array([0, 0, 0])

    while True:
        right_frame = None

        old_frame = current_frame.copy()

        if args.dataset:
            current_frame = cv2.imread(images0[id])
            right_frame = cv2.imread(images1[id])
        else:
            current_frame = cam0.read()
            right_frame = cam1.read()

        current_frame = undistort.undistort_pinhole(current_frame, K0, D0, R0, P0)
        right_frame = undistort.undistort_pinhole(right_frame, K1, D1, R1, P1)

        current_frame_orig = current_frame.copy()

        height, width = right_frame.shape[:2]

        T, depth = odometry.odometry(old_frame, current_frame, right_frame, P0, P1, k_left)

        old_T_tot = T_tot
        if id > 10:
            T_tot = T_tot.dot(T)

        old_position = position


        xs = T_tot[0, 3]
        ys = T_tot[1, 3]
        zs = T_tot[2, 3]
        position = np.array([xs, ys, zs])

        dist = np.linalg.norm(position - old_position)

        if dist > 0.25:
            #print("caught a jump")
            T_tot = old_T_tot
            position = old_position

        # append position to line set in open3d
        line_set.points.append([position[0], position[1], position[2]])
        line_set.colors.append([255, 0, 0])
        p = len(line_set.points)
        line_set.lines.append([p - 1, p - 2])

        color_image = current_frame.copy()
        depth_image = depth.copy()

        # convert to rgbd image
        color_raw = o3d.geometry.Image(current_frame)
        depth_raw = o3d.geometry.Image(depth.astype(np.float32))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw)

        # positions, colors = odometry.reproject(current_frame, depth, k_left_inv)
        # print(positions)

        # convert to point cloud
        cx = k_left[0, 2]
        cy = k_left[1, 2]
        fx = k_left[0, 0]
        fy = k_left[1, 1]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        new_pcd = pcd.create_from_rgbd_image(
            rgbd_image,
            intrinsic,
            np.eye(4),
            True
        )

        scaling_mat = np.array([[1000, 0, 0, 0], [0, -1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1]])
        new_pcd.transform(T_tot.dot(scaling_mat))

        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        output_points = np.asarray(new_pcd.points)
        output_colors = np.asarray(new_pcd.colors)

        vis.update_geometry(line_set)
        vis.update_geometry(pcd)

        vis.get_view_control().set_constant_z_near(0.1)
        vis.get_view_control().set_constant_z_far(500)
        #vis.get_view_control().set_lookat(position)
        #vis.get_view_control().set_zoom(5.0)
        vis.poll_events()
        vis.update_renderer()

        id += 1

        key = cv2.waitKey(1)
        if key == 27:  # exit on esc
            break