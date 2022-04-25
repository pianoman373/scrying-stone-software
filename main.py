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
import time

color_image = None
depth_image = None

send_packet = False

def server_thread(h_ip):

    HOST, PORT = h_ip, 7777

    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("server running on {}:{}".format(HOST, PORT))

    server_address = (HOST, PORT)

    tcp_socket.bind(server_address)

    tcp_socket.listen(10)

    while True:
        print('waiting for connection...')
        connection, client_address = tcp_socket.accept()
        print('connection from', client_address)

        while True:
            try:
                print('waiting for data...')
                #data = connection.recv(1024)
                #print('received data:', data)
                time.sleep(0.5)

                h, w = color_image.shape[:2]
                magic = 424242

                color_bytes = color_image.astype(np.uint8).flatten().tobytes()
                depth_bytes = (depth_image*1000).astype(np.uint32).flatten().tobytes()
                header = np.array([magic, w, h], dtype=np.uint32).tobytes()
                k_bytes = k_left.astype(np.float32).flatten().tobytes()
                T_bytes = T_tot.astype(np.float32).flatten().tobytes()


                packet = header + k_bytes + T_bytes + color_bytes + depth_bytes

                print("sent ", len(packet), " bytes")

                connection.sendall(packet)

                # j = json.dumps({
                #     'length': len(output_points),
                #     'points': (output_points * 1000).astype(np.int32).flatten().tolist(),
                #     'colors': (output_colors * 255).astype(np.uint8).flatten().tolist(),
                #     'data': packet.decode('utf-8')
                # }, separators=(',', ':'))
                #
                # connection.sendall(bytes(j+'\n', 'utf-8'))
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

    cam0 = None
    cam1 = None
    old_frame = None
    current_frame = None
    images0 = None
    images1 = None

    if args.dataset:
        images0 = glob.glob(args.dataset + "/left/*.png")
        images1 = glob.glob(args.dataset + "/right/*.png")
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
        if (np.shape(T) != (4, 4)):
            T = np.eye(4)

        old_T_tot = T_tot
        if id > 10:
            T_tot = T_tot.dot(T)

        old_position = position


        xs = T_tot[0, 3]
        ys = T_tot[1, 3]
        zs = T_tot[2, 3]
        position = np.array([xs, ys, zs])

        dist = np.linalg.norm(position - old_position)

        # if dist > 0.25:
        #     #print("caught a jump")
        #     T_tot = old_T_tot
        #     position = old_position


        color_image = current_frame.copy()
        depth_image = depth.copy()

        id += 1

        key = cv2.waitKey(1)
        if key == 27:  # exit on esc
            break
