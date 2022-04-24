import socket
import sys
import numpy as np
import json
import open3d as o3d
import cv2

if __name__ == '__main__':
    HOST, PORT = "localhost", 7777
    data = "hello TCP"

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

    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))

        while True:
            sock.sendall(bytes(data + "\n", "utf-8"))

            # Receive data from the server and shut down
            received = sock.recv(8192*1024)
            #print("received: {}".format(received, "utf-8"))

            header = np.frombuffer(received[:8], dtype=np.uint32)
            w = header[0]
            h = header[1]

            k_left = np.frombuffer(received[8:8+36], dtype=np.float32)
            k_left = k_left.reshape((3, 3))


            T_tot = np.frombuffer(received[8+36:8+36+64], dtype=np.float32)
            T_tot = T_tot.reshape((4, 4))


            color = np.frombuffer(received[8+36+64:8+36+64+w*h*3], dtype=np.uint8)
            color = color.reshape((h, w, 3))

            depth = np.frombuffer(received[8+36+64+w*h*3:8+36+64+w*h*3+w*h*4], dtype=np.uint32)
            depth = depth.astype(np.float32) / 1000.0
            depth = depth.reshape((h, w, 1))
            cv2.imshow("color", color)
            cv2.imshow("depth", depth*0.02)

            xs = T_tot[0, 3]
            ys = T_tot[1, 3]
            zs = T_tot[2, 3]
            position = np.array([xs, ys, zs])


            print(position)

            # append position to line set in open3d
            line_set.points.append([position[0], position[1], position[2]])
            line_set.colors.append([255, 0, 0])
            p = len(line_set.points)
            line_set.lines.append([p - 1, p - 2])

            # convert to rgbd image
            color_raw = o3d.geometry.Image(color)
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
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
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
            # vis.get_view_control().set_lookat(position)
            # vis.get_view_control().set_zoom(5.0)
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1000)
            if key == 27:  # exit on esc
                break