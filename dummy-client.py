import socket
import sys
import numpy as np
import json
import open3d as o3d
import cv2

if __name__ == '__main__':
    HOST, PORT = "localhost", 7777
    data = "hello TCP"



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

            color = np.frombuffer(received[8:8+w*h*3], dtype=np.uint8)
            color = color.reshape((h, w, 3))

            depth = np.frombuffer(received[8+w*h*3:8+w*h*3+w*h*4], dtype=np.uint32)
            depth = depth.astype(np.float32) / 1000.0
            depth = depth.reshape((h, w, 1))
            cv2.imshow("color", color)
            cv2.imshow("depth", depth*0.02)

            key = cv2.waitKey(1000)
            if key == 27:  # exit on esc
                break