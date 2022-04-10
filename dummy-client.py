import socket
import sys
import numpy as np

if __name__ == '__main__':
    HOST, PORT = "192.168.1.4", 7777
    data = "hello TCP"

    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        sock.sendall(bytes(data + "\n", "utf-8"))

        # Receive data from the server and shut down
        received = sock.recv(8192)
        print("received: {}".format(received, "utf-8"))
    #
    # print("Sent:     {}".format(data))
    # print("Received: {}".format(received))