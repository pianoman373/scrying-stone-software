import cv2
import numpy as np
import threading

chessboard_size = (6, 9)
corners_list = np.zeros(0)
computing = False

def find_corners(image):
    global corners_list
    global computing

    computing = True
    ret, corners = cv2.findChessboardCorners(image, chessboard_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        corners_list = corners
    else:
        corners_list = np.zeros(0)

    computing = False


if __name__ == '__main__':
    cam = cv2.VideoCapture(1)

    while True:
        check, frame = cam.read()

        # Load image
        image = frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # find chessboard corners


        if not computing:
            t1 = threading.Thread(target=find_corners, args=(gray_image,))
            t1.start()


        cv2.drawChessboardCorners(image, chessboard_size, corners_list, True)

        cv2.imshow('video', image)

        key = cv2.waitKey(1)
        if key == ord('a'):
            break

    cam.release()
    cv2.destroyAllWindows()