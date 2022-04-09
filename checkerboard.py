import cv2
import numpy as np
import threading
import time
import camera

chessboard_size = (6, 9)
corners_list0 = np.zeros(0)
corners_list1 = np.zeros(0)
found_corners = False
computing = False

def find_corners(image0, image1):
    global corners_list0
    global corners_list1
    global computing
    global found_corners

    computing = True
    ret0, corners0 = cv2.findChessboardCorners(image0, chessboard_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    ret1, corners1 = cv2.findChessboardCorners(image1, chessboard_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    found_corners = ret0 or ret1

    if found_corners == True:
        corners_list0 = corners0
        corners_list1 = corners1
    else:
        corners_list0 = np.zeros(0)
        corners_list1 = np.zeros(0)

    print("worker thread exited")
    computing = False

def back(*args):
    pass

if __name__ == '__main__':
    cam0 = camera.CameraFeed(0)
    cam1 = camera.CameraFeed(1)
    index = 0

    while True:
        frame0 = cam0.read(False)
        frame1 = cam1.read(False)

        gray_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite("calibration_images/left/image" + str(index) + ".png", frame0)
            cv2.imwrite("calibration_images/right/image" + str(index) + ".png", frame1)
            index = index + 1

        # find chessboard corners
        if not computing:
            t1 = threading.Thread(target=find_corners, args=(gray_frame0,gray_frame1,))
            t1.start()

        cv2.drawChessboardCorners(frame0, chessboard_size, corners_list0, True)
        cv2.drawChessboardCorners(frame1, chessboard_size, corners_list1, True)

        cv2.putText(frame0, "wrote " + str(index) + "images",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

        cv2.imshow('left', frame0)
        cv2.imshow('right', frame1)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break

    cv2.destroyAllWindows()