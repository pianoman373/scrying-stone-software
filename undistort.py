import cv2.fisheye
import numpy as np
import cv2

f = 248.28171122
cx = 346.5
cy = 349.5

def nothing(x):
    pass

if __name__ == "__main__":
    cv2.namedWindow('options', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('options', 600, 600)

    cv2.createTrackbar('f', 'options', 341, 1000, nothing)
    cv2.createTrackbar('cx', 'options', 388, 500, nothing)
    cv2.createTrackbar('cy', 'options', 388, 500, nothing)

    cv2.createTrackbar('k1', 'options', 0, 100, nothing)
    cv2.createTrackbar('k2', 'options', 0, 100, nothing)
    cv2.createTrackbar('k3', 'options', 0, 100, nothing)
    cv2.createTrackbar('k4', 'options', 0, 100, nothing)

    cv2.createTrackbar('balance', 'options', 0, 100, nothing)

    while True:
        f = cv2.getTrackbarPos('f', 'options')
        cx = cv2.getTrackbarPos('cx', 'options')
        cy = cv2.getTrackbarPos('cy', 'options')

        k1 = cv2.getTrackbarPos('k1', 'options')*0.01
        k2 = cv2.getTrackbarPos('k2', 'options')*0.01
        k3 = cv2.getTrackbarPos('k3', 'options')*0.01
        k4 = cv2.getTrackbarPos('k4', 'options')*0.01

        balance = cv2.getTrackbarPos('balance', 'options')*0.01

        K = np.array([[f, 0.0, cx],
                      [0.0, f, cy],
                      [0.0, 0.0, 1.0]])

        D = np.array([k1, k2, k3, k4], dtype=np.float32)

        img = cv2.imread('./calibration_images/image8.png')

        h, w = img.shape[:2]
        # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        # undistorted_img = cv.fisheye.undistortImage(img, K, D)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow("undistorted image", undistorted_img)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break