import cv2.fisheye
import numpy as np
import cv2


def undistort_fisheye(img, K, D):
    h, w = img.shape[:2]

    # undistort
    #new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    # undistorted_img = cv.fisheye.undistortImage(img, K, D)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img

def nothing(x):
    pass

if __name__ == "__main__":
    cv2.namedWindow('options', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('options', 600, 600)

    cv2.createTrackbar('fx', 'options', 242, 1000, nothing)
    cv2.createTrackbar('fy', 'options', 313, 1000, nothing)
    cv2.createTrackbar('cx', 'options', 517, 1000, nothing)
    cv2.createTrackbar('cy', 'options', 541, 1000, nothing)

    cv2.createTrackbar('k1', 'options', 0, 100, nothing)
    cv2.createTrackbar('k2', 'options', 0, 100, nothing)
    cv2.createTrackbar('k3', 'options', 0, 100, nothing)
    cv2.createTrackbar('k4', 'options', 0, 100, nothing)

    cv2.createTrackbar('balance', 'options', 0, 100, nothing)

    while True:
        fx = cv2.getTrackbarPos('fx', 'options')
        fy = cv2.getTrackbarPos('fy', 'options')
        cx = cv2.getTrackbarPos('cx', 'options')
        cy = cv2.getTrackbarPos('cy', 'options')

        k1 = cv2.getTrackbarPos('k1', 'options')*0.001
        k2 = cv2.getTrackbarPos('k2', 'options')*0.001
        k3 = cv2.getTrackbarPos('k3', 'options')*0.001
        k4 = cv2.getTrackbarPos('k4', 'options')*0.001

        k1 = 0.080
        k2 = -0.048
        k3 = 0.0071
        k4 = -0.00034

        balance = cv2.getTrackbarPos('balance', 'options')*0.01

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])

        D = np.array([k1, k2, k3, k4], dtype=np.float32)

        img = cv2.imread('./calibration_images/left/image0.png')

        h, w = img.shape[:2]
        # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        undistorted_img = undistort_fisheye(img, K, D)
        cv2.imshow("undistorted image", undistorted_img)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break