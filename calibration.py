import cv2.fisheye
import numpy as np
import cv2 as cv
import glob

CHECKERBOARD = (6,9)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# checkerboard flags
checkerboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

#fisheye calibration flags
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./calibration_images/*')

if __name__ == "__main__":
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, checkerboard_flags)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("found chessboard")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print("chessboard not found")
    cv.destroyAllWindows()

    img = cv.imread('./calibration_images/image8.png')
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    N_OK = len(objpoints)
    K = np.zeros((3,3))
    D = np.zeros((4,1))
    rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]

    rms, _, _, _, _ = \
    cv.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    np.save("./camera_params/ret", ret)
    np.save("./camera_params/K", K)
    np.save("./camera_params/dist", D)
    np.save("./camera_params/rvecs", rvecs)
    np.save("./camera_params/tvecs", tvecs)

    h,  w = img.shape[:2]
    # undistort
    map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv.CV_16SC2)
    undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    cv.imshow("undistorted image", undistorted_img)

    while True:
        key = cv.waitKey(20)
        if key == 27: #exit on esc
            break

