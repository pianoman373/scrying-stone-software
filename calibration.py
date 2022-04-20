import cv2.fisheye
import numpy as np
import cv2 as cv
import glob
import undistort
import utils

CHECKERBOARD = (6,9)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

checkerboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS

calibration_flags = cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5

stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC

stereo_rectify_flags = 0


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space

imgpoints0 = [] # 2d points in image plane.
imgpoints1 = [] # 2d points in image plane.
images0 = glob.glob('./calibration_images/left/*.png')
images1 = glob.glob('./calibration_images/right/*.png')

if __name__ == "__main__":
    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame0', 400, 400)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    print("gathering points")

    for i in range(len(images0)):
        fname0 = images0[i]
        fname1 = images1[i]
        img0 = cv.imread(fname0)
        img1 = cv.imread(fname1)
        #
        # img0 = utils.crop_sphere(img0, 350)
        # img1 = utils.crop_sphere(img1, 350)

        gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret0, corners0 = cv.findChessboardCorners(gray0, CHECKERBOARD, checkerboard_flags)
        ret1, corners1 = cv.findChessboardCorners(gray1, CHECKERBOARD, checkerboard_flags)

        # If found, add object points, image points (after refining them)
        if ret0 == True and ret1 == True:
            print("found chessboard")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray0, corners0, (3,3), (-1,-1), criteria)
            corners3 = cv.cornerSubPix(gray1, corners1, (3, 3), (-1, -1), criteria)

            imgpoints0.append(corners2)
            imgpoints1.append(corners3)

            # Draw and display the corners
            cv.drawChessboardCorners(img0, CHECKERBOARD, corners2, ret0)
            cv.imshow('frame0', img0)

            cv.drawChessboardCorners(img1, CHECKERBOARD, corners3, ret1)
            cv.imshow('frame1', img1)
            cv.waitKey(500)
        else:
            print("chessboard not found")

    cv.destroyAllWindows()

    img0 = cv.imread(images0[0])
    img1 = cv.imread(images1[0])
    # img0 = utils.crop_sphere(img0, 350)
    # img1 = utils.crop_sphere(img1, 350)
    gray_image0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    gray_image1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    N_OK = len(objpoints)
    K0 = np.zeros((3,3))
    D0 = np.zeros((4,1))
    K1 = np.zeros((3,3))
    D1 = np.zeros((4,1))
    rvecs0 = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
    tvecs0 = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
    rvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    print("computing intrinsics")

    rms, _, _, _, _ = cv.calibrateCamera(
        objpoints,
        imgpoints0,
        gray0.shape[::-1],
        K0,
        D0,
        rvecs0,
        tvecs0,
        calibration_flags
    )

    rms, _, _, _, _ = cv.calibrateCamera(
        objpoints,
        imgpoints1,
        gray1.shape[::-1],
        K1,
        D1,
        rvecs1,
        tvecs1,
        calibration_flags
    )

    print("K0: ", K0)
    print("K1: ", K1)
    print("D0: ", K0)
    print("D1: ", K1)

    h, w = img0.shape[:2]
    R = np.zeros((3, 3))
    T = np.zeros((3), dtype=np.float64)
    E = np.zeros((3, 3))
    F = np.zeros((3, 3))

    print("performing stereo calibration...")
    rms, _, _, _, _, _, _, _, _ = cv.stereoCalibrate(
        objpoints,
        imgpoints0,
        imgpoints1,
        K0,
        D0,
        K1,
        D1,
        (w,h),
        R,
        T,
        E,
        F,
        stereo_calibration_flags
    )

    print("R: ", R)
    print("T: ", T)
    print("E: ", E)
    print("F: ", F)

    print("performing stereo rectification...")

    R0 = np.zeros((3, 3))
    R1 = np.zeros((3, 3))
    P0 = np.zeros((3, 3))
    P1 = np.zeros((3, 3))
    Q = np.zeros((4, 4))
    cv2.stereoRectify(
        K0,
        D0,
        K1,
        D1,
        (w,h),
        R,
        T,
        R0,
        R1,
        P0,
        P1,
        Q
    )

    np.save("camera_params/K0.npy", K0)
    np.save("camera_params/K1.npy", K1)

    np.save("camera_params/D0.npy", D0)
    np.save("camera_params/D1.npy", D1)

    np.save("camera_params/R0.npy", R0)
    np.save("camera_params/R1.npy", R1)

    np.save("camera_params/P0.npy", P0)
    np.save("camera_params/P1.npy", P1)

    np.save("camera_params/Q.npy", Q)

    undistorted_img0 = undistort.undistort_pinhole(img0, K0, D0, R)
    undistorted_img1 = undistort.undistort_pinhole(img1, K1, D1, R)

    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame0', 400, 400)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    print("complete!")

    cv.imshow('frame0', undistorted_img0)
    cv.imshow('frame1', undistorted_img1)

    cv.imshow('original0', img0)
    cv.imshow('original1', img1)

    while True:
        key = cv.waitKey(20)
        if key == 27: #exit on esc
            break

