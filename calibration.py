import cv2.fisheye
import numpy as np
import cv2 as cv
import glob
import undistort

CHECKERBOARD = (6,9)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# checkerboard flags
checkerboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

#fisheye calibration flags
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space

print(objp)

imgpoints0 = [] # 2d points in image plane.
imgpoints1 = [] # 2d points in image plane.
images0 = glob.glob('./calibration_images/left/*.png')
images1 = glob.glob('./calibration_images/right/*.png')

def full_stack():
    import traceback, sys
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]       # remove call of full_stack, the printed exception
                            # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
         stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr

if __name__ == "__main__":
    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame0', 400, 400)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    for i in range(len(images0)):
        fname0 = images0[i]
        fname1 = images1[i]
        img0 = cv.imread(fname0)
        img1 = cv.imread(fname1)

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

    img0 = cv.imread('./calibration_images/left/image0.png')
    img1 = cv.imread('./calibration_images/right/image0.png')
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

    rms, _, _, _, _ = cv.fisheye.calibrate(
        objpoints,
        imgpoints0,
        gray0.shape[::-1],
        K0,
        D0,
        rvecs0,
        tvecs0,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    rms, _, _, _, _ = cv.fisheye.calibrate(
        objpoints,
        imgpoints1,
        gray1.shape[::-1],
        K1,
        D1,
        rvecs1,
        tvecs1,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    h, w = img0.shape[:2]
    R = np.zeros((3, 3))
    T = np.zeros((3), dtype=np.float64)

    print(T)

    # rms, _, _, _, _ = cv.fisheye.stereoCalibrate(
    #     objpoints,
    #     imgpoints0,
    #     imgpoints1,
    #     K0,
    #     D0,
    #     K1,
    #     D1,
    #     (w,h),
    #     R,
    #     T,
    #     calibration_flags,
    #     (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    # )

    # undistort
    undistorted_img0 = undistort.undistort_fisheye(img0, K0, D0)
    undistorted_img1 = undistort.undistort_fisheye(img1, K1, D1)

    np.save("camera_params/K0.npy", K0)
    np.save("camera_params/K1.npy", K1)

    np.save("camera_params/D0.npy", D0)
    np.save("camera_params/D1.npy", D1)

    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame0', 400, 400)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    cv.imshow('frame0', undistorted_img0)
    cv.imshow('frame1', undistorted_img1)

    while True:
        key = cv.waitKey(20)
        if key == 27: #exit on esc
            break

