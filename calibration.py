import cv2.fisheye
import numpy as np
import cv2 as cv
import glob
import undistort
import utils
import argparse

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

checkerboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK



# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space

imgpoints0 = [] # 2d points in image plane.
imgpoints1 = [] # 2d points in image plane.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the stereo calibration output folder')
    parser.add_argument('--images', type=str, required=True, help='Folder containing left and right images')
    parser.add_argument('--grid_size', type=tuple, required=False, default=(6, 0, 9), help='Checkerboard grid size')
    parser.add_argument('--square_size', type=float, required=False, default=76.4, help='Checkerboard square size (mm)')

    args = parser.parse_args()

    CHECKERBOARD = (int(args.grid_size[0]), int(args.grid_size[2]))

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * 0.001 * args.square_size

    images0 = glob.glob(args.images + "/left/*.png")
    images1 = glob.glob(args.images + "/right/*.png")
    utils.sort_files(images0)
    utils.sort_files(images1)

    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame0', 400, 400)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    print("gathering points")

    for i in range(0, len(images0), 1):
        fname0 = images0[i]
        fname1 = images1[i]
        img0 = cv.imread(fname0)
        img1 = cv.imread(fname1)

        cv.imshow('frame0', img0)
        cv.imshow('frame1', img1)

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
            cv.waitKey(1)
        else:

            print("chessboard not found")

        cv2.waitKey(1)

    cv.destroyAllWindows()

    img0 = cv.imread(images0[0])
    img1 = cv.imread(images1[0])
    # img0 = utils.crop_sphere(img0, 350)
    # img1 = utils.crop_sphere(img1, 350)
    gray_image0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    gray_image1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    N_OK = len(objpoints)


    h, w = img0.shape[:2]


    print("computing intrinsics")

    ret, K0, D0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints, imgpoints0, (w, h), None, None)

    ret, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, (w, h), None, None)

    print("K0: ", K0)
    print("K1: ", K1)
    print("D0: ", D0)
    print("D1: ", D1)

    np.save(args.output_folder+"/K0.npy", K0)
    np.save(args.output_folder+"/K1.npy", K1)
    np.save(args.output_folder+"/D0.npy", D0)
    np.save(args.output_folder+"/D1.npy", D1)

    print("performing stereo calibration...")
    ret, K0, D0, K1, D1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints0, imgpoints1, K0, D0, K1, D1, (w, h), criteria, cv.CALIB_FIX_INTRINSIC)

    print("R: ", R)
    print("T: ", T)
    print("E: ", E)
    print("F: ", F)

    print("performing stereo rectification...")

    R0, R1, P0, P1, Q, roi0, roi1 = cv.stereoRectify(K0, D0, K1, D1, (w, h), R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    np.save(args.output_foler+"/R0.npy", R0)
    np.save(args.output_foler+"/R1.npy", R1)

    np.save(args.output_foler+"/P0.npy", P0)
    np.save(args.output_foler+"/P1.npy", P1)

    np.save(args.output_foler+"/Q.npy", Q)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv2.CV_32FC1)
    left_rectified = cv2.remap(img0, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    right_rectified = cv2.remap(img1, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cv2.imwrite("undistorted_left.png", left_rectified)
    cv2.imwrite("undistorted_right.png", right_rectified)

    cv2.namedWindow('frame0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame0', 400, 400)

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    print("complete!")

    undistorted_img0 = utils.draw_stereo_lines(left_rectified)
    undistorted_img1 = utils.draw_stereo_lines(right_rectified)

    cv.imshow('frame0', undistorted_img0)
    cv.imshow('frame1', undistorted_img1)

    cv.imshow('original0', img0)
    cv.imshow('original1', img1)

    while True:
        key = cv.waitKey(20)
        if key == 27: #exit on esc
            break

