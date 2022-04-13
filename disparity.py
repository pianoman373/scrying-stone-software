import cv2.fisheye
import numpy as np
import cv2
import utils
import camera
import undistort

def nothing(x):
    pass

def reconstruct_init():
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 200)

    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disparity', 600, 600)

    cv2.createTrackbar('minDisparity', 'disp', 0, 17, nothing)
    cv2.createTrackbar('maxDisparity', 'disp', 10, 100, nothing)
    cv2.createTrackbar('blockSize', 'disp', 3, 11, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 18, 20, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 100, 200, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 100, 200, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 200, 600, nothing)
    cv2.createTrackbar('P1', 'disp', 8, 50, nothing)
    cv2.createTrackbar('P2', 'disp', 32, 100, nothing)

def reconstruct(img_1, img_2):
    # =========================================================
    # Stereo 3D reconstruction
    # =========================================================

    #convert images to grayscale
    img1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # Downsample each image 3 times (because they're too big)
    img_1_downsampled = img1_gray#downsample_image(img1_gray, 1)
    img_2_downsampled = img2_gray#downsample_image(img2_gray, 1)

    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')*16
    maxDisparity = cv2.getTrackbarPos('maxDisparity', 'disp')*16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    P1 = cv2.getTrackbarPos('P1', 'disp')
    P2 = cv2.getTrackbarPos('P2', 'disp')

    # Note: disparity range is tuned according to specific parameters obtained through trial and error.
    min_disp = -minDisparity
    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=maxDisparity,
                                   blockSize=blockSize,
                                   uniquenessRatio=uniquenessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange,
                                   disp12MaxDiff=disp12MaxDiff,
                                   P1=P1*blockSize*blockSize,  # 8*3*win_size**2,
                                   P2=P2*blockSize*blockSize)  # 32*3*win_size**2)


    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)
    disparity_map_right = stereo.compute(img_2_downsampled, img_1_downsampled)
    #
    # filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    #
    # filtered = filter.filter(disparity_map, img_1_downsampled, img_2_downsampled, disparity_map)
    #
    #
    # confidence = filter.getConfidenceMap()
    #
    # cv2.imshow('confidence', confidence)

    return disparity_map

if __name__ == "__main__":
    cam0 = camera.CameraFeed(0)
    cam1 = camera.CameraFeed(1)
    frame = cv2.imread("parallel.png")

    reconstruct_init()

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame2', 400, 400)

    while True:
        frame1 = cam0.read(False)
        frame2 = cam1.read(False)

        frame1 = cv2.imread("calibration_images/left/image0.png")
        frame2 = cv2.imread("calibration_images/right/image0.png")

        K0 = np.load("camera_params/K0.npy")
        K1 = np.load("camera_params/K1.npy")
        D0 = np.load("camera_params/D0.npy")
        D1 = np.load("camera_params/D1.npy")


        frame1 = undistort.undistort_fisheye(frame1, K0, D0)
        frame2 = undistort.undistort_fisheye(frame2, K1, D1)

        frame1 = utils.downsample_image(frame1, 1)
        frame2 = utils.downsample_image(frame2, 1)

        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)

        disp = reconstruct(frame1, frame2)

        disp_colored = utils.colormap_depth(disp)

        cv2.imshow("disparity", disp_colored)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break