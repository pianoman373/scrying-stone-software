import cv2.fisheye
import numpy as np
import cv2
import camera

f = 129
cx = 388
cy = 388
k1 = 0.19
k2 = 0.19
k3 = 0
k4 = 0


def crop_left(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

def crop_right(img):
    cropped_img = img[0:img.shape[0], int(img.shape[1]/2):int(img.shape[1])]
    return cropped_img

# crop padding from opencv image
def crop_padding(image, left, right, top, bottom):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

def nothing(x):
    pass

def reconstruct_init():
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 200)

    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disparity', 600, 600)

    cv2.createTrackbar('minDisparity', 'disp', 0, 17, nothing)
    cv2.createTrackbar('maxDisparity', 'disp', 3, 100, nothing)
    cv2.createTrackbar('blockSize', 'disp', 3, 11, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 0, 20, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 100, 200, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 100, 200, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 200, 600, nothing)
    cv2.createTrackbar('P1', 'disp', 8, 50, nothing)
    cv2.createTrackbar('P2', 'disp', 32, 100, nothing)

# Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

def reconstruct(img_1, img_2):
    # =========================================================
    # Stereo 3D reconstruction
    # =========================================================
    # Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
    h, w = img_2.shape[:2]

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

    return disparity_map


def undistort(img):
    h, w = img.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    f = cv2.getTrackbarPos('f', 'options')
    cx = cv2.getTrackbarPos('cx', 'options')
    cy = cv2.getTrackbarPos('cy', 'options')

    k1 = cv2.getTrackbarPos('k1', 'options') * 0.01
    k2 = cv2.getTrackbarPos('k2', 'options') * 0.01
    k3 = cv2.getTrackbarPos('k3', 'options') * 0.01
    k4 = cv2.getTrackbarPos('k4', 'options') * 0.01

    K = np.array([[f, 0.0, cx],
                  [0.0, f, cy],
                  [0.0, 0.0, 1.0]])

    D = np.array([k1, k2, k3, k4], dtype=np.float32)

    # undistort
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    # undistorted_img = cv.fisheye.undistortImage(img, K, D)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img

if __name__ == "__main__":
    cam0 = camera.CameraFeed(0)
    cam1 = camera.CameraFeed(1)
    frame = cv2.imread("parallel.png")

    reconstruct_init()

    cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1', 400, 400)

    cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame2', 400, 400)

    cv2.namedWindow('options', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('options', 600, 400)

    cv2.createTrackbar('f', 'options', 341, 2000, nothing)
    cv2.createTrackbar('cx', 'options', 388, 1000, nothing)
    cv2.createTrackbar('cy', 'options', 388, 1000, nothing)

    cv2.createTrackbar('k1', 'options', 0, 100, nothing)
    cv2.createTrackbar('k2', 'options', 0, 100, nothing)
    cv2.createTrackbar('k3', 'options', 0, 100, nothing)
    cv2.createTrackbar('k4', 'options', 0, 100, nothing)

    while True:
        frame1 = cam0.read()
        frame2 = cam1.read()

        # frame1 = cv2.imread("im0.png")
        # frame1 = downsample_image(frame1, 2)
        # frame2 = cv2.imread("im1.png")
        # frame2 = downsample_image(frame2, 2)

        frame1 = undistort(frame1)
        frame2 = undistort(frame2)

        frame1 = downsample_image(frame1, 1)
        frame2 = downsample_image(frame2, 1)

        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)

        disp = reconstruct(frame1, frame2)
        disp = (disp * 16) - 16

        scaledDisparity = disp - np.min(disp)
        scaledDisparity = scaledDisparity * (255/np.max(scaledDisparity))
        scaledDisparity = scaledDisparity.astype(np.uint8)

        disp = scaledDisparity.astype(np.uint8)

        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        cv2.imshow("disparity", disp)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break