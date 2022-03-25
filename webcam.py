import cv2
import numpy as np
import threading
import time

def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=1432,
    display_height=1080,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

chessboard_size = (6, 9)
corners_list = np.zeros(0)
found_corners = False
computing = False

def find_corners(image):
    global corners_list
    global computing
    global found_corners

    computing = True
    ret, corners = cv2.findChessboardCorners(image, chessboard_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    found_corners = ret

    if ret == True:
        corners_list = corners
    else:
        corners_list = np.zeros(0)

    computing = False

def back(*args):
    pass

def crop_bottom_half(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

if __name__ == '__main__':
    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    #cam = cv2.VideoCapture(0)
    index = 0
    lastSavedTime = time.time()

    while True:
        check, frame = cam.read()

        size = frame.shape

        # Load image
        image = crop_bottom_half(frame)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # find chessboard corners


        if not computing:
            currentTime = time.time()

            if currentTime - lastSavedTime > 3:
                cv2.imwrite("calibration_images/image" + str(index) + ".png", image)
                index = index + 1
                lastSavedTime = currentTime

                t1 = threading.Thread(target=find_corners, args=(gray_image,))
                t1.start()


        cv2.drawChessboardCorners(image, chessboard_size, corners_list, True)
        cv2.putText(image, "wrote " + str(index) + "images",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

        cv2.imshow('video', image)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break

    cam.release()
    cv2.destroyAllWindows()