import cv2
import numpy as np
import threading
import time
import camera

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

    print("worker thread exited")
    computing = False

def back(*args):
    pass

# crop padding from opencv image
def crop_padding(image, left, right, top, bottom):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

def crop_bottom_half(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

if __name__ == '__main__':
    cam = camera.open_camera()
    index = 0
    lastSavedTime = time.time()

    while True:
        check, frame = cam.read()
        cv2.imshow('frame', frame)

        # Load image
        image = crop_bottom_half(frame)
        image = crop_padding(image, 150, 150, 150, 150)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite("calibration_images/image" + str(index) + ".png", image)
            index = index + 1

        # find chessboard corners
        if not computing:
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