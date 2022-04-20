import cv2
import numpy as np

import camera
import glob

lk_params=dict(winSize  = (161,161), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1130, 0.01))


images0 = glob.glob('./Movement2/*.jpg')
images1 = glob.glob('./MovementTest/right/*.png')

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

images0.sort(key=natural_keys)

def detect(detector, img):
    p0 = detector.detect(img)

    return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

def get_mono_coordinates(t):
    # We multiply by the diagonal matrix to fix our vector
    # onto same coordinate axis as true values
    diag = np.array([[-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])
    adj_coord = np.matmul(diag, t)

    return adj_coord.flatten()

if __name__ == "__main__":
    cam = camera.CameraFeed(0)

    detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

    current_frame = cam.read(False)
    n_features = 0
    p0 = np.array([], dtype=np.float32)
    p1 = np.array([], dtype=np.float32)
    good_old = np.array([], dtype=np.float32)
    good_new = np.array([], dtype=np.float32)
    id = 1

    focal = 0.8560
    pp = (int(current_frame.shape[0]/2), int(current_frame.shape[1]/2))

    R = np.zeros(shape=(3, 3))
    t = np.zeros(shape=(3, 3))

    K = np.load("camera_params/K0.npy")
    D = np.load("camera_params/D0.npy")

    cv2.namedWindow('old frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('old frame', 400, 400)

    cv2.namedWindow('new frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('new frame', 400, 400)

    while True:
        old_frame = cv2.imread(images0[id-1])
        current_frame = cv2.imread(images0[id])

        old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if n_features < 2000:
            print("detecting features")
            p0 = detect(detector, old_frame_gray)



        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, current_frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_old = p0[st == 1]
            good_new = p1[st == 1]

        # if len(good_new) <= 0:
        #     continue
        #
        # if id < 2:
        #     E, _ = cv2.findEssentialMat(good_new, good_old, focal, pp, cv2.RANSAC, 0.999, 1.0,
        #                                 None)
        #
        #     if E is not None:
        #         print(R)
        #
        #         cv2.recoverPose(good_old, good_new, K, D, K, D, None, R, t)
        #         # _, R, t, _ = cv2.recoverPose(E, good_old, good_new, R, t, focal,
        #         #                                        pp, None)
        #
        # else:
        #     E, _ = cv2.findEssentialMat(good_new, good_old, focal, pp, cv2.RANSAC, 0.999, 1.0,
        #                                 None)
        #
        #     if E is not None and E.shape == (3, 3):
        #         _, _, R_new, t_new, _ = cv2.recoverPose(good_old, good_new, K, D, K, D, None, R.copy(), t.copy())
        #
        #         print(t_new)
        #
        #         absolute_scale = 1.0
        #         if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
        #             t = t + absolute_scale * R.dot(t_new)
        #             R = R_new.dot(R)

        id += 1

        n_features = good_new.shape[0]

        #draw keypoints
        # kp = []
        # for i in good_new:
        #     kp.append(cv2.KeyPoint(x=i[0], y=i[1], size=0, angle=0, response=0, octave=0, class_id=0))

        print('position:', get_mono_coordinates(t))

        kp = cv2.KeyPoint_convert(good_old)
        old_out_frame = old_frame.copy()
        cv2.drawKeypoints(old_frame, kp, old_out_frame, color=(0,0,255))

        kp = cv2.KeyPoint_convert(good_new)
        current_out_frame = current_frame.copy()
        cv2.drawKeypoints(current_frame, kp, current_out_frame, color=(0, 0, 255))

        cv2.imshow("old frame", old_out_frame)
        cv2.imshow("new frame", current_out_frame)

        key = cv2.waitKey(10000)
        if key == 27:  # exit on esc
            break
