import cv2
import numpy as np

import camera

lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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
    id = 0

    focal = 718.8560
    pp = (int(current_frame.shape[0]/2), int(current_frame.shape[1]/2))

    R = np.zeros(shape=(3, 3))
    t = np.zeros(shape=(3, 3))

    K = np.load("camera_params/K0.npy")
    D = np.load("camera_params/D0.npy")

    while True:
        old_frame = current_frame.copy()
        current_frame = cam.read(False)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if n_features < 2000:
            p0 = detect(detector, old_frame)


        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, current_frame, p0, None, **lk_params)

        if p1 is not None:
            good_old = p0[st == 1]
            good_new = p1[st == 1]

        if len(good_new) <= 0:
            continue

        if id < 2:
            E, _ = cv2.findEssentialMat(good_new, good_old, focal, pp, cv2.RANSAC, 0.999, 1.0,
                                        None)

            if E is not None:
                print(R)

                cv2.recoverPose(good_old, good_new, K, D, K, D, None, R, t)
                # _, R, t, _ = cv2.recoverPose(E, good_old, good_new, R, t, focal,
                #                                        pp, None)

                id += 2
        else:
            E, _ = cv2.findEssentialMat(good_new, good_old, focal, pp, cv2.RANSAC, 0.999, 1.0,
                                        None)

            if E is not None and E.shape == (3, 3):
                _, _, R_new, t_new, _ = cv2.recoverPose(good_old, good_new, K, D, K, D, None, R.copy(), t.copy())

            absolute_scale = 1.0
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                t = t + absolute_scale * R.dot(t_new)
                R = R_new.dot(R)

            id += 1

        n_features = good_new.shape[0]

        #draw keypoints
        # kp = []
        # for i in good_new:
        #     kp.append(cv2.KeyPoint(x=i[0], y=i[1], size=0, angle=0, response=0, octave=0, class_id=0))

        print('position:', get_mono_coordinates(t))

        kp = cv2.KeyPoint_convert(good_new)
        out_frame = current_frame.copy()
        cv2.drawKeypoints(current_frame, kp, out_frame, color=(0,0,255))

        cv2.imshow("frame", out_frame)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break
