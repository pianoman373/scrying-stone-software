import cv2
import numpy as np
import matplotlib.pyplot as plt
import camera
import glob
import disparity
import undistort
import utils
import argparse

# images0 = glob.glob("D:/mav0/cam0/data/*.png")
# images1 = glob.glob('D:/mav0/cam1/data/*.png')
# images0 = glob.glob("D:/00/image_0/*.png")
# images1 = glob.glob('D:/00/image_1/*.png')

# P0 = np.array([7.188560000000e+02,0.000000000000e+00,6.071928000000e+02,0.000000000000e+00,0.000000000000e+00,7.188560000000e+02,1.852157000000e+02,0.000000000000e+00,0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,0.000000000000e+00]).reshape((3,4))
# P1 = np.array([7.188560000000e+02,0.000000000000e+00,6.071928000000e+02,-3.861448000000e+02,0.000000000000e+00,7.188560000000e+02,1.852157000000e+02,0.000000000000e+00,0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,0.000000000000e+00]).reshape((3,4))
#
# P0 = np.load("camera_params/P0.npy")
# P1 = np.load("camera_params/P1.npy")

def extract_features(image, detector='orb', mask=None):
    """
    To extract features in an image using two different method
    """
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()

    kp, des = det.detectAndCompute(image, mask)
    return kp, des

def match_features(des1, des2, detector='orb', k=2):
    """
    To match features of two images using Brute-force method
    """
    if detector == 'sift':
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    elif detector == 'orb':
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)

    matches = matcher.knnMatch(des1, des2, k=k)

    return matches

def filter_matches_distance(matches, dist_threshold=0.45):
    """
    To filter out matches with ratios of distance higher than the threshold
    """
    filtered_matches = []
    for m, n in matches:
        if m.distance <= dist_threshold * n.distance:
            filtered_matches.append(m)

    return filtered_matches

def calc_depth_map(disp_left, k_left, t_left, t_right):
    """
    To calculate the depth values of each pixel and generate a map
    """
    b = abs(t_right[0] - t_left[0])

    f = k_left[0][0]

    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1

    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left

    return depth_map

def compute_left_disparity_map(img_left, img_right, matcher='sgbm'):
    """
    To calculate the disparity between the left frame and right frame
    """

    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = matcher

    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)

    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8 * 1 * block_size ** 2,
                                        P2=32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16

    return disp_left

def decompose_projection_matrix(p):
    """
    To extract useful matrices from projection matrix
    """
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]

    return k, r, t

def stereo_2_depth(img_left, img_right, P0, P1):
    """
    To calculate the depth of stereo cameras
    """
    disp = compute_left_disparity_map(img_left,
                                      img_right,
                                      matcher='sgbm')

    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)

    depth = calc_depth_map(disp, k_left, t_left, t_right)

    return depth

def estimate_motion(matches, kp1, kp2, k, depth1, max_depth=3000):
    """
    To estimate the pose of the camera in each frame
    """

    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])

    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]

    object_points = np.zeros((0, 3))
    delete = []

    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(round(v)), int(round(u))]

        if z > max_depth:
            delete.append(i)
            continue

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)

    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
    rmat = cv2.Rodrigues(rvec)[0]

    return rmat, tvec, image1_points, image2_points


def odometry(old_frame, current_frame, right_frame, P0, P1, k_left, headless=False):
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    right_frame_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Get keypoints and descriptors for left camera image of two sequential frames
    kp0, des0 = extract_features(old_frame_gray)
    kp1, des1 = extract_features(current_frame_gray)

    depth = stereo_2_depth(current_frame_gray, right_frame_gray, P0, P1)
    
    if (type(des1).__name__ == 'NoneType'):
        return np.eye(4), depth
    
    
    # Get matches between features detected in two subsequent frames
    matches_unfilt = match_features(des0, des1)

    # Filter matches if a distance threshold is provided by user
    matches = filter_matches_distance(matches_unfilt)

    image1_points = [kp0[m.queryIdx] for m in matches]
    image2_points = [kp1[m.trainIdx] for m in matches]

    

    if len(matches) < 10:
        return [0, 0, 0], depth

    # Estimate motion between sequential images of the left camera
    rmat, tvec, img1_points, img2_points = estimate_motion(matches,
                                                           kp0,
                                                           kp1,
                                                           k_left,
                                                           depth
                                                           )

    # Create a blank homogeneous transformation matrix
    Tmat = np.eye(4)
    Tmat[:3, :3] = rmat
    Tmat[:3, 3] = tvec.T

    T = np.linalg.inv(Tmat)

    if not headless:
        disp_color = utils.colormap_depth(depth)
        cv2.imshow("depth", depth * 0.2)

        old_frame_points = old_frame.copy()
        cv2.drawKeypoints(old_frame, image1_points, old_frame_points, color=(0, 0, 255))

        current_frame_points = current_frame.copy()
        cv2.drawKeypoints(current_frame, image2_points, current_frame_points, color=(0, 0, 255))

        cv2.imshow("old frame", old_frame_points)
        cv2.imshow("new frame", current_frame_points)
        cv2.imshow("right frame", right_frame)

    return T, depth

def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

def reproject(color, depth, K_inv):
    height, width = color.shape[:2]

    colors = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    colors = colors/255.0

    pixel_coords = pixel_coord_np(width, height)

    cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()

    return cam_coords.T[:, :3], colors
