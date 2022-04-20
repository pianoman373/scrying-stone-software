import cv2
import numpy as np
import matplotlib.pyplot as plt
import camera
import glob
import open3d as o3d
import disparity
import undistort
import utils
#
# images0 = glob.glob("D:/mav0/cam0/data/*.png")
# images1 = glob.glob('D:/mav0/cam1/data/*.png')
images0 = glob.glob("D:/00/image_0/*.png")
images1 = glob.glob('D:/00/image_1/*.png')
#
# images0 = glob.glob("D:/MovementTest/left/*.png")
# images1 = glob.glob('D:/MovementTest/right/*.png')

import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


images0.sort(key=natural_keys)
images1.sort(key=natural_keys)



P0 = np.array([7.188560000000e+02,0.000000000000e+00,6.071928000000e+02,0.000000000000e+00,0.000000000000e+00,7.188560000000e+02,1.852157000000e+02,0.000000000000e+00,0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,0.000000000000e+00]).reshape((3,4))
P1 = np.array([7.188560000000e+02,0.000000000000e+00,6.071928000000e+02,-3.861448000000e+02,0.000000000000e+00,7.188560000000e+02,1.852157000000e+02,0.000000000000e+00,0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,0.000000000000e+00]).reshape((3,4))


def extract_features(image, detector='sift', mask=None):
    """
    To extract features in an image using two different method
    """
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()

    kp, des = det.detectAndCompute(image, mask)
    return kp, des

def match_features(des1, des2, detector='sift', k=2):
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

def generate_pointcloud(depth, color):
    w, h = depth.shape[:2]



if __name__ == "__main__":
    # cv2.namedWindow('old frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('old frame', 400, 400)
    #
    # cv2.namedWindow('new frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('new frame', 400, 400)
    #
    # cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('disparity', 400, 400)

    # Generate a plt
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)

    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left = decompose_projection_matrix(P0)

    id = 1

    T_tot = np.eye(4)
    trajectory = np.zeros((len(images0), 3, 4))
    trajectory[0] = T_tot[:3, :]


    vis = o3d.visualization.Visualizer()
    vis.create_window()

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    while True:
        if id >= len(images0):
            id = 1

        old_frame = cv2.imread(images0[id-1])
        current_frame = cv2.imread(images0[id])
        right_frame = cv2.imread(images1[id])

        height, width = right_frame.shape[:2]
        T = np.float32([[1, 0, 000], [0, 1, -29]])

        # We use warpAffine to transform
        # the image using the matrix, T
        #right_frame = cv2.warpAffine(right_frame, T, (width, height))

        old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        right_frame_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Get keypoints and descriptors for left camera image of two sequential frames
        kp0, des0 = extract_features(old_frame_gray)
        kp1, des1 = extract_features(current_frame_gray)

        # Get matches between features detected in two subsequent frames
        matches_unfilt = match_features(des0,
                                        des1
                                        )

        # Filter matches if a distance threshold is provided by user
        matches = filter_matches_distance(matches_unfilt)

        image1_points = [kp0[m.queryIdx] for m in matches]
        image2_points = [kp1[m.trainIdx] for m in matches]

        depth = stereo_2_depth(current_frame_gray, right_frame_gray, P0, P1)

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

        T_tot = T_tot.dot(np.linalg.inv(Tmat))

        trajectory[id + 1, :, :] = T_tot[:3, :]

        xs = trajectory[:id + 2, 0, 3]
        ys = trajectory[:id + 2, 1, 3]
        zs = trajectory[:id + 2, 2, 3]

        disp_color = utils.colormap_depth(depth)
        cv2.imshow("depth", depth*0.02)

        cv2.drawKeypoints(old_frame, image1_points, old_frame, color=(0,0,255))

        cv2.drawKeypoints(current_frame, image2_points, current_frame, color=(0, 0, 255))

        cv2.imshow("old frame", old_frame)
        cv2.imshow("new frame", current_frame)
        cv2.imshow("right frame", right_frame)

        generate_pointcloud(depth, current_frame)

        position = [xs[-2]*0.01, ys[-2]*0.01, zs[-2]*0.01]

        line_set.points.append([position[0], position[1], position[2]])
        line_set.colors.append([255, 0, 0])
        p = len(line_set.points)
        line_set.lines.append([p, p - 1])

        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()

        id += 1

        key = cv2.waitKey(1)
        if key == 27:  # exit on esc
            break
