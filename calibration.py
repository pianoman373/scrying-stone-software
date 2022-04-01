import cv2
import numpy as np
import glob
from tqdm import tqdm

#============================================
# Camera calibration
#============================================
#Define size of chessboard target.
chessboard_size = (6,9)

if __name__ == '__main__':
    # Define arrays to save detected points
    obj_points = []  # 3D points in real world space
    img_points = []  # 3D points in image plane
    # Prepare grid and points to display
    objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # read images
    calibration_paths = glob.glob('./calibration_images/*')
    # Iterate over images to find intrinsic matrix
    for image_path in tqdm(calibration_paths):
        # Load image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Image loaded, Analizying...")
        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            print("Chessboard detected!")
            cv2.drawChessboardCorners(image, chessboard_size, corners, True)
            cv2.namedWindow("detected corners", cv2.WINDOW_NORMAL)
            cv2.imshow("detected corners", image)
            cv2.waitKey()
            # define criteria for subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # refine corner location (to subpixel accuracy) based on criteria.
            cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
        else:
            print("no chessboard detected :(")

    # Calibrate camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[::-1], None, None)
    # Save parameters into numpy file
    print("ret", ret);
    print("K", K);
    print("dist", dist);
    print("rvecs", rvecs);
    print("tvecs", tvecs);

    np.save("./camera_params/ret", ret)
    np.save("./camera_params/K", K)
    np.save("./camera_params/dist", dist)
    np.save("./camera_params/rvecs", rvecs)
    np.save("./camera_params/tvecs", tvecs)

    # print(exif_data)
    # focal_length_exif = exif_data['FocalLength']
    # print(focal_length_exif)
    # np.save("./camera_params/FocalLength", focal_length_exif)


    image = cv2.imread(calibration_paths[0])
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    img_1_undistorted = cv2.undistort(image, K, dist, None, new_camera_matrix)
    cv2.namedWindow("undistort", cv2.WINDOW_NORMAL)
    cv2.imshow("undistort", img_1_undistorted)
    cv2.waitKey()