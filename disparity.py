import cv2
import numpy as np
import open3d as o3d


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


# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

first = True

block_size = 5
def block_size_update(val):
    global block_size
    block_size = val

focal_length = 1.0
def focal_length_update(val):
    global focal_length
    focal_length = val

def reconstruct(img_1, img_2):
    # =========================================================
    # Stereo 3D reconstruction
    # =========================================================
    # Load camera parameters
    ret = np.load('./camera_params/ret.npy')
    K = np.load('./camera_params/K.npy')
    dist = np.load('./camera_params/dist.npy')
    # Specify image paths
    img_path1 = './reconstruct_this/left.jpg'
    img_path2 = './reconstruct_this/right.jpg'

    # Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
    h, w = img_2.shape[:2]

    # Get optimal camera matrix for better undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    # Undistort images
    img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
    img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)
    # Downsample each image 3 times (because they're too big)
    img_1_downsampled = img_1#downsample_image(img_1, 1)
    img_2_downsampled = img_2#downsample_image(img_2, 1)

    global first
    global block_size
    global focal_length

    # Set disparity parameters
    # Note: disparity range is tuned according to specific parameters obtained through trial and error.
    win_size = 5
    min_disp = -1
    max_disp = 63  # min_disp * 9
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=block_size,
                                   uniquenessRatio=5,
                                   speckleWindowSize=5,
                                   speckleRange=5,
                                   disp12MaxDiff=1,
                                   P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                   P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)
    # Compute disparity map
    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

    cv2.imshow('disparity', disparity_map)

    if first:
        cv2.createTrackbar('block size', 'disparity', 0, 10, block_size_update)
        cv2.createTrackbar('focal length', 'disparity', 0, 3, focal_length_update)
        first = False

    # Generate  point cloud.
    print("\nGenerating the 3D map...")
    # Get new downsampled width and height
    h, w = img_2_downsampled.shape[:2]
    # Load focal length.

    # Perspective transformation matrix
    # This transformation matrix is from the openCV documentation, didn't seem to work for me.
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1, 0]])
    # This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
    # Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
    Q2 = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, focal_length * 0.05, 0],  # Focal length multiplication obtained experimentally.
                     [0, 0, 0, 1]])
    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    # Get color points
    colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)
    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()
    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    # Define name for output file
    output_file = 'reconstructed.ply'
    # Generate point cloud
    print("\n Creating the output file... \n")
    create_output(output_points, output_colors, output_file)

    cloud = o3d.io.read_point_cloud("reconstructed.ply")  # Read the point cloud
    o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud

if __name__ == '__main__':
    cam1 = cv2.VideoCapture(2)
    cam2 = cv2.VideoCapture(3)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    while True:
        check1, frame1 = cam1.read()
        check2, frame2 = cam2.read()

        cv2.imshow('left', frame1)
        cv2.imshow('right', frame2)

        reconstruct(frame1, frame2)

        key = cv2.waitKey(1)
        if key == ord('a'):
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

