import cv2
import numpy as np
#import open3d as o3d
import threading

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

def nothing(x):
    pass

def reconstruct_init():
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 1200, 1200)

    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disparity', 1200, 1200)

    cv2.createTrackbar('minDisparity', 'disp', 0, 17, nothing)
    cv2.createTrackbar('maxDisparity', 'disp', 5, 100, nothing)
    cv2.createTrackbar('blockSize', 'disp', 2, 11, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 5, 20, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 5, 20, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 5, 20, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 1, 20, nothing)
    cv2.createTrackbar('win_size', 'disp', 5, 20, nothing)

computing = False
#pcd = o3d.geometry.PointCloud()

def reconstruct(img_1, img_2):
    global computing

    computing = True
    # =========================================================
    # Stereo 3D reconstruction
    # =========================================================
    # Load camera parameters
    ret = np.load('./camera_params/ret.npy')
    K = np.load('./camera_params/K.npy')
    dist = np.load('./camera_params/dist.npy')

    # Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
    h, w = img_2.shape[:2]

    # Get optimal camera matrix for better undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    #convert images to grayscale
    img1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # Undistort images
    img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
    img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)
    # Downsample each image 3 times (because they're too big)
    img_1_downsampled = img1_gray#downsample_image(img1_gray, 1)
    img_2_downsampled = img2_gray#downsample_image(img2_gray, 1)

    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')
    maxDisparity = cv2.getTrackbarPos('maxDisparity', 'disp')*16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    win_size = cv2.getTrackbarPos('win_size', 'disp')

    # Note: disparity range is tuned according to specific parameters obtained through trial and error.
    min_disp = minDisparity
    max_disp = maxDisparity  # min_disp * 9
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=blockSize,
                                   uniquenessRatio=uniquenessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange,
                                   disp12MaxDiff=disp12MaxDiff,
                                   P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                   P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)
    # Compute disparity map
    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

    # cv2.imshow('disparity', disparity_map)

    # Generate  point cloud.
    print("\nGenerating the 3D map...")
    # Get new downsampled width and height
    h, w = img_2_downsampled.shape[:2]
    # Load focal length.
    focal_length = 3.6

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
    colors = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()
    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    #global pcd
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(output_points)
    # Define name for output file
    output_file = 'reconstructed.ply'
    # Generate point cloud
    #print("\n Creating the output file... \n")
    #create_output(output_points, output_colors, output_file)

    computing = False


def crop_left(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

def crop_right(img):
    cropped_img = img[0:img.shape[0], int(img.shape[1]/2):int(img.shape[1])]
    return cropped_img

if __name__ == '__main__':
    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    check, frame = cam.read()

    #vis = o3d.visualization.Visualizer()
    #vis.create_window()

    reconstruct_init()

    frame1 = crop_left(frame)
    frame2 = crop_right(frame)

    t1 = threading.Thread(target=reconstruct, args=(frame1, frame2,))
    t1.start()

    print("now looping")
    while True:
        check, frame = cam.read()

        frame1 = crop_left(frame)
        frame2 = crop_right(frame)

        cv2.imshow('left', frame1)
        cv2.imshow('right', frame2)


        if not computing:
            #cloud = o3d.io.read_point_cloud("reconstructed.ply")  # Read the point cloud
            #vis.clear_geometries()
            #vis.add_geometry(pcd)

            t1 = threading.Thread(target=reconstruct, args=(frame1, frame2,))
            t1.start()

        #vis.poll_events()
        #vis.update_renderer()

        key = cv2.waitKey(1)
        if key == ord('a'):
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

