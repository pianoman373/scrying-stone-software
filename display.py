import cv2

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

if __name__ == '__main__':
    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cv2.namedWindow("video")

    if cam.isOpened():

        while True:
            check, frame = cam.read()

            if check:
                cv2.imshow('video', frame)
            else:
                print("Error: failed to retrieve frame")

            key = cv2.waitKey(20)
            if key == 27: #exit on esc
                break

        cam.release()
        cv2.destroyAllWindows()
    else:
        print("Error: failed to open video capture")