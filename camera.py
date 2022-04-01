import cv2
import sys, getopt


def gstreamer_pipeline(
    capture_width=3264*1.5,
    capture_height=2464,
    display_width=1432*1.5,
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

def open_camera():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:")
    except getopt.GetoptError:
        print("arguments: -i <input number>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            if arg == "gstreamer":
                return cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            else:
                input_number = int(arg)
                return cv2.VideoCapture(input_number)

    return cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)