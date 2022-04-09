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

class CameraFeed:
    def __init__(self, index):
        self.index = index
        self.gstreamer = True

        for arg in sys.argv[1:]:
            if arg == "-w":
                self.gstreamer = False
                break

        if self.gstreamer:
            self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            print("Error opening video stream")

    def read(self):
        ret, frame = self.cap.read()

        if ret:
            return frame
        else:
            print("Error reading video capture")
            return None
