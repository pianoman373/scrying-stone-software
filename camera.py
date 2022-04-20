import cv2
import sys, getopt
import utils


def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=1680,
    display_height=840,
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
        "video/x-raw, format=(string)BGR ! appsink drop=True !"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

gstream_cap = None

class CameraFeed:
    def __init__(self, index):
        global gstream_cap
        self.index = index
        self.gstreamer = True
        self.fisheye = True

        for arg in sys.argv[1:]:
            if arg == "-w":
                self.gstreamer = False

            if arg == "-o":
                if self.index == 0:
                    self.index = 1
                elif self.index == 1:
                    self.index = 3

            if arg == "p":
                self.fisheye = False

        if self.gstreamer:
            if gstream_cap == None:
                self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
                gstream_cap = self.cap
            else:
                self.cap = gstream_cap


        else:
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            print("Error opening video stream")

    def read(self, padding=True):
        ret, frame = self.cap.read()

        if ret:
            if self.gstreamer:
                if self.index == 0:
                    frame = utils.crop_left(frame)

                if self.index == 1:
                    frame = utils.crop_right(frame)

                if padding:
                    frame = utils.crop_padding(frame, 150, 150, 150, 150)

            return frame
        else:
            print("Error reading video capture")
            return None
