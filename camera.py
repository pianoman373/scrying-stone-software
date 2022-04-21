import cv2
import sys, getopt
import utils


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,
    capture_height=2464,
    display_width=1120,
    display_height=840,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True !"
        % (
            sensor_id,
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
    def __init__(self, index, config):
        global gstream_cap
        self.index = index
        self.gstreamer = False
        self.fisheye = False

        if config == 'gstreamer_fisheye':
            self.gstreamer = True
            self.fisheye = True

        if config == 'gstreamer':
            self.gstreamer = True
            self.fisheye = False

        if config == 'webcam':
            self.gstreamer = False
            self.fisheye = False

        if config == 'webcam_offset':
            if self.index == 0:
                self.index = 1
            elif self.index == 1:
                self.index = 3


        if self.gstreamer:
            if self.fisheye:
                if gstream_cap == None:
                    self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
                    gstream_cap = self.cap
                else:
                    self.cap = gstream_cap

            else:
                print("passing gstreamer args: ", gstreamer_pipeline(index))
                self.cap = cv2.VideoCapture(gstreamer_pipeline(index), cv2.CAP_GSTREAMER)

        else:
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            print("Error opening video stream")

    def read(self):
        ret, frame = self.cap.read()

        if ret:
            if self.fisheye:
                if self.index == 0:
                    frame = utils.crop_left(frame)

                if self.index == 1:
                    frame = utils.crop_right(frame)

            return frame
        else:
            print("Error reading video capture")
            return None

    def release(self):
        self.cap.release()