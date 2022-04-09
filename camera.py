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

# crop padding from opencv image
def crop_padding(image, left, right, top, bottom):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

def crop_left(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

def crop_right(img):
    cropped_img = img[0:img.shape[0], int(img.shape[1]/2):int(img.shape[1])]
    return cropped_img

gstream_cap = None

class CameraFeed:
    def __init__(self, index):
        global gstream_cap
        self.index = index
        self.gstreamer = True

        for arg in sys.argv[1:]:
            if arg == "-w":
                self.gstreamer = False
                break

        if self.gstreamer:
            if gstream_cap == None:
                self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
                gstream_cap = self.cap
            else:
                self.cap = gstream_cap


        else:
            self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            print("Error opening video stream")

    def read(self):
        ret, frame = self.cap.read()

        if ret:
            if self.gstreamer:
                if self.index == 0:
                    frame = crop_left(frame)

                if self.index == 1:
                    frame = crop_right(frame)

                frame = crop_padding(frame, 150, 150, 150, 150)

            return frame
        else:
            print("Error reading video capture")
            return None
