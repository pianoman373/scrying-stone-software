import cv2
import camera
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera display')
    parser.add_argument('--mode', type=str, required=True, help='Camera mode. Either gstreamer_fisheye, gstreamer, webcam, or webcam_offset')
    args = parser.parse_args()

    cam = camera.CameraFeed(0, args.mode)
    cam2 = camera.CameraFeed(1, args.mode)
    cv2.namedWindow("video")

    while True:
        frame = cam.read()
        frame2 = cam2.read()

        cv2.imshow('video', frame)
        cv2.imshow('video2', frame2)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break

    cam.release()
    cam2.release()
    cv2.destroyAllWindows()