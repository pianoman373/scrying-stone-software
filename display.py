import cv2
import camera

if __name__ == '__main__':
    cam = camera.CameraFeed(0)
    cam2 = camera.CameraFeed(1)
    cv2.namedWindow("video")

    while True:
        frame = cam.read(False)
        frame2 = cam2.read(False)

        cv2.imshow('video', frame)
        cv2.imshow('video2', frame2)

        key = cv2.waitKey(20)
        if key == 27:  # exit on esc
            break

    cam.release()
    cv2.destroyAllWindows()