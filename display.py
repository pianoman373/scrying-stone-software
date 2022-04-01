import cv2
import camera

if __name__ == '__main__':
    cam = camera.open_camera()
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