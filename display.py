import cv2

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
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