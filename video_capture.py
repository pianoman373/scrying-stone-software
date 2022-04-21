import cv2
import camera
import argparse
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cameera display')
    parser.add_argument('--mode', type=str, required=True, help='Camera mode. Either gstreamer_fisheye, gstreamer, webcam, or webcam_offset')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--framerate', type=int, required=False, default=20, help='Capture framerate')
    args = parser.parse_args()

    cam = camera.CameraFeed(0, args.mode)
    cam2 = camera.CameraFeed(1, args.mode)
    cv2.namedWindow("video")

    target_frametime = 1.0 / args.framerate
    frame_number = 0

    if not os.path.exists(args.output + '/left'):
        os.makedirs(args.output + '/left')

    if not os.path.exists(args.output + '/right'):
        os.makedirs(args.output + '/right')

    while True:
        start = time.time()

        frame = cam.read()
        frame2 = cam2.read()

        cv2.imshow('video', frame)
        cv2.imshow('video2', frame2)

        cv2.imwrite(args.output + '/left/' + str(frame_number).zfill(5) + '.png', frame)
        cv2.imwrite(args.output + '/right/' + str(frame_number).zfill(5) + '.png', frame2)

        frame_number += 1

        end = time.time()
        elapsed = end - start

        wait_time_ms = max(1, int((target_frametime - elapsed) * 1000))

        key = cv2.waitKey(wait_time_ms)
        if key == 27:  # exit on esc
            break

    cam.release()
    cam2.release()
    cv2.destroyAllWindows()