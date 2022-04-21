import cv2
import numpy as np

import re

# crop padding from opencv image
def crop_padding(image, left, right, top, bottom):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

def crop_left(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

def crop_right(img):
    cropped_img = img[0:img.shape[0], int(img.shape[1]/2):int(img.shape[1])]
    return cropped_img

def crop_sphere(image, radius):
    # define circles
    hh, ww = image.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2
    yc = hh2
    xc = ww2

    # draw filled circle in white on black background as mask
    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (xc, yc), radius, (255, 255, 255), -1)

    # apply mask to image
    result = cv2.bitwise_and(image, mask)

    result = crop_padding(result, ww2-radius, ww2-radius, hh2-radius, hh2-radius)

    return result

# Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

def colormap_depth(image):
    scaledDisparity = image - np.min(image)
    scaledDisparity = scaledDisparity * (255 / np.max(scaledDisparity))
    scaledDisparity = scaledDisparity.astype(np.uint8)

    disp = scaledDisparity.astype(np.uint8)

    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)

def draw_stereo_lines(image):
    ret_image = image.copy()

    #draw n horizontal lines across image
    for i in range(0, image.shape[0], 10):
        cv2.line(ret_image, (0, i), (image.shape[1], i), (0, 255, 0), 1)

    return ret_image

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def sort_files(files):
    return files.sort(key=natural_keys)