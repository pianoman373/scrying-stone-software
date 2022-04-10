import cv2
import numpy as np

# crop padding from opencv image
def crop_padding(image, left, right, top, bottom):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

def crop_left(img):
    cropped_img = img[0:img.shape[0], 0:int(img.shape[1]/2)]
    return cropped_img

def crop_right(img):
    cropped_img = img[0:img.shape[0], int(img.shape[1]/2):int(img.shape[1])]
    return cropped_img


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