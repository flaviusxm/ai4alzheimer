import cv2 as cv
import os
from PIL import Image
from utilities import show_image, save_image
from global_constants import IMAGE_SIZE, SCALED_DIMENSIONS, PAD_AMOUNT


# (200, 190): 31107
# (180, 180): 6447
# (176, 208): 6446
# (496, 248): 86437

def find_first_img_by_dimension(folder, target_dimensions):
    n = len(target_dimensions)
    file_names = {}

    for file_name in os.listdir(folder):
        path = f"{folder}/{file_name}"

        with Image.open(path) as img:
            w, h = img.size
            if (w, h) not in file_names:
                file_names[(w, h)] = file_name
                if len(file_names) == n:
                    break

    return file_names


def resize_image_hardcoded(img):
    save_image(img, "test", "original.jpg")

    h_old, w_old = img.shape[0], img.shape[1]

    h_scaled, w_scaled = SCALED_DIMENSIONS[h_old]
    img = cv.resize(img, (w_scaled, h_scaled), interpolation=cv.INTER_CUBIC)

    if h_scaled != IMAGE_SIZE:
        pad_top, pad_bottom = PAD_AMOUNT[h_old]
        img = cv.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=0, right=0,
                                borderType=cv.BORDER_CONSTANT, value=img[0, 0].tolist())
    elif w_scaled != IMAGE_SIZE:
        pad_left, pad_right = PAD_AMOUNT[h_old]
        img = cv.copyMakeBorder(img, top=0, bottom=0, left=pad_left, right=pad_right,
                                borderType=cv.BORDER_CONSTANT, value=img[0, 0].tolist())

    save_image(img, "test", "padded.jpg")

    return img


img = cv.imread("non-dementia/non-dementia-00002.jpg")
resize_image_hardcoded(img)


# dimensions = ((200, 190), (180, 180), (176, 208), (496, 248))
# file_names = find_first_img_by_dimension("moderate-dementia", dimensions)
# print(file_names)

# NON
# 496x248: 06563.jpg
# 200x190: 00002.jpg
# 180x180: 00001.jpg
# 176x208: 01415.jpg

# VERY MILD
# 496x248: 06114.jpg
# 200x190: 00001.jpg
# 180x180: 00005.jpg
# 176x208: 01310.jpg

# MILD
# 496x248: 05781.jpg
# 200x190: 00001.jpg
# 180x180: 00003.jpg
# 176x208: 01361.jpg

# MODERATE
# 496x248: 04086.jpg
# 200x190: 00001.jpg
# 180x180: 00003.jpg
# 176x208: 01023.jpg
