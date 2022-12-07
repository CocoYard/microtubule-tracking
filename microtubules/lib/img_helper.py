from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import cv2 as cv


def crop_img(img, x1, x2, y1, y2):
    mask = np.zeros(img.shape)
    mask[x1:x2,y1:y2] = 1
    new_img = mask * img
    return new_img


def thresholding(img, k):
    _, img_bin = cv.threshold(img, k, img.max(), cv.THRESH_BINARY)
    return img_bin


def denoising(img):
    img = np.array(img.astype(np.uint8))
    # blured = cv.medianBlur(img, 11)
    blured = img
    im = np.array(blured.astype(np.uint8))
    clahe = cv.createCLAHE(clipLimit=None, tileGridSize=(8, 8))
    im = clahe.apply(im)

    im2 = np.array(img.astype(np.uint8))
    clahe = cv.createCLAHE(clipLimit=None, tileGridSize=(8, 8))
    im2 = clahe.apply(im2)
    return im, im2


def erosion(img_bin, k, t):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=t)
    return img_bin


# preallocate empty array and assign slice by chrisaycock
# at https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def closing(img_bin, k, d):
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for i in range(low, high):
                kernel[i, j] = 1
        up, down = 0, 0
        for i in range(k):
            if kernel[i, 0] == 1:
                up = i
                break
        for i in range(k-1, -1, -1):
            if kernel[i, -1] == 1:
                down = k-1-i
                break
        if up - down > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], -1, 0)
        elif down - up > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], 1, 0)
    else:
        d = 1/d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for j in range(low, high):
                kernel[i, j] = 1
        front, back = 0, 0
        for j in range(k):
            if kernel[-1, j] == 1:
                front = j
                break
        for j in range(k-1, -1, -1):
            if kernel[0, j] == 1:
                back = k-1-j
                break
        if front - back > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], -1, 0)
        elif back - front > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], 1, 0)

    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def opening(img_bin, k, d):
    print(d)
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for i in range(low, high):
                kernel[i, j] = 1
        up, down = 0, 0
        for i in range(k):
            if kernel[i, 0] == 1:
                up = i
                break
        for i in range(k-1, -1, -1):
            if kernel[i, -1] == 1:
                down = k-1-i
                break
        if up - down > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], -1, 0)
        elif down - up > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], 1, 0)
    else:
        d = 1/d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for j in range(low, high):
                kernel[i, j] = 1
        front, back = 0, 0
        for j in range(k):
            if kernel[-1, j] == 1:
                front = j
                break
        for j in range(k-1, -1, -1):
            if kernel[0, j] == 1:
                back = k-1-j
                break
        if front - back > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], -1, 0)
        elif back - front > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], 1, 0)
    print(kernel)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def close_open(img_bin, k, d):
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for i in range(low, high):
                kernel[i, j] = 1
        up, down = 0, 0
        for i in range(k):
            if kernel[i, 0] == 1:
                up = i
                break
        for i in range(k-1, -1, -1):
            if kernel[i, -1] == 1:
                down = k-1-i
                break
        if up - down > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], -1, 0)
        elif down - up > 1:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], 1, 0)
    else:
        d = 1/d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - 1)), min(k, round(mid))
            for j in range(low, high):
                kernel[i, j] = 1
        front, back = 0, 0
        for j in range(k):
            if kernel[-1, j] == 1:
                front = j
                break
        for j in range(k-1, -1, -1):
            if kernel[0, j] == 1:
                back = k-1-j
                break
        if front - back > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], -1, 0)
        elif back - front > 1:
            for i in range(k):
                kernel[i] = shift(kernel[i], 1, 0)
        d = 1/d

    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)

    if -1 < d < 1:
        up, down = 0, 0
        for i in range(k):
            if kernel[i, 0] == 1:
                up = i
                break
        for i in range(k - 1, -1, -1):
            if kernel[i, -1] == 1:
                down = k - 1 - i
                break
        if up - down > 0:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], -1, 0)
        elif down - up > 0:
            for j in range(k):
                kernel[:, j] = shift(kernel[:, j], 1, 0)
    else:
        front, back = 0, 0
        for j in range(k):
            if kernel[-1, j] == 1:
                front = j
                break
        for j in range(k - 1, -1, -1):
            if kernel[0, j] == 1:
                back = k - 1 - j
                break
        if front - back > 0:
            for i in range(k):
                kernel[i] = shift(kernel[i], -1, 0)
        elif back - front > 0:
            for i in range(k):
                kernel[i] = shift(kernel[i], 1, 0)
    print(kernel)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def normal_closing(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def normal_opening(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def darken(polygon, img, ratio):
    """
    in place operate the img to make the local area darker, no output.
    Parameters
    ----------
    polygon : 2d array
        the coordinates of the polygon in order.
    img : (0 - 255) 2d array
    ratio : float
        darkening ratio

    Returns
    -------
    None
    """
    # k
    polygon = polygon.round().astype('uint')
    plygn = Polygon(polygon)
    # find the square boundary
    left = max(polygon[:, 1].min(), 0)
    right = min(polygon[:, 1].max(), img.shape[1])
    top = max(polygon[:, 0].min(), 0)
    bot = min(polygon[:, 0].max(), img.shape[0])
    dists = []
    for j in range(left, right):
        for i in range(top, bot):
            if plygn.contains(Point(i, j)):
                dists.append(plygn.boundary.distance(Point(i, j)))
    dmax = max(dists)
    k = ratio / np.log(dmax/2 + 1)
    m = 0

    for j in range(left, right):
        for i in range(top, bot):
            if plygn.contains(Point(i, j)):
                diff = k*np.log(dists[m]+1)
                m += 1
                if diff > img[i, j]:
                    img[i, j] = 0
                    continue
                img[i, j] -= diff
