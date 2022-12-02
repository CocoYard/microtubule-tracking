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


def denosing(img):
    img = np.array(img.astype(np.uint8))
    blured = cv.medianBlur(img, 5)

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


def closing(img_bin, k, d):
    one_count = 0
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - k/5)), min(k, round(mid + k/5))
            for i in range(low, high):
                kernel[i, j] = 1
                one_count += 1
    else:
        d = 1/d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - k/5)), min(k, round(mid + k/5))
            for j in range(low, high):
                kernel[i, j] = 1
                one_count += 1
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def normal_closing(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def normal_opening(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def opening(img_bin, k, d):
    print(d)
    one_count = 0
    kernel = np.zeros((k, k), np.uint8)
    if -1 < d < 1:
        offset = round((k - d * k + 1) / 2)
        for j in range(k):
            mid = round(j * d) + offset
            low, high = max(0, round(mid - k/5)), min(k, round(mid + k/5))
            for i in range(low, high):
                kernel[i, j] = 1
                one_count += 1
    else:
        d = 1/d
        offset = round((k - d * k + 1) / 2)
        for i in range(k):
            mid = round(i * d) + offset
            low, high = max(0, round(mid - k/5)), min(k, round(mid + k/5))
            for j in range(low, high):
                kernel[i, j] = 1
                one_count += 1
    print(kernel)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def darken(polygon, img, ratio):
    """
    in place modify the img to make the local area darker, no output.
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
    left = polygon[:, 1].min()
    right = polygon[:, 1].max()
    top = polygon[:, 0].min()
    bot = polygon[:, 0].max()
    dists = []
    for j in range(left, right):
        for i in range(top, bot):
            if plygn.contains(Point(i, j)):
                # print(i, j)
                # print(plygn)
                dists.append(plygn.boundary.distance(Point(i, j)))
    dmax = max(dists)
    # print(dists)
    # print(dmax)
    k = ratio / np.log(dmax/2 + 1)
    m = 0

    for j in range(left, right):
        for i in range(top, bot):
            if plygn.contains(Point(i, j)):
                # print('k = ', k)
                diff = k*np.log(dists[m]+1)
                # print('diff = ', diff)
                m += 1
                # print('before ', img[i, j], end=' ')
                if diff > img[i, j]:
                    img[i, j] = 0
                    continue
                img[i, j] -= diff