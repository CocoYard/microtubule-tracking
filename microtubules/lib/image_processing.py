import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import generic_filter
import math
from PIL import Image, ImageEnhance
from skimage.morphology import medial_axis
import cv2 as cv


def tiffToGray(img):
    return 255 * ((img - 0) / 65535)


def denosing(img, blur, method):
    if method == 'g':
        blured = cv.GaussianBlur(img, (blur, blur), 10, 10)
        print(method, '1')
    else:
        blured = cv.blur(img, (blur, blur))
        print(method, '2')
    im = Image.fromarray(np.uint8(blured))
    enhance = ImageEnhance.Contrast(im)
    im = enhance.enhance(3)
    im = np.array(im)
    return im, blured


def erosion(img_bin, k, t):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=t)
    return img_bin


def closing(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def opening(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def thresholding(img, k):
    _, img_bin = cv.threshold(img, k, img.max(), cv.THRESH_BINARY)
    return img_bin


def lineEnds(P):
    ## Central pixel and just one other must be set to be a line end
    return 255 * ((P[4] == 255) and np.sum(P) == 510)


def findEnds(bin_img):
    pts = []
    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] != 0:
                pts.append([i, j])
    max_dis = 0
    for p1 in pts:
        for p2 in pts:
            p1 = np.array(p1)
            p2 = np.array(p2)
            d = np.linalg.norm(p2 - p1)
            if d > max_dis:
                max_dis = d
                output = np.array([p1, p2])
    return output


def detectLine(img, line, blur, method):
    pix1 = [round(line[0][1]), round(line[0][2])]
    pix2 = [round(line[1][1]), round(line[1][2])]
    ## gaussian blur and enhance contrast
    img, blured = denosing(img, blur, method)
    ## dynamic threshold, use the mean pixel value of the two surrounding points
    thres = (img[pix1[0] - 5:pix1[0] + 5, pix1[1] - 5:pix1[1] + 5].mean() + img[pix2[0] - 5:pix2[0] + 5,
                                                                            pix2[1] - 5:pix2[1] + 5].mean()) / 2

    bin_img = thresholding(img, thres)
    temp = thresholding(img, thres)
    bin_img = closing(bin_img, 5)
    _, label = cv.connectedComponents(bin_img)
    # find the label using the mode of the labels around the selected two points
    counts = np.bincount(np.ndarray.flatten(np.concatenate((label[pix1[0] - 5: pix1[0] + 5, pix1[1] - 5:pix1[1] + 5],
                                                            label[pix2[0] - 5: pix2[0] + 5, pix2[1] - 5:pix2[1] + 5]),
                                                           axis=0)))
    counts[0] = 0
    target_label = np.argmax(counts)
    label[label != target_label] = 0
    label[label == target_label] = 1
    bin_img = bin_img * label
    p1 = np.array(pix1)
    p2 = np.array(pix2)
    l = np.linalg.norm(p2 - p1)
    # delete remote points to the input line
    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] != 0:
                p3 = np.array([i, j])
                d = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

                if d > 3:
                    bin_img[i, j] = 0
    # threshold
    thres_img = bin_img
    bin_img = bin_img.astype(np.uint8)
    bin_img = erosion(bin_img, 3, 1)
    skeleton = (medial_axis(bin_img) * 255).astype(np.uint8)

    result = generic_filter(skeleton, lineEnds, (3, 3))
    end_points = findEnds(result)

    im = Image.fromarray(np.uint8(skeleton))
    return im, end_points, skeleton, thres_img, img, blured, temp
