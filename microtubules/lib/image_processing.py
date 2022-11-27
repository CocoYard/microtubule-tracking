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
    # if method == 'g':
    #     blured = cv.GaussianBlur(img, (blur, blur), 10, 10)
    #     # print(method, '1')
    # else:
    #     blured = cv.blur(img, (blur, blur))
    #     # print(method, '2')
    # im = Image.fromarray(np.uint8(blured))
    # enhance = ImageEnhance.Contrast(im)
    # im = enhance.enhance(3)
    # im = np.array(im)
    im = np.array(img.astype(np.uint8))
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    im = clahe.apply(im)
    # im = np.array(im)
    im = np.array(im.astype(np.uint8))
    blured =[]
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
    # img = np.array(img)
    # blured=[]

    ## dynamic threshold, use the mean pixel value of the two surrounding points
    # thres = (img[pix1[0] - 5:pix1[0] + 5, pix1[1] - 5:pix1[1] + 5].mean() + img[pix2[0] - 5:pix2[0] + 5,
    #                                                                         pix2[1] - 5:pix2[1] + 5].mean()) / 2
    """ """
    # s = 0
    # d = (pix1[1] - pix2[1]) / (pix1[0] - pix2[0])
    # n = 0
    # x = 1
    # if pix1[0] > pix2[0]:
    #     x = -1
    # for i in range(pix1[0], pix2[0], x):
    #     mid = (i - pix1[0]) * d
    #     high = round(mid + 5)
    #     low = round(mid - 5)
    #     for j in range(low, high):
    #         s += img[i, j]
    #         n += 1
    # thres = s / n

    temp11 = max(min(pix1[0], pix2[0]) - 5, 0)
    temp12 = min(max(pix1[0], pix2[0]) + 5, img.shape[0])
    temp21 = max(min(pix1[1], pix2[1]) - 5, 0)
    temp22 = min(max(pix1[1], pix2[1]) + 5, img.shape[1])
    total = 0
    count_nonzero = 0
    thresholdmatrix = img[temp11:temp12, temp21:temp22]
    for i in range(thresholdmatrix.shape[0]):
        for j in range(thresholdmatrix.shape[1]):
            if thresholdmatrix[i, j] > 6:
                count_nonzero += 1
                total += thresholdmatrix[i, j]
    thres = total / count_nonzero
    """ """
    pix3 = (np.array(pix1) + np.array(pix2)) // 2
    # thres_max = img[pix3[0] - 5:pix3[0] + 5, pix3[1] - 5:pix3[1] + 5].max()
    # threshold = img[pix3[0] - 5:pix3[0] + 5, pix3[1] - 5:pix3[1] + 5].mean()


    # edges = cv.Canny(img, threshold, thres_max)
    # Image.fromarray(np.uint8(edges)).show()

    bin_img = thresholding(img, thres)
    temp = thresholding(img, thres)
    bin_img = bin_img.astype(np.uint8)
    bin_img = opening(bin_img, 5)
    # bin_img = closing(bin_img, 3)
    _, label = cv.connectedComponents(bin_img)
    # find the label using the mode of the labels around the selected two points

    counts_all = np.bincount(np.ndarray.flatten(label))
    counts_all[0] = 0
    counts = np.bincount(np.ndarray.flatten(label[temp11: temp12, temp21:temp22]))

    counts[0] = 0

    # print(counts)

    # target_label = np.argmax(counts)
    target_labels = np.where(counts != 0)[0]
    targets = []
    if len(target_labels) > 1:
        for i in target_labels:
            if i == np.argmax(counts):
                targets.append(i)
            else:
                if (counts[i] / counts_all[i]) > 0.85:
                    targets.append(i)
    else:
        targets = target_labels
    targets = np.array(targets)

    label = np.isin(label, targets).astype(np.uint8)
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
                d2 = np.linalg.norm(p3 - pix3)
                if d > l/20 or d2 > 3 / 4 * l:
                    bin_img[i, j] = 0
    # threshold
    thres_img = bin_img
    bin_img = bin_img.astype(np.uint8)
    # bin_img = erosion(bin_img, 3, 1)
    skeleton = (medial_axis(bin_img) * 255).astype(np.uint8)

    result = generic_filter(skeleton, lineEnds, (3, 3))
    end_points = findEnds(result)

    im = Image.fromarray(np.uint8(skeleton))
    return im, end_points, skeleton, thres_img, img, blured, temp
