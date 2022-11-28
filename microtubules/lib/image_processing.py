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
    im = np.array(img.astype(np.uint8))
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    im = clahe.apply(im)
    im = np.array(im.astype(np.uint8))
    return im


def erosion(img_bin, k, t):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=t)
    return img_bin


def closing(img_bin, k):
    kernel = np.ones((k, k), np.uint8)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    return img_bin


def opening(img_bin, k, d):
    k = 10
    kernel = np.zeros((k, k), np.uint8)
    """
        0 0 0 0     j
        0 0 1 0     j
        0 1 0 0     j
        1 0 0 0     j
    i   i   i   i
    
    """
    d = -1
    for i in range(k):
        mid = k + round(i * d)
        low = max(0, mid - 2)
        high = min(k, mid + 2)
        for j in range(low, high):
            kernel[i, j] = 1
    print(kernel)
    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    return img_bin


def thresholding(img, k):
    _, img_bin = cv.threshold(img, k, img.max(), cv.THRESH_BINARY)
    return img_bin


def lineEnds(P):
    ## Central pixel and just one other must be set to be a line end
    return 255 * ((P[4] == 255) and np.sum(P) == 510)


def findEnds(bin_img):
    """
    find the endpoints of a connected binary image by searching for
    the farthest points in the connected image.
    Parameters
    ----------
    bin_img : (0 - 255) ndarray
    it may contain more than one connected component

    Returns
    -------
    output : 2d list, one for each endpoint
    """
    pts = []
    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] != 0:
                pts.append([i, j])
    max_dis = 0
    # find the farthest distance in the binary image
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
    """
        1. use Clahe to adjust the global contrast
        2. choose threshold to get binary image
        3. do threshold
        4. solve the cross problem by select the tubules in only one direction
        5. find all connected labels in the rectangle area formed by the input
                # To be edited....

            / /       /
                     /
                    //
                # this scenario should have been considered
        6. extract all pixels of the target labels
                        /
                      /
                    //
                  /
                  # con capture stretching and shrinking
        additional function, thinning
    """
    pix1 = [round(line[0][1]), round(line[0][2])]
    pix2 = [round(line[1][1]), round(line[1][2])]
    """ 1. use Clahe to adjust the global contrast """
    img = denosing(img, blur, method)

    # calculate derivative
    if pix2[0] < pix1[0]:
        # swap
        temp = pix2.copy()
        pix2 = pix1.copy()
        pix1 = temp
    x1 = pix1[0]
    x2 = pix2[0]
    y1 = pix1[1]
    y2 = pix2[1]
    d = (y2 - y1) / (x2 - x1)
    """ 2. choose threshold to get binary image """
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
    pix3 = (np.array(pix1) + np.array(pix2)) // 2
    """ 3. do threshold """
    bin_img = thresholding(img, thres)

    temp = bin_img.copy()
    bin_img = bin_img.astype(np.uint8)

    """ 4. solve the cross problem by opening in one direction """
    bin_img = opening(bin_img, 5, d)
    _, label = cv.connectedComponents(bin_img)
    # find the label using the mode of the labels around the selected two points

    counts_all = np.bincount(np.ndarray.flatten(label))
    counts_all[0] = 0
    counts = np.bincount(np.ndarray.flatten(label[temp11: temp12, temp21:temp22]))

    # set the background label as 0
    counts[0] = 0

    # target_label = np.argmax(counts)
    # all label indices which are inside the rectangle area formed by the end points
    target_labels = np.where(counts != 0)[0]
    targets = []
    """ 5. find all connected labels in the rectangle area formed by the input """
    if len(target_labels) > 1:
        for i in target_labels:
            if i == np.argmax(counts):
                targets.append(i)
            else:
                # To be edited....
                """
                      /
                     /   
                    //
                """    # this scenario should be considered later
                if (counts[i] / counts_all[i]) > 0.85:
                    targets.append(i)
    else:
        targets = target_labels
    targets = np.array(targets)
    # extract all pixels of the target labels
    label = np.isin(label, targets).astype(np.uint8)
    bin_img = bin_img * label
    bin_img = bin_img.astype(np.uint8)
    """ additional function, thinning """
    skeleton = (medial_axis(bin_img) * 255).astype(np.uint8)
    """"""
    result = generic_filter(skeleton, lineEnds, (3, 3))
    end_points = findEnds(result)

    return end_points, skeleton, bin_img, img, temp
