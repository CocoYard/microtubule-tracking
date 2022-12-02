import numpy as np
from scipy.ndimage import generic_filter
from skimage.morphology import medial_axis
import cv2 as cv
from lib.img_helper import *
from lib.line_helper import *


def tiffToGray(img):
    return 255 * ((img - 0) / 65535)


def detectLine(img, line, k=10, gap=10, threshold=1, hgthres=20):
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
    img2, img = denosing(img)
    # calculate derivative
    x1, x2 = pix1[1], pix2[1]
    y1, y2 = pix1[0], pix2[0]
    derivative = (y2 - y1) / (x2 - x1+0.001)
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
            if thresholdmatrix[i, j] > 150:
                count_nonzero += 1
                total += thresholdmatrix[i, j]
    thres = total / count_nonzero * threshold
    print('thres = ', thres)
    """ 3. do threshold """
    bin_img = thresholding(img, thres)
    img2 = thresholding(img2, thres-5)
    img2 = img2.astype(np.uint8)
    img2 = normal_opening(img2, 3)

    bin_img = bin_img * (img2 / 255)

    bin_img = crop_img(bin_img,max(temp11-100,0), min(temp12+100, bin_img.shape[0]),
                       max(temp21-100,0), min(temp22+100, bin_img.shape[1]))

    first_bin = bin_img.copy()
    bin_img = bin_img.astype(np.uint8)

    """ 4. solve the cross problem by opening in one direction """
    bin_img = close_open(bin_img, k, derivative)
    # bin_img = closing(bin_img, k, derivative)
    # bin_img = opening(bin_img, k, derivative)

    _, label = cv.connectedComponents(bin_img)
    # find the labels in the square area
    counts_all = np.bincount(np.ndarray.flatten(label))
    counts_all[0] = 0
    # label -> number of label
    counts = np.bincount(np.ndarray.flatten(label[temp11: temp12, temp21:temp22]))

    # set the background label as 0
    counts[0] = 0

    # target_label = np.argmax(counts)
    # all label indices which are inside the rectangle area formed by the end points
    target_labels = np.where(counts != 0)[0]
    targets = []
    """ 5. find all connected labels in the rectangle area formed by the input """
    if len(target_labels) > 1:
        for labeli in target_labels:
            if labeli == np.argmax(counts):
                targets.append(labeli)
            elif counts[labeli]/counts_all[labeli] > 0.6:
                targets.append(labeli)
    else:
        targets = target_labels
    targets = np.array(targets)
    # extract all pixels of the target labels
    label = np.isin(label, targets).astype(np.uint8)
    temp = bin_img.copy()
    bin_img = bin_img * label
    temp1 = bin_img.copy()
    [[y1,x1,y2,x2]], derivative, hglines, hgline = line_detect_possible_demo(bin_img,pix1,pix2, hgthres, gap)
    bin_img = close_open(bin_img, k, derivative)
    # bin_img = closing(bin_img, k, derivative)
    # bin_img = opening(bin_img, k, derivative)

    """ 6. find all connected labels in the rectangle area formed by the Hough line """

    p1 = np.array([x1,y1])
    p2 = np.array([x2,y2])
    pix3 = (np.array(p1) + np.array(p2)) // 2
    l = np.linalg.norm(p2 - p1)

    """ delete the lines whose main part not in the incline area """
    # temp11 = max(min(p1[0], p2[0]) - 5, 0)
    # temp12 = min(max(p1[0], p2[0]) + 5, img.shape[0])
    # temp21 = max(min(p1[1], p2[1]) - 5, 0)
    # temp22 = min(max(p1[1], p2[1]) + 5, img.shape[1])
    # delete remote points to the Hough line
    for i in range(max(temp11-100,0), min(temp12+100, bin_img.shape[0])):
        for j in range(max(temp21-100,0), min(temp22+100, bin_img.shape[1])):
            if bin_img[i, j] != 0:
                p3 = np.array([i, j])
                d = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
                d2 = np.linalg.norm(p3 - pix3)
                if d > 6 or d2 > 6/11 * l:
                    bin_img[i, j] = 0
    bin_img = bin_img.astype(np.uint8)
    """ additional function, thinning """
    skeleton = (medial_axis(bin_img) * 255).astype(np.uint8)
    """"""
    # result = generic_filter(skeleton, lineEnds, (3, 3))
    # end_points = findEnds(result)
    end_points = [p1, p2]
    return end_points, skeleton, bin_img, img, temp, temp1, first_bin, hglines, hgline
