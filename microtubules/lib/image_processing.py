import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import generic_filter
import math
from PIL import Image, ImageEnhance
from skimage.morphology import medial_axis
import cv2 as cv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def tiffToGray(img):
    return 255 * ((img - 0) / 65535)


def denosing(img):
    blured = cv.medianBlur(img,3)
    im = np.array(blured.astype(np.uint8))
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
    print(d)
    one_count = 0
    kernel = np.zeros((k, k), np.uint8)
    offset = round((k - d*k) / 2)
    for j in range(k):
        mid = round(j * d) + offset
        low, high = max(0, mid - 2), min(k, mid + 2)
        for i in range(low, high):
            kernel[i, j] = 1
            one_count += 1
    if one_count < 2*k:
        kernel[:, k//3:2*k//3] = 1
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
    bin_img : (0 - 255) 2d array
        it may contain more than one connected component

    Returns
    -------
    output : 2d array, one for each endpoint
    """
    pts = []
    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] != 0:
                pts.append([i, j])
    max_dis = 0
    output = 'err'
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


def draw_line(mat, x0, y0, x1, y1, inplace=False):
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('Invalid coordinates.')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 2
        return mat if not inplace else None
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Write line ends
    mat[x0, y0] = 2
    mat[x1, y1] = 2
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    mat[x, y] = 1
    if not inplace:
        return mat if not transpose else mat.T


def line_detect_possible_demo(image,pix1,pix2):
    w1 = 0.003
    w2 = 10
    blank = np.zeros(image.shape)
    hglines = blank.copy()
    d_1 = (pix1[0] - pix2[0]) / (pix1[1] - pix2[1] + 0.001)
    comp_angle = math.atan(d_1) + math.pi if math.atan(d_1) < 0 else math.atan(d_1)
    lines = cv.HoughLinesP(image, 1, np.pi / 180, 30,maxLineGap=20)
    out = []
    min_loss = 100000
    out_d = 0
    for line in lines:
        print(line)
        y1,x1,y2,x2 = line[0]
        hglines += draw_line(blank, x1, y1, x2, y2) * 255

        p1 = np.array([x1,y1])
        p2 = np.array([x2,y2])
        d = (x1 - x2) / (y1 - y2 + 0.001)
        angle = math.atan(d) + math.pi if math.atan(d) < 0 else math.atan(d)
        loss1 = w1*(np.linalg.norm(p1-pix1)**2+np.linalg.norm(p2-pix2)**2) + w2 * abs(angle-comp_angle)
        loss2 = w1*(np.linalg.norm(p1-pix2)**2+np.linalg.norm(p2-pix1)**2) + w2 * abs(comp_angle-angle)
        loss = min(loss1,loss2)
        if loss < min_loss:
            min_loss=loss
            out = line
            out_d = d
        print((loss - w2 * abs(angle-comp_angle)), ' and ', abs(angle-comp_angle) * w2)


        # cv.line(blank,(x1,y1),(x2,y2),(0,0,255),2)

    return out, out_d, hglines


def detectLine(img, line, polygon, k=10, dark_ratio=2):
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
    img = denosing(img)
    """ dark the area """
    if polygon is not None:
        darken(polygon, img, dark_ratio)
    temp = img.copy()
    # calculate derivative
    x1, x2 = pix1[1], pix2[1]
    y1, y2 = pix1[0], pix2[0]
    # derivative = (y2 - y1) / (x2 - x1)
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
            if thresholdmatrix[i, j] > 80:
                count_nonzero += 1
                total += thresholdmatrix[i, j]
    thres = total / count_nonzero
    pix3 = (np.array(pix1) + np.array(pix2)) // 2
    """ 3. do threshold """
    bin_img = thresholding(img, thres)

    first_bin = bin_img.copy()
    bin_img = bin_img.astype(np.uint8)

    """ 4. solve the cross problem by opening in one direction """
    # bin_img = opening(bin_img, k, derivative)
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
                targets.append(i)
    else:
        targets = target_labels
    targets = np.array(targets)
    # extract all pixels of the target labels
    label = np.isin(label, targets).astype(np.uint8)
    bin_img = bin_img * label

    [[y1,x1,y2,x2]], derivative, hglines =line_detect_possible_demo(bin_img,pix1,pix2)

    bin_img = opening(bin_img, k, derivative)


    p1 = np.array([x1,y1])
    p2 = np.array([x2,y2])
    l = np.linalg.norm(p2 - p1)
    # delete remote points to the input line
    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] != 0:
                p3 = np.array([i, j])
                d = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
                d2 = np.linalg.norm(p3 - pix3)
                if d > l / 20 or d2 > 3 / 4 * l:
                    bin_img[i, j] = 0
    bin_img = bin_img.astype(np.uint8)
    """ additional function, thinning """
    skeleton = (medial_axis(bin_img) * 255).astype(np.uint8)
    """"""
    result = generic_filter(skeleton, lineEnds, (3, 3))
    end_points = findEnds(result)

    return end_points, skeleton, bin_img, img, temp, first_bin, hglines
