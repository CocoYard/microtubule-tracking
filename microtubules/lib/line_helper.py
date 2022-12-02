import numpy as np
import math
import cv2 as cv


def line_detect_possible_demo(image,pix1,pix2,thres, gap):
    w1 = 0.003
    w2 = 10
    blank = np.zeros(image.shape)
    hglines = blank.copy()
    d_1 = (pix1[0] - pix2[0]) / (pix1[1] - pix2[1] + 0.001)
    comp_angle = math.atan(d_1) + math.pi if math.atan(d_1) < 0 else math.atan(d_1)
    lines = cv.HoughLinesP(image, 1, np.pi / 180, threshold=thres, maxLineGap=gap)
    out = []
    min_loss = 100000
    out_d = 0
    if lines is None:
        cv.imshow('no hough lines', image)
        return
    # h = []
    # heapq.heapify(h)
    for line in lines:
        # print(line)
        y1,x1,y2,x2 = line[0]
        hglines += draw_line(blank, x1, y1, x2, y2) * 255

        p1 = np.array([x1,y1])
        p2 = np.array([x2,y2])
        d = (x1 - x2) / (y1 - y2 + 0.001)
        angle = math.atan(d) + math.pi if math.atan(d) < 0 else math.atan(d)
        loss1 = w1*(np.linalg.norm(p1-pix1)**2+np.linalg.norm(p2-pix2)**2) + w2 * abs(angle-comp_angle) - 0.1*math.sqrt((x1-x2)**2 + (y1-y2)**2)
        loss2 = w1*(np.linalg.norm(p1-pix2)**2+np.linalg.norm(p2-pix1)**2) + w2 * abs(comp_angle-angle) - 0.1*math.sqrt((x1-x2)**2 + (y1-y2)**2)
        loss = min(loss1,loss2)
        # item = (-loss, line, d)
        # if len(h) < 2:
        #     heapq.heappush(h, item)
        # elif loss > h[0][0]:
        #     heapq.heappop(h)
        #     heapq.heappush(h, item)
        if loss < min_loss:
            min_loss = loss
            out = line
            out_d = d
            print('selected line ', line, (loss - w2 * abs(angle-comp_angle) + 0.1*math.sqrt((x1-x2)**2 + (y1-y2)**2)), ' and ', abs(angle-comp_angle) * w2,
                  ' and ', 0.1*math.sqrt((x1-x2)**2 + (y1-y2)**2))
    # for i in range(0, len(h)):
    #     y1,x1,y2,x2 = h[i][1][0]
    #     hglines += draw_line(blank, x1, y1, x2, y2) * 255
    # out = h[0][1]
    # out_d = h[0][2]
    # for i in range(1, len(h)):
    #     out += h[i][1]
    #     out_d += h[i][2]
    # out = np.round(out / len(h))
    # out_d = out_d / len(h)
    y1, x1, y2, x2 = out[0]
    # hglines += draw_line(blank, x1, y1, x2, y2) * 255
    return out, out_d, hglines


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
