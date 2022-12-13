import numpy as np
import math
import cv2 as cv


def select_line(image, pix1, pix2):
    """
    Selects the target Hough line from a whole bunch of detected Hough lines.

    Parameters
    ----------
    image : 2d array
        Source image without Hough transformation yet.
    pix1 : list
        One of the endpoints' coordinates.
    pix2 : list
        The other endpoints' coordinates.

    Returns
    -------
    None or the following:
    out : 2d array
        The selected Hough line
    out_d : float
        The selected Hough line's derivative.
    hglines : 2d array
        A drawn image of all Hough lines for displaying purpose.
    tgt_hgline : list
        A drawn image of the selected Hough line for displaying purpose.
    """

    pix1 = np.array(pix1)
    pix2 = np.array(pix2)
    length = np.linalg.norm(pix2 - pix1)
    if length > 130:
        thres = 50
        gap = 20
    elif length > 80:
        thres = 30
        gap = 15
    elif length > 50:
        thres = 20
        gap = 10
    else:
        thres = 15
        gap = 5
    w1 = 0.005
    w2 = 15
    blank = np.zeros(image.shape)
    hglines = blank.copy()
    tgt_hgline = blank.copy()
    d_1 = (pix1[0] - pix2[0]) / (pix1[1] - pix2[1] + 0.001)
    comp_angle = math.atan(d_1) + math.pi if math.atan(d_1) < 0 else math.atan(d_1)
    lines = cv.HoughLinesP(image, 1, np.pi / 180, threshold=thres, maxLineGap=gap)
    out = []
    min_loss = 100000
    out_d = 0
    if lines is None:
        return
    for line in lines:
        y1, x1, y2, x2 = line[0]
        hglines += draw_line(blank, x1, y1, x2, y2) * 255
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        d = (x1 - x2) / (y1 - y2 + 0.001)
        angle = math.atan(d) + math.pi if math.atan(d) < 0 else math.atan(d)
        rotation = w2 * abs(angle-comp_angle)
        distance = min(w1*(np.linalg.norm(p1-pix1)**2+np.linalg.norm(p2-pix2)**2),
                       w1*(np.linalg.norm(p1-pix2)**2+np.linalg.norm(p2-pix1)**2))
        main_length = length/math.sqrt((x1-x2)**2 + (y1-y2)**2)
        loss = distance + rotation + main_length
        if loss < min_loss:
            tgt_hgline = draw_line(blank, x1, y1, x2, y2) * 255
            min_loss = loss
            out = line
            out_d = d
    return out, out_d, hglines, tgt_hgline


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
