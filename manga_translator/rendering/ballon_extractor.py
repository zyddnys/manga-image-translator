import cv2
from typing import Tuple, List
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0) -> List:
    assert ratio > 1.0
    
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]

    # https://numpy.org/doc/stable/reference/generated/numpy.roots.html
    coeff = [aspect_ratio, w+h*aspect_ratio, (1-ratio)*w*h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
    rect[::2] = np.clip(rect[::2], 0, im_w - 1)
    rect[1::2] = np.clip(rect[1::2], 0, im_h - 1)
    return rect.tolist()

def extract_ballon_region(img: np.ndarray, ballon_rect: List, enlarge_ratio=1, verbose=False) -> Tuple[np.ndarray, int, List]:

    x1, y1, x2, y2 = ballon_rect[0], ballon_rect[1], ballon_rect[2] + ballon_rect[0], ballon_rect[3] + ballon_rect[1]
    if enlarge_ratio > 1:
        x1, y1, x2, y2 = enlarge_window([x1, y1, x2, y2], img.shape[1], img.shape[0], enlarge_ratio, aspect_ratio=ballon_rect[3] / ballon_rect[2])

    img = img[y1:y2, x1:x2].copy()

    kernel = np.ones((3,3), np.uint8)
    orih, oriw = img.shape[0], img.shape[1]
    scaleR = 1
    if orih > 300 and oriw > 300:
        scaleR = 0.6
    elif orih < 120 or oriw < 120:
        scaleR = 1.4

    if scaleR != 1:
        h, w = img.shape[0], img.shape[1]
        orimg = np.copy(img)
        img = cv2.resize(img, (int(w*scaleR), int(h*scaleR)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[0], img.shape[1]
    img_area = h * w

    cpimg = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(cpimg, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
    cons, hiers = cv2.findContours(detected_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), BLACK, 1, cv2.LINE_8)

    ballon_mask = np.zeros((h, w), np.uint8)
    min_retval = np.inf
    mask = np.zeros((h, w), np.uint8)
    difres = 10
    seedpnt = (int(w/2), int(h/2))
    for i in range(len(cons)):
        rect = cv2.boundingRect(cons[i])
        if rect[2]*rect[3] < img_area*0.4:
            continue

        mask = cv2.drawContours(mask, cons, i, (255), 2)
        cpmask = np.copy(mask)
        cv2.rectangle(mask, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
        retval, _, _, rect = cv2.floodFill(cpmask, mask=None, seedPoint=seedpnt, flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))

        if retval <= img_area * 0.3:
            mask = cv2.drawContours(mask, cons, i, (0), 2)
        if retval < min_retval and retval > img_area * 0.3:
            min_retval = retval
            ballon_mask = cpmask

    ballon_mask = 127 - ballon_mask
    ballon_mask = cv2.dilate(ballon_mask, kernel,iterations = 1)
    ballon_area, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt, flags=4, newVal=(30), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
    ballon_mask = 30 - ballon_mask    
    retval, ballon_mask = cv2.threshold(ballon_mask, 1, 255, cv2.THRESH_BINARY)
    ballon_mask = cv2.bitwise_not(ballon_mask, ballon_mask)

    box_kernel = int(np.sqrt(ballon_area) / 30)
    if box_kernel > 1:
        box_kernel = np.ones((box_kernel,box_kernel),np.uint8)
        ballon_mask = cv2.dilate(ballon_mask, box_kernel, iterations = 1)
        ballon_mask = cv2.erode(ballon_mask, box_kernel, iterations = 1)

    if scaleR != 1:
        img = orimg
        ballon_mask = cv2.resize(ballon_mask, (oriw, orih))

    if verbose:
        cv2.imshow('ballon_mask', ballon_mask)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return ballon_mask, [x1, y1, x2, y2]
