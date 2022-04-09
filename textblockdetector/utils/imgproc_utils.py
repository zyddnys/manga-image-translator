import numpy as np
import cv2
import random

def hex2bgr(hex):
    gmask = 254 << 8
    rmask = 254
    b = hex >> 16
    g = (hex & gmask) >> 8
    r = hex & rmask
    return np.stack([b, g, r]).transpose()

def union_area(bboxa, bboxb):
    x1 = max(bboxa[0], bboxb[0])
    y1 = max(bboxa[1], bboxb[1])
    x2 = min(bboxa[2], bboxb[2])
    y2 = min(bboxa[3], bboxb[3])
    if y2 < y1 or x2 < x1:
        return -1
    return (y2 - y1) * (x2 - x1)

def get_yololabel_strings(clslist, labellist):
    content = ''
    for cls, xywh in zip(clslist, labellist):
        content += str(int(cls)) + ' ' + ' '.join([str(e) for e in xywh]) + '\n'
    if len(content) != 0:
        content = content[:-1]
    return content

# 4 points bbox to 8 points polygon
def xywh2xyxypoly(xywh, to_int=True):
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    if to_int:
        xyxypoly = xyxypoly.astype(np.int64)
    return xyxypoly

def xyxy2yolo(xyxy, w: int, h: int):
    if xyxy == [] or xyxy == np.array([]) or len(xyxy) == 0:
        return None
    if isinstance(xyxy, list):
        xyxy = np.array(xyxy)
    if len(xyxy.shape) == 1:
        xyxy = np.array([xyxy])
    yolo = np.copy(xyxy).astype(np.float64)
    yolo[:, [0, 2]] =  yolo[:, [0, 2]] / w
    yolo[:, [1, 3]] = yolo[:, [1, 3]] / h
    yolo[:, [2, 3]] -= yolo[:, [0, 1]]
    yolo[:, [0, 1]] += yolo[:, [2, 3]] / 2
    return yolo

def yolo_xywh2xyxy(xywh: np.array, w: int, h:  int, to_int=True):
    if xywh is None:
        return None
    if len(xywh) == 0:
        return None
    if len(xywh.shape) == 1:
        xywh = np.array([xywh])
    xywh[:, [0, 2]] *= w
    xywh[:, [1, 3]] *= h
    xywh[:, [0, 1]] -= xywh[:, [2, 3]] / 2
    xywh[:, [2, 3]] += xywh[:, [0, 1]]
    if to_int:
        xywh = xywh.astype(np.int64)
    return xywh

def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)
    
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=128):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2
    dh, dw = int(dh), int(dw)

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def resize_keepasp(im, new_shape=640, scaleup=True, interpolation=cv2.INTER_LINEAR, stride=None):
    shape = im.shape[:2]  # current shape [height, width]

    if new_shape is not None:
        if not isinstance(new_shape, tuple):
            new_shape = (new_shape, new_shape)
    else:
        new_shape = shape

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    if stride is not None:
        h, w = new_unpad
        if new_shape[0] % stride != 0 :
            new_h = (stride - (new_shape[0] % stride)) + h
        else :
            new_h = h
        if w % stride != 0 :
            new_w = (stride - (w % stride)) + w
        else :
            new_w = w
        new_unpad = (new_h, new_w)

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=interpolation)
    return im

def expand_textwindow(img_size, xyxy, expand_r=8, shrink=False):
    im_h, im_w = img_size[:2]
    x1, y1 , x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    paddings = int(round((max(h, w) * 0.25 + min(h, w) * 0.75) / expand_r))
    if shrink:
        paddings *= -1
    x1, y1 = max(0, x1 - paddings), max(0, y1 - paddings)
    x2, y2 = min(im_w-1, x2+paddings), min(im_h-1, y2+paddings)
    return [x1, y1, x2, y2]

def draw_connected_labels(num_labels, labels, stats, centroids, names="draw_connected_labels", skip_background=True):
    labdraw = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    max_ind = 0
    if isinstance(num_labels, int):
        num_labels = range(num_labels)
    
    # for ind, lab in enumerate((range(num_labels))):
    for lab in num_labels:
        if skip_background and lab == 0:
            continue
        randcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        labdraw[np.where(labels==lab)] = randcolor
        maxr, minr = 0.5, 0.001
        maxw, maxh = stats[max_ind][2] * maxr, stats[max_ind][3] * maxr
        minarea = labdraw.shape[0] * labdraw.shape[1] * minr

        stat = stats[lab]
        bboxarea = stat[2] * stat[3]
        if stat[2] < maxw and stat[3] < maxh and bboxarea > minarea:
            pix = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)
            pix[np.where(labels==lab)] = 255

            rect = cv2.minAreaRect(cv2.findNonZero(pix))
            box = np.int0(cv2.boxPoints(rect))
            labdraw = cv2.drawContours(labdraw, [box], 0, randcolor, 2)
            labdraw = cv2.circle(labdraw, (int(centroids[lab][0]),int(centroids[lab][1])), radius=5, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), thickness=-1)                

    cv2.imshow(names, labdraw)
    return labdraw

