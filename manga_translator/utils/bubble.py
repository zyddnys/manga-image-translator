import cv2
import numpy as np

'''
Starting from the edge of the text block recognized by OCR, delete irrelevant white or black lines one pixel at a time until the smallest text area is retained. 
Then expand the text area by 10 -20 pixels and check if there are three or more pure white borders (>0.98) outside the boundary. If yes, it is determined to be a speech bubble. Calculate the proportion of white pixels on each of the four edges, and if the sum of the four proportions is greater than 3.5, it is considered to be a speech bubble. 
For the upper and lower points on the left and right sides of the text block, white point in all point.


'''


# sd : offset sd. extend rect area
def offset_margin(x, y, text_w, text_h, img_gray, sd=10):
    img_h, img_w = img_gray.shape[:2]
    # left top->bottom
    roi1 = img_gray[max(y - sd, 0):min(y + text_h + sd, img_h), max(x - sd, 0):x]
    # right top->bottom
    roi2 = img_gray[max(y - sd, 0):min(y + text_h + sd, img_h), x + text_w:min(x + text_w + sd, img_w)]
    # top x->text_w
    roi3 = img_gray[max(y - sd, 0):y, x:x + text_w]
    # bottom x->text_w
    roi4 = img_gray[y + text_h:min(y + text_h + sd, img_h), x:x + text_w]
    roi1_flat, roi2_flat, roi3_flat, roi4_flat = roi1.ravel(), roi2.ravel(), roi3.ravel(), roi4.ravel()
    len_roi1, len_roi2, len_roi3, len_roi4 = len(roi1_flat), len(roi2_flat), len(roi3_flat), len(roi4_flat)
    if len_roi1 < 1 or len_roi2 < 1 or len_roi3 < 1 or len_roi4 < 1:
        return None, None
    # 计算合并后数组的长度
    r1gt200 = np.count_nonzero(roi1_flat > 200)
    r2gt200 = np.count_nonzero(roi2_flat > 200)
    r3gt200 = np.count_nonzero(roi3_flat > 200)
    r4gt200 = np.count_nonzero(roi4_flat > 200)
    pc1, pc2, pc3, pc4 = r1gt200 / len_roi1, r2gt200 / len_roi2, r3gt200 / len_roi3, r4gt200 / len_roi4
    pall = pc1 + pc2 + pc3 + pc4
    gt9 = 0
    if pc1 >= 0.9:
        gt9 += 1
    if pc2 >= 0.9:
        gt9 += 1
    if pc3 >= 0.9:
        gt9 += 1
    if pc4 >= 0.9:
        gt9 += 1
    return gt9, pall


#  scale text_bock  deletel outer balck area
# In order to keep it simple and clear, the code has not been merged.
def clear_outerwhite(x, y, text_w, text_h, new_mask_thresh):
    # ===================left
    n, dis, start = 0, 0, max(x - 1, 0)
    while n < text_w // 3:
        n += 1
        start += 1
        pxpoint = new_mask_thresh[y:y + text_h, start:start + 1]
        pe = np.count_nonzero(pxpoint == 0)
        top, bot = pxpoint.size * 0.98, pxpoint.size * 0.02
        if pe >= top or pe <= bot:
            dis += 1
            new_mask_thresh[y:y + text_h, start:start + 1] = 0
        else:
            break
    x += dis
    text_w -= dis
    ## ==================right
    n, dis, start = 0, 0, x + text_w + 1
    while n < text_w // 3:
        n += 1
        start -= 1
        pxpoint = new_mask_thresh[y:y + text_h, start - 1:start]
        pe = np.count_nonzero(pxpoint == 0)
        top, bot = pxpoint.size * 0.98, pxpoint.size * 0.02
        if pe >= top or pe <= bot:
            dis += 1
            new_mask_thresh[y:y + text_h, start - 1:start] = 0
        else:
            break
    text_w -= dis
    # ======================top
    n, dis, start = 0, 0, max(y - 1, 0)
    while n < text_h // 3:
        n += 1
        start += 1
        pxpoint = new_mask_thresh[start:start + 1, x:x + text_w]
        pe = np.count_nonzero(pxpoint == 0)
        top, bot = pxpoint.size * 0.98, pxpoint.size * 0.02
        if pe >= top or pe <= bot:
            dis += 1
            new_mask_thresh[start:start + 1, x:x + text_w] = 0
        else:
            break
    y += dis
    text_h -= dis
    # ======================bottom
    n, dis, start = 0, 0, y + text_h + 1
    while n < text_h // 3:
        n += 1
        start -= 1
        pxpoint = new_mask_thresh[start - 1:start, x:x + text_w]
        pe = np.count_nonzero(pxpoint == 0)
        top, bot = pxpoint.size * 0.98, pxpoint.size * 0.02
        if pe >= top or pe <= bot:
            dis += 1
            new_mask_thresh[start - 1:start, x:x + text_w] = 0
        else:
            break
    text_h -= dis
    return x, y, text_w, text_h


# img_gray a text_block to inner white point percent
def getpercent(img_gray, x, y, w, h, pos='lt'):
    points = {
        "all": img_gray[y:y + h, x:x + w],
        "lt": img_gray[y:y + 50, x:x + 50],
        "rt": img_gray[y:y + 50, x + w - 50:x + w],
        "rb": img_gray[y + h - 50:y + h, x + w - 50:x + w],
        "lb": img_gray[y + h - 50:y + h, x:x + 50]
    }
    if pos in points:
        roi1 = points[pos].ravel()
        return np.count_nonzero(roi1 > 200) / len(roi1)
    return None


# outer rect offset 15 + 15 angle , In order to keep it simple and clear, the code has not been merged.
def rect_offset(rawx, rawy, text_w, text_h, img_gray):
    img_h, img_w = img_gray.shape[:2]
    numbers, exceptpos, total_ok, offset = 0, '', 0, 15
    text_block_percent=getpercent(img_gray, rawx, rawy, text_w, text_h, 'all')
    if text_block_percent < 0.45:
        return False
    while numbers < 2:
        # lt
        if exceptpos != 'lt' and rawy - offset >= 0 and rawx - offset >= 0:
            x, y = rawx, rawy
            roi1 = img_gray[y - 15:y + 15, x - 15:x].ravel()
            roi1_1 = img_gray[y - 15:y, x:x + 15].ravel()
            percent = 0 if len(roi1) < 1 else np.count_nonzero(roi1 > 200) / len(roi1)
            percent_1 = 0 if len(roi1_1) < 1 else np.count_nonzero(roi1_1 > 200) / len(roi1_1)
            if percent > 0.9 and percent_1 > 0.9:
                total_ok += 1
                exceptpos = 'lt'
        # rt
        if exceptpos != 'rt' and rawy - offset >= 0 and rawx + text_w + offset <= img_w:
            x, y = rawx + text_w, rawy
            roi1 = img_gray[y - 15:y + 15, x:x + 15].ravel()
            roi1_1 = img_gray[y - 15:y, x - 15:x].ravel()
            percent = 0 if len(roi1) < 1 else np.count_nonzero(roi1 > 200) / len(roi1)
            percent_1 = 0 if len(roi1_1) < 1 else np.count_nonzero(roi1_1 > 200) / len(roi1_1)
            if percent > 0.9 and percent_1 > 0.9:
                total_ok += 1
                exceptpos = 'rt'
        if total_ok > 1:
            return True
        # rb
        if exceptpos != 'rb' and rawy + text_h + offset <= img_h and rawx + text_w + offset <= img_w:
            x, y = rawx + text_w, rawy + text_h
            roi1 = img_gray[y - 15:y + 15, x:x + 15].ravel()
            roi1_1 = img_gray[y:y + 15, x - 15:x].ravel()
            percent = 0 if len(roi1) < 1 else np.count_nonzero(roi1 > 200) / len(roi1)
            percent_1 = 0 if len(roi1_1) < 1 else np.count_nonzero(roi1_1 > 200) / len(roi1_1)
            if percent > 0.9 and percent_1 > 0.9:
                total_ok += 1
                exceptpos = 'rb'
        if total_ok > 1:
            return True
        # lb
        if exceptpos != 'lb' and rawy + text_h + offset <= img_h and rawx - offset >= 0:
            x, y = rawx, rawy + text_h
            roi1 = img_gray[y - 15:y + 15, x - 15:x].ravel()
            roi1_1 = img_gray[y:y + 15, x:x + 15].ravel()
            percent = 0 if len(roi1) < 1 else np.count_nonzero(roi1 > 200) / len(roi1)
            percent_1 = 0 if len(roi1_1) < 1 else np.count_nonzero(roi1_1 > 200) / len(roi1_1)
            if percent > 0.9 and percent_1 > 0.9:
                total_ok += 1
                exceptpos = 'lb'
        if total_ok > 1:
            return True
        offset = 8
        numbers += 1
    return False


# is bubble
def is_bubble(img: np.ndarray, x: int, y: int, text_w: int, text_h: int):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    mask[y:y + text_h, x:x + text_w] = 255
    _, new_mask_thresh = cv2.threshold(cv2.bitwise_and(img_gray, mask), 127, 255, cv2.THRESH_BINARY_INV)
    new_mask_thresh[0:y, :] = 0
    new_mask_thresh[y + text_h:, :] = 0
    new_mask_thresh[y:y + text_h, 0:x] = 0
    new_mask_thresh[y:y + text_h, x + text_w:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # kernel
    new_mask_thresh = cv2.morphologyEx(new_mask_thresh, cv2.MORPH_CLOSE, kernel)  # close
    new_mask_thresh = cv2.dilate(new_mask_thresh, kernel)
    new_mask_thresh = cv2.erode(new_mask_thresh, kernel)
    # text_block new position
    x, y, text_w, text_h = clear_outerwhite(x, y, text_w, text_h, new_mask_thresh)
    # sd add to 10
    gt9, pall = offset_margin(x, y, text_w, text_h, img_gray, 10)
    if gt9 == None and pall == None:
        cv2.rectangle(new_mask_thresh, (x, y), (x + text_w, y + text_h), (0, 0, 0), -1)
        return False, new_mask_thresh
    checkset = [3.2, 2.9]
    # gt9:1-4。pall:0.0-4.0，
    if gt9 >= 3 or (gt9 >= 1 and pall >= checkset[0]) or (gt9 <= 1 and pall < 1.2):
        # sd add to 20
        gt9, pall = offset_margin(x, y, text_w, text_h, img_gray, 20)
        if gt9 >= 3 or pall >= checkset[1] or pall <= 1.5:
            return True, new_mask_thresh
    # four top to outer
    if rect_offset(x, y, text_w, text_h, img_gray):
        return True, new_mask_thresh
    cv2.rectangle(new_mask_thresh, (x, y), (x + text_w, y + text_h), (0, 0, 0), -1)
    return False, new_mask_thresh


#  return new text_regions
def handel(img_bbox_raw, text_regions_raw, verbose=False):
    img_bbox_raw = cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR)
    text_regions, mask = [], None
    for j, blk in enumerate(text_regions_raw):
        bx1, by1, bx2, by2 = blk.xyxy
        res, mask = is_bubble(img_bbox_raw, bx1, by1, bx2 - bx1, by2 - by1)
        if res:
            text_regions.append(text_regions_raw[j])
            if verbose:
                img_rgb = np.copy(img_bbox_raw)
                cv2.rectangle(img_rgb, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.imwrite(f'result/img_rgb-{j}-x{bx1}-y{by1}.jpg', img_rgb)
                cv2.imwrite(f'result/mask-{j}-x{bx1}-y{by1}_thresh.png', mask)
    return text_regions
