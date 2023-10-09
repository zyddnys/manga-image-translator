from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from PIL import Image

KERNEL_SIZE = 7
BORDER_SIZE = 10


def panel_process_image(img: Image.Image):
    """Preprocesses an image to make it easier to find panels.

    Args:
        img: The image to preprocess.

    Returns:
        The preprocessed image.
    """

    img_gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (KERNEL_SIZE, KERNEL_SIZE), 0)
    img_gray = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY)[1]

    # Add black border to image, to help with finding contours
    img_gray = cv.copyMakeBorder(
        img_gray,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv.BORDER_CONSTANT,
        value=255,
    )
    # Invert image
    img_gray = cv.bitwise_not(img_gray)
    return img_gray


def remove_contained_contours(polygons):
    """Removes polygons from a list if any completely contain the other.

    Args:
        polygons: A list of polygons.

    Returns:
        A list of polygons with any contained polygons removed.
    """

    # Create a new list to store the filtered polygons.
    filtered_polygons = []

    # Iterate over the polygons.
    for polygon in polygons:
        # Check if the polygon contains any of the other polygons.
        contains = False
        for other_polygon in polygons:
            # Check if the polygon contains the other polygon and that the polygons
            if np.array_equal(other_polygon, polygon):
                continue
            rect1 = cv.boundingRect(other_polygon)
            rect2 = cv.boundingRect(polygon)
            # Check if rect2 is completely within rect1
            if (
                rect2[0] >= rect1[0]
                and rect2[1] >= rect1[1]
                and rect2[0] + rect2[2] <= rect1[0] + rect1[2]
                and rect2[1] + rect2[3] <= rect1[1] + rect1[3]
            ):
                contains = True
                break

        # If the polygon does not contain any of the other polygons, add it to the
        # filtered list.
        if not contains:
            filtered_polygons.append(polygon)

    return filtered_polygons


def calc_panel_contours(im: Image.Image):
    img_gray = panel_process_image(im)
    contours_raw, hierarchy = cv.findContours(
        img_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours = contours_raw
    min_area = 10000
    contours = [i for i in contours if cv.contourArea(i) > min_area]
    contours = [cv.convexHull(i) for i in contours]
    contours = remove_contained_contours(contours)

    # Remap the contours to the original image
    contours = [i + np.array([[-BORDER_SIZE, -BORDER_SIZE]]) for i in contours]

    # Sort the contours by their y-coordinate.
    contours = order_panels(contours, img_gray)
    return contours


def determine_panel_order_from_contours(contours):
    """
    build a tree of regions that are determined vertically
    order by an n like pattern
    """


def draw_contours(im, contours):
    """Debugging, draws the contours on the image."""
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    im_contour = np.array(im)

    for i, contour in enumerate(range(len(contours))):
        color = colors[i % len(colors)]
        im_contour = cv.drawContours(im_contour, contours, i, color, 4, cv.LINE_AA)
        # Draw a number at the top left of contour
        x, y, _, _ = cv.boundingRect(contours[i])
        cv.putText(
            im_contour,
            str(i),
            (x + 50, y + 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv.LINE_AA,
        )

    img = Image.fromarray(im_contour)

    return img


def save_draw_contours(pth: Path | str):
    if str:
        pth = Path(pth)
    pth_out = pth.parent / (pth.stem + "-contours")

    if not pth_out.exists():
        pth_out.mkdir()

    # Glob get all images in folder

    pths = [i for i in pth.iterdir() if i.suffix in [".png", ".jpg", ".jpeg"]]
    for t in pths:
        print(t)
        im = Image.open(t)
        contours = calc_panel_contours(im)

        img_panels = draw_contours(im, contours)
        f_name = t.stem + t.suffix
        img_panels.save(pth_out / f_name)


def order_panels(contours, img_gray):
    """Orders the panels in a comic book page.

    Args:
      contours: A list of contours, where each contour is a list of points.

    Returns:
      A list of contours, where each contour is a list of points, ordered by
      their vertical position.
    """

    # Get the bounding boxes for each contour.
    bounding_boxes = [cv.boundingRect(contour) for contour in contours]

    # Generate groups of vertically overlapping bounding boxes.
    groups_indices = generate_vertical_bounding_box_groups_indices(bounding_boxes)

    c = []

    for group in groups_indices:
        # Reorder contours based on reverse z-order,

        cs = [bounding_boxes[i] for i in group]
        ymax, xmax = img_gray.shape
        order_scores = [1 * (ymax - i[1]) + i[0] * 1 for i in cs]

        # Sort the list based on the location score value
        combined_list = list(zip(group, order_scores))
        sorted_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
        c.extend(sorted_list)

    ordered_contours = [contours[i[0]] for i in c]
    return ordered_contours


def generate_vertical_bounding_box_groups_indices(bounding_boxes):
    """Generates groups of vertically overlapping bounding boxes.

    Args:
      bounding_boxes: A list of bounding boxes, where each bounding box is a tuple
        of (x, y, width, height).

    Returns:
      A list of groups, where each group is a list of bounding boxes that overlap
      vertically.
    """

    # Operate on indices Sort the bounding boxes by their y-coordinate.

    bbox_inds = np.argsort([i[1] for i in bounding_boxes])

    # generate groups of vertically overlapping bounding boxes
    groups = [[bbox_inds[0]]]
    for i in bbox_inds[1:]:
        is_old_group = False
        bbox = bounding_boxes[i]
        start1 = bbox[1]
        end1 = bbox[1] + bbox[3]
        for n, group in enumerate(groups):
            for ind in group:
                _bbox = bounding_boxes[ind]
                start2 = _bbox[1]
                end2 = _bbox[1] + _bbox[3]

                # Check for any partial overlapping
                if check_overlap((start1, end1), (start2, end2)):
                    groups[n] = group + [i]
                    is_old_group = True
                    break

            if is_old_group:
                break
        else:
            groups.append([i])
    return groups


def check_overlap(range1, range2):
    # Check if range1 is before range2
    if range1[1] < range2[0]:
        return False
    # Check if range1 is after range2
    elif range1[0] > range2[1]:
        return False
    # If neither of the above conditions are met, the ranges must overlap
    else:
        return True


if __name__ == "__main__":
    save_draw_contours(sys.argv[1])
