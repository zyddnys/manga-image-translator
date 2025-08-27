import numpy as np
import cv2

def check_color(image):
    """
    Determine whether there are colors in non-black, gray, white, and other gray areas in an RGB color image.
    params：
    image -- np.array
    return：
    True -- Colors with non black, gray, white, and other grayscale areas
    False -- Images are all grayscale areas
    """
    # Calculate grayscale version of the image using vectorized operations
    gray_image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    gray_image = gray_image[..., np.newaxis]

    # Calculate color distance for all pixels in a vectorized manner
    color_distance = np.sum((image - gray_image) ** 2, axis=-1)

    # Count the number of pixels where color distance exceeds the threshold
    n = np.sum(color_distance > 100)

    # Return True if there are more than 10 such pixels
    # TODO:
    # Proportion should be used
    return n > 10

def is_ignore(region_img, ignore_bubble = 0):
    """
    Principle: Normally, white bubbles and their text boxes are mostly white, while black bubbles and their text boxes are mostly black. We calculate the ratio of white or black pixels around the text block to the total pixels, and judge whether the area is a normal bubble area or not. Based on the value of the --ignore-bubble parameter, if the ratio is greater than the base value and less than (100-base value), then it is considered a non-bubble area.
    The normal range for ignore-bubble is 1-50, and other values are considered not input. The recommended value for ignore-bubble is 10. The smaller it is, the more likely it is to recognize normal bubbles as image text and skip them. The larger it is, the more likely it is to recognize image text as normal bubbles.

    Assuming ignore-bubble = 10
    The text block is surrounded by white if it is <10, and the text block is very likely to be a normal white bubble.
    The text block is surrounded by black if it is >90, and the text block is very likely to be a normal black bubble.
    Between 10 and 90, if there are black and white spots around it, the text block is very likely not a normal bubble, but an image.

    The input parameter is the image data of the text block processed by OCR.
    Calculate the ratio of black or white pixels in the four rectangular areas formed by taking 2 pixels from the edges of the four sides of the image.
    Return the overall ratio. If it is between ignore_bubble and (100-ignore_bubble), skip it.

    last determine if there is color, consider the colored text as invalid information and skip it without translation
    """
    # Current issues with bubble detection:
    # 1. Misjudgment of solid color backgrounds (core issue):
    # Reason: The code calculates the black/white pixel ratio in a 2-pixel edge area around the text box. If the text box is on a large solid white background (e.g., black text on white paper), the edges will mostly be white, resulting in a very low ratio (close to 0), which falls below the ignore_bubble threshold. The code then mistakenly considers this as a "normal white bubble background" and fails to ignore it (i.e., it treats it as regular bubble text that needs translation). While this text does require translation, it is not actually bubble text.
    # Fundamental flaw: This method does not detect bubble boundaries or contours; it only checks local background color.
    # 2. Inability to recognize bubble boundaries:
    # Reason: The code does not involve any shape or contour detection. It cannot determine whether there is a closed, relatively uniform-colored line surrounding the text box.
    # Consequence: Unable to distinguish between actual bubbles (with boundaries) and cases where the background color coincidentally meets the ratio criteria.
    # 3. Insensitivity to bubble size and relative position:
    # Reason: Only examines the immediate 2-pixel area, without considering the overall size, shape of the bubble, or the text box's relative position within the bubble.
    # Consequence: Cannot utilize common-sense features like "bubbles typically surround the text box and are moderately sized."
    # 4. Connected bubble issue:
    # Reason: The current logic is entirely based on the local environment of a single text box and cannot detect whether there is a shared bubble structure spanning multiple text boxes.
    # Consequence: Unable to handle cases where a large or complex-shaped bubble contains multiple independent text blocks, nor can it determine which part of the bubble corresponds to which text block.
    if ignore_bubble<1 or ignore_bubble>50:
        return  False
    _, binary_raw_mask = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY)
    height, width = binary_raw_mask.shape[:2]

    total=0
    val0=0

    val0+= sum(binary_raw_mask[0:2, 0:width].ravel() == 0)
    total+= binary_raw_mask[0:2, 0:width].size

    val0+= sum(binary_raw_mask[height-2:height, 0:width].ravel() == 0)
    total+= binary_raw_mask[height-2:height, 0:width].size

    val0+= sum(binary_raw_mask[2:height-2, 0:2].ravel() == 0)
    total+= binary_raw_mask[2:height-2, 0:2].size

    val0+= sum(binary_raw_mask[2:height-2, width-2:width].ravel() == 0)
    total += binary_raw_mask[2:height-2, width-2:width].size

    ratio = round( val0 / total, 6)*100
    # ignore
    if ratio>=ignore_bubble and ratio<=(100-ignore_bubble):
        return True
    # To determine if there is color, consider the colored text as invalid information and skip it without translation
    if check_color(region_img):
        return True
    return False

