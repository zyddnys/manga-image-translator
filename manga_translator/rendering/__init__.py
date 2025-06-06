import os
import cv2
import numpy as np
from typing import List, Optional
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from .text_render_pillow_eng import render_textblock_list_eng as render_textblock_list_eng_pillow
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):  
    """  
    Adjusts the size of text regions to accommodate font size and translated text length. 

    This function adjusts the bounding box of each text region based on the following logic:  
    1. Determine the minimum font size.  
    2. Calculate the target font size based on the original font size, fixed font size, and offset.  
    3. **If the translated text length is greater than the original text, calculate a scaling factor based on the length ratio and constrain it between 1.1 and 1.4 times.**  
    4. **Use the calculated scaling factor (if applied) and target font size to adjust the size of the original bounding box.**  
    5. Clip the adjusted bounding box to within the image boundaries.  
    6. Update the font size of the TextBlock object.  

    Args:  
        img (np.ndarray): The input image.  
        text_regions (List[TextBlock]): A list of TextBlock objects to process.  
        font_size_fixed (int): Fixed font size (if provided, other font size parameters are ignored).  
        font_size_offset (int): Font size offset.  
        font_size_minimum (int): Minimum font size. If -1, it's automatically calculated based on image dimensions.  

    Returns:  
        List[np.ndarray]: A list of adjusted text region bounding boxes, where each bounding box is a (4, 2) NumPy array.  
    """  
    # 1. Determine the minimum font size  
    if font_size_minimum == -1:  
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)  
    # logger.debug(f'font_size_minimum {font_size_minimum}')  
    font_size_minimum = max(1, font_size_minimum) # Ensure minimum font size is at least 1  

    dst_points_list = []  
    for region in text_regions:  
        # Store the original font size for the region  
        original_region_font_size = region.font_size  
        # Ensure the original font size is valid  
        if original_region_font_size <= 0:  
            # logger.warning(f"Invalid original font size ({original_region_font_size}) for text '{region.translation[:10]}'. Using default value {font_size_minimum}.")  
            original_region_font_size = font_size_minimum # Use minimum as default  

        # 2. Determine the target font size  
        current_base_font_size = original_region_font_size  
        if font_size_fixed is not None:  
            target_font_size = font_size_fixed  
        else:  
            # Apply the offset to the original font size  
            target_font_size = current_base_font_size + font_size_offset  

        # Apply the minimum font size constraint  
        target_font_size = max(target_font_size, font_size_minimum)  
        # Ensure font size is at least 1  
        target_font_size = max(1, target_font_size)  
        # logger.debug(f"Calculated target font size: {target_font_size} for text '{region.translation[:10]}'")  

        # 3. Calculate a scaling factor based on text length ratio  
        #char_count_orig = len(region.text.strip())  
        orig_text = getattr(region, "text_raw", region.text)  # Fallback to existing text if text_raw is not saved  
        char_count_orig = len(orig_text.strip())  
        char_count_trans = len(region.translation.strip())  
        length_ratio = 1.0 # Default scaling factor is 1.0  

        if char_count_orig > 0 and char_count_trans > char_count_orig:  
             # Translated text is longer, calculate length ratio  
            length_ratio = char_count_trans / char_count_orig  
            # logger.debug(f"Text length ratio: {length_ratio:.2f} ({char_count_trans} / {char_count_orig}) for text '{region.translation}'")  
            # Constrain the scaling factor between 1.1 and 1.4 times  
            target_scale = max(1.1, min(length_ratio, 1.4))  
            # logger.debug(f"Applying length ratio scaling, target scale (constrained): {target_scale:.2f}")  
        else:  
            # Translated text is not longer than original, do not apply length ratio scaling, only consider font size adjustment  
            target_scale = 1.1  # Cannot be 1, sometimes font size shrinks even if shorter than original, There is still logic to shrink the font size somewhere else..  
            # print("-" * 50)  
            # logger.debug(f"Translated text is not longer than original ({char_count_trans} <= {char_count_orig}) or original length is 0, no length ratio scaling applied. Target scale: {target_scale:.2f}")  


        # 4. Calculate the final scaling factor based on target font size and length ratio (if applied)  
        # We need a scaling factor to adjust the original bounding box to accommodate the new font size and potentially longer text.  
        # A simple approach is to combine the font size change and length ratio.  
        # If original font size is valid and different from target, first consider font size scaling.  
        font_size_scale = target_font_size / original_region_font_size if original_region_font_size > 0 else 1.0  
        # If length ratio scaling was applied, take the maximum of font size scaling and length ratio scaling.  
        # This ensures the region can accommodate at least the longer text or larger font.  
        final_scale = max(font_size_scale, target_scale) # Use the previously calculated target_scale (considering length ratio)  
        # Ensure the final scaling factor is at least 1.0  
        final_scale = max(1.0, final_scale)  

        # logger.debug(f"Font size scaling factor: {font_size_scale:.2f}")  
        # logger.debug(f"Final bounding box scaling factor: {final_scale:.2f}")  


        # 5. Scale the bounding box, rotate it back, and clip  
        if final_scale > 1.001: # Apply scaling only if it's significantly greater than 1  
            # logger.debug(f"Scaling bounding box required: text='{region.translation}', scale={final_scale:.2f}")  
            try:  
                # Use unrotated_min_rect for scaling  
                poly = Polygon(region.unrotated_min_rect[0])  
                # Scale from the center  
                poly = affinity.scale(poly, xfact=final_scale, yfact=final_scale, origin='center')  
                scaled_unrotated_points = np.array(poly.exterior.coords[:4])  

                # Rotate the scaled points back to the original orientation  
                # Use to_int=False to preserve precision for clipping  
                dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)  

                # Clip coordinates to within the image boundaries  
                # Use img.shape[1]-1 and img.shape[0]-1 to avoid off-by-one issues  
                dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)  
                dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)  

                # Convert to int64 after clipping  
                dst_points = dst_points.astype(np.int64)  

                # Reshape to ensure correct final shape (just in case)  
                dst_points = dst_points.reshape((-1, 4, 2))  
                # logger.debug(f"Finished calculating scaled dst_points.")  

            except Exception as e:  
                # If an error occurs during scaling/rotating the geometric shape, use the original min_rect  
                # logger.error(f"Error during scaling/rotating geometric shape for text '{region.translation}': {e}. Using original min_rect.")  
                dst_points = region.min_rect # Use original value on error  
        else:  
            # No significant scaling needed, use the original min_rect  
            # logger.debug(f"No significant scaling needed for text '{region.translation}'. Using original min_rect.")  
            dst_points = region.min_rect  

        # 6. Store the final dst_points and update the region's font size  
        dst_points_list.append(dst_points)  
        region.font_size = int(target_font_size) # Update the TextBlock's font size to the calculated target font size  

    return dst_points_list  

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False,
    upscale_ratio: Optional[int] = None
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border, upscale_ratio)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border,
    upscale_ratio: Optional[int] = None
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    # 如果配置中设定了非自动模式，则直接使用配置决定方向
    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    #print(f"Region text: {region.text}, forced_direction: {forced_direction}, render_horizontally: {render_horizontally}")

    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
        )
    h, w, _ = temp_box.shape
    r_temp = w / h

    # Extend temporary box so that it has same ratio as original
    box = None  
    #print("\n" + "="*50)  
    #print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    #print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    #print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    #print(f"Target language: {region.target_lang}")      
    #print(f"Region horizontal: {region.horizontal}")  
    #print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    if region.horizontal:  
        #print("Processing HORIZONTAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:  
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [h_ext:h_ext+h, :w] = [{h_ext}:{h_ext+h}, 0:{w}]")  
                # 列已排满，行居中
                # Columns fully filled, rows centered
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()  
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) // 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [:, :w] = [0:{h}, 0:{w}]")  
         
                # 行已排满，文字左侧不留空列，否则当存在多个文本框的左边线处于一条线上时译后文本无法对齐，搭配左对齐更美观。常见场景：无框漫画、漫画后记    
                # 当前页面存在气泡时则可改为居中：box[0:h, w_ext:w_ext+w] = temp_box，需更准确的气泡检测。但不改也没太大影响。                
                # The line is full, and there should be no empty columns on the left side of the text. Otherwise, when multiple text boxes are aligned on the left, the translated text cannot be aligned. Common scenarios: borderless comics, comic postscript.  
                # When there are bubbles on the current page, it can be changed to center: box[0:h, w_ext:w_ext+w] = temp_box, requiring more accurate bubble detection. But not changing it doesn't have much impact.
                box[0:h, 0:w] = temp_box  
            else:  
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()  
    else:  
        #print("Processing VERTICAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0   
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:   
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, 0:w] = [0:{h}, 0:{w}]")  
                # 列已排满，文字的上方不留空行，否则当存在多个文本框的上边线在一条线上时文本无法对齐，常见场景：无框漫画、CG
                # 当前页面存在气泡时则可改为居中：box[h_ext:h_ext+h, 0:w] = temp_box，需更准确的气泡检测。
                # The rows are full, and there should be no empty lines above the text; otherwise, when multiple text boxes have their top edges aligned, the text cannot be aligned. Common scenario: borderless comics, CG. 
                # When there are bubbles on the current page, it can be changed to center: box[h_ext:h_ext+h, 0:w] = temp_box, requiring more accurate bubble detection.
                box[0:h, 0:w] = temp_box  
            else:   
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()   
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) / 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, w_ext:w_ext+w] = [0:{h}, {w_ext}:{w_ext+w}]") 
                # 行已排满，列居中                
                # Rows are fully filled, columns are centered
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:   
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()   
    #print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # 当开启了upscaler且upscale_ratio不为空时使用线性插值
    interpolation = cv2.INTER_LINEAR if upscale_ratio is not None else cv2.INTER_NEAREST
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)

async def dispatch_eng_render_pillow(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/NotoSansMonoCJK-VF.ttf.ttc')
    text_render.set_font(font_path)

    return render_textblock_list_eng_pillow(font_path, img_canvas, text_regions, original_img=original_img, downscale_constraint=0.95)
