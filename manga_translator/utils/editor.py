import os
import sys
import json
import asyncio
import websockets
import pathlib
import platform
import subprocess
import threading
import queue
import cv2
import freetype
import glob
from pathlib import Path
from typing import List, Optional, Tuple


import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import colorchooser, filedialog, ttk

from ..config import Config
from ..rendering import text_render
from ..rendering.ballon_extractor import extract_ballon_region
from .textblock import TextBlock
from ..utils import BASE_PATH, rect_distance
import bisect
from ..utils import natural_sort  # Use the project's existing natural sort utility if available
import re

import subprocess

async def stream_context_to_editor_server(ctx_dict, dest_path, config, host="127.0.0.1", port=5003):
    """
    Serializes contextual text arrays, processes tracking tags, and packages 
    compressed binary buffers for zero-loss matrix transmission to the Editor.
    """
    uri = f"ws://{host}:{port}/ws/ipc/append_image"
    
    # 1. Extract heavy image array matrices out of the dictionary block
    img_rgb = ctx_dict.get('img_rgb', None)
    img_inpainted = ctx_dict.get('img_inpainted', None)
    
    # 2. Transform custom internal descriptors recursively down into pure dict primitives
    def sanitize(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            if obj.size <= 32:
                return obj.tolist()
            return None
        elif hasattr(obj, '__dict__'):
            data_dict = {}
            for k, v in obj.__dict__.items():
                data_dict[k] = sanitize(v)
            for attr_name in ('xyxy', 'xywh', 'bg_colors', 'fg_colors'):
                if hasattr(obj, attr_name) and attr_name not in data_dict:
                    val = getattr(obj, attr_name)
                    if hasattr(val, 'tolist'):
                        data_dict[attr_name] = val.tolist()
                    elif hasattr(val, 'cpu'):
                        data_dict[attr_name] = val.cpu().numpy().tolist()
                    else:
                        data_dict[attr_name] = sanitize(val)
            return data_dict
        else:
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return str(obj)

    # 3. Compress spatial image data profiles to optimize transmission efficiency
    rgb_bytes = b""
    if img_rgb is not None:
        success, encoded = cv2.imencode('.png', img_rgb)
        if success:
            rgb_bytes = encoded.tobytes()

    inpainted_bytes = b""
    if img_inpainted is not None:
        success, encoded = cv2.imencode('.png', img_inpainted)
        if success:
            inpainted_bytes = encoded.tobytes()

    # 4. FIX: Sanitize the config object down to serializable base primitives
    payload_metadata = {
        "dest": str(dest_path),
        "ctx": sanitize(ctx_dict),
        "config": sanitize(config),
        "has_rgb": img_rgb is not None,
        "has_inpainted": img_inpainted is not None
    }
    
    metadata_json_bytes = json.dumps(payload_metadata).encode('utf-8')
    
    # 5. Pack binary frames via custom layout headers to bypass pickling boundaries
    json_len = len(metadata_json_bytes)
    rgb_len = len(rgb_bytes)
    
    header = json_len.to_bytes(4, byteorder='big') + rgb_len.to_bytes(4, byteorder='big')
    full_binary_packet = header + metadata_json_bytes + rgb_bytes + inpainted_bytes

    # 6. Stripped multi-gpu and gpu_id routing channels out cleanly
    try:
        async with websockets.connect(uri, max_size=None) as websocket:
            await websocket.send(full_binary_packet)
            await websocket.recv()  # Await acknowledgment response block
    except ConnectionRefusedError:
        pass
    except Exception as e:
        print(f"[IPC Emitter Error] Failed to stream multi-part chunk payload: {e}", file=sys.stderr)

def spawn_detached_editor_ui(initial_images=None):
    """Runs the Tkinter event loop in a dedicated GUI thread alongside the IPC server."""
    # Open local loopback interface port listener 
    server_thread = threading.Thread(
        target=run_async_server_loop, 
        args=("127.0.0.1", 5003), 
        daemon=True,
        name="MangaTranslator-IPC-Server"
    )
    server_thread.start()

    root = tk.Tk()
    root.title("Live Stream Editor")
    root.geometry("1280x720")

    root.overrideredirect(True)
    root.overrideredirect(False)

    #root.attributes("-fullscreen", True)
    #root.wm_attributes("-topmost", 1)
    root.focus_set()

    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))#lambda event:root.destroy())
    root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))

    editor_widget = LiveEditor(root, initial_image_dict=initial_images)
    editor_widget.pack(fill="both", expand=True)
    editor_widget.register_close_protocol()

    # Map the live widget reference so the background socket server can pass data frames directly into it
    register_editor_instance(editor_widget)
    
    root.mainloop()

    subprocess.run(["killall", "python"], shell=True)
    # Clean up loop tasks when the Tkinter window workspace is destroyed by the user
    if _SERVER_LOOP is not None:
        _SERVER_LOOP.call_soon_threadsafe(_SERVER_LOOP.stop)
        
    shutdown_event.set()

def natural_sort_key(s: str):
    """Generates an alphanumeric sorting key split (e.g., 'page_10' comes after 'page_2')"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

# --- Global Registries ---
_ACTIVE_EDITOR_WIDGET = None
_SERVER_LOOP = None
_RESOLVED_FONT_PATH = None

def register_editor_instance(instance):
    """Binds the active Tkinter editor widget to our WebSocket IPC server."""
    global _ACTIVE_EDITOR_WIDGET
    _ACTIVE_EDITOR_WIDGET = instance

def discover_global_font_path() -> str:
    """
    Dynamically tracks the target font asset across command line arguments,
    JSON config declarations, and system defaults.
    """
    global _RESOLVED_FONT_PATH
    if _RESOLVED_FONT_PATH is not None:
        return _RESOLVED_FONT_PATH

    # 1. Fallback default font token specification
    fallback_font = "fonts/Arial-Unicode-Regular.ttf"
    
    # 2. Check Command Line Arguments directly
    if '--font-path' in sys.argv:
        try:
            idx = sys.argv.index('--font-path')
            if idx + 1 < len(sys.argv):
                target = sys.argv[idx + 1]
                if os.path.exists(target):
                    _RESOLVED_FONT_PATH = target
                    return _RESOLVED_FONT_PATH
        except Exception:
            pass

    # 3. Check JSON Configuration Mapping profiles
    config_file_path = None
    if '--config-file' in sys.argv:
        try:
            idx = sys.argv.index('--config-file')
            if idx + 1 < len(sys.argv):
                config_file_path = sys.argv[idx + 1]
        except Exception:
            pass
            
    if config_file_path and os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)
                # Parse standard manga-translator JSON schema nesting layouts
                font_opt = cfg_data.get('font_path') or cfg_data.get('render', {}).get('gimp_font')
                if font_opt and os.path.exists(str(font_opt)):
                    _RESOLVED_FONT_PATH = str(font_opt)
                    return _RESOLVED_FONT_PATH
        except Exception:
            pass

    # 4. Enforce system default path resolution
    _RESOLVED_FONT_PATH = fallback_font if os.path.exists(fallback_font) else "Arial"
    return _RESOLVED_FONT_PATH

def get_system_and_fallback_fonts():
    """
    Scans typical system directories and internal configurations to build 
    a dictionary of available { "Font Name": "/absolute/path/to/font.ttf" }.
    """
    font_map = {}
    
    # 1. Register explicitly configured application fallbacks first
    fallback_dir = "fonts"
    if os.path.isdir(fallback_dir):
        for f_path in glob.glob(os.path.join(fallback_dir, "*.ttf")):
            name = os.path.splitext(os.path.basename(f_path))[0]
            font_map[f"{name} (App Fallback)"] = f_path.replace('\\', '/')

    # 2. Fast OS-Agnostic System Directory Scanning Paths
    search_paths = [
        "/usr/share/fonts",                          # Linux standard system fonts path
        "/usr/local/share/fonts",                    # Custom user linux paths
        "~/.fonts",                                  # Local user linux path profile
        "/System/Library/Fonts",                     # macOS base framework fonts directory
        "/Library/Fonts",                            # macOS global applications font space
        "~/Library/Fonts",                           # macOS local user profile fonts path
        os.path.expandvars(r"%WINDIR%\Fonts"),       # Windows native operating system fonts layout
    ]

    for raw_path in search_paths:
        resolved_path = os.path.expanduser(raw_path)
        if os.path.isdir(resolved_path):
            for ext in ('*.ttf', '*.otf', '*/*.ttf', '*/*.otf'):
                for f_path in glob.glob(os.path.join(resolved_path, ext)):
                    base_name = os.path.splitext(os.path.basename(f_path))[0]
                    if base_name not in font_map:
                        font_map[base_name] = f_path.replace('\\', '/')

    if not font_map:
        font_map["Arial-Unicode-Regular"] = "fonts/Arial-Unicode-Regular.ttf"
        
    return font_map

class ReconstitutedNamespace:
    """
    Dynamically wraps dictionaries into dot-notation lookups (e.g., r.translation).
    Enforces clean, guaranteed 3-element integer arrays for color attributes
    even when accessed directly via dot notation (e.g., region.fg_colors).
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ReconstitutedNamespace(value))
            elif isinstance(value, list):
                if key in ('xyxy', 'xywh') and value:
                    setattr(self, key, np.array(value, dtype=np.float32))
                elif key in ('bg_colors', 'fg_colors') and value:
                    # Parse color arrays immediately on initialization
                    setattr(self, key, self._normalize_color(value))
                else:
                    setattr(self, key, [ReconstitutedNamespace(i) if isinstance(i, dict) else i for i in value])
            else:
                # Handle raw scalars if passed in dictionary attributes
                if key in ('bg_colors', 'fg_colors'):
                    setattr(self, key, self._normalize_color(value))
                else:
                    setattr(self, key, value)

    def _normalize_color(self, color_val, default_val=None):
        """Helper to safely format any incoming variant into a 3-channel BGR vector."""
        if default_val is None:
            default_val = np.array([0, 0, 0], dtype=np.uint8)
            
        if isinstance(color_val, np.ndarray):
            flat = color_val.flatten()
            if flat.size >= 3:
                return flat[:3].astype(np.uint8)
            elif flat.size > 0:
                return np.array([flat[0], flat[0], flat[0]], dtype=np.uint8)
        elif isinstance(color_val, (list, tuple)):
            if len(color_val) >= 3:
                return np.array(color_val[:3], dtype=np.uint8)
            elif len(color_val) > 0:
                return np.array([color_val[0], color_val[0], color_val[0]], dtype=np.uint8)
        elif isinstance(color_val, (int, float)):
            val = int(color_val)
            return np.array([val, val, val], dtype=np.uint8)
            
        return default_val
                
    def get_font_colors(self):
        """Mimics the original TextRegion class method."""
        # Grab properties directly from internal namespace mapping 
        font_color = self.__dict__.get('fg_colors', np.array([0, 0, 0], dtype=np.uint8))
        stroke_color = self.__dict__.get('bg_colors', np.array([255, 255, 255], dtype=np.uint8))
        return font_color, stroke_color

    def set_fg_colors(self, col):
        """Mimics the original TextRegion class method."""
        # Grab properties directly from internal namespace mapping 
        self.__dict__['fg_colors'] = col #, np.array([0, 0, 0], dtype=np.uint8))
        self.__dict__['font_color'] = col #, np.array([255, 255, 255], dtype=np.uint8))
        return

    def set_bg_colors(self, col):
        self.__dict__['bg_colors'] = col
        self.__dict__['stroke_color'] = col
        return

    def __getattr__(self, name):
        # Fallback vectors to ensure index layout routines never crash if fields are empty
        if name in ('xyxy', 'xywh'):
            return np.zeros((4,), dtype=np.float32)
        elif name == 'fg_colors':
            return np.array([0, 0, 0], dtype=np.uint8)
        elif name == 'bg_colors':
            return np.array([255, 255, 255], dtype=np.uint8)
        return ""

async def pure_websocket_handler(websocket):
    """
    Pure Python WebSocket connection router. Handles extraction 
    and reconstruction of image structures, tracking hardware metrics.
    Defensively enforces unique data isolation to prevent image-repeating bugs.
    """
    try:
        async for packet_bytes in websocket:
            if len(packet_bytes) < 8:
                continue
                
            # 1. Parse custom multi-part offset boundaries 
            json_len = int.from_bytes(packet_bytes[0:4], byteorder='big')
            rgb_len = int.from_bytes(packet_bytes[4:8], byteorder='big')
            
            meta_start = 8
            meta_end = meta_start + json_len
            rgb_start = meta_end
            rgb_end = rgb_start + rgb_len
            inpainted_start = rgb_end
            
            # 2. Extract alphanumeric payload properties
            metadata_payload = json.loads(packet_bytes[meta_start:meta_end].decode('utf-8'))
            path_string = metadata_payload.get("dest")
            config_data = metadata_payload.get("config")

            image_dict = metadata_payload.get("ctx", {})
            if not isinstance(image_dict, dict):
                image_dict = {"raw_data": image_dict}
            
            # 3. Decompress binary frame allocations back to NumPy array grids
            if metadata_payload.get("has_rgb") and rgb_len > 0:
                rgb_buf = np.frombuffer(packet_bytes[rgb_start:rgb_end], dtype=np.uint8)
                image_dict["img_rgb"] = cv2.imdecode(rgb_buf, cv2.IMREAD_COLOR)
            else:
                image_dict["img_rgb"] = None

            if metadata_payload.get("has_inpainted"):
                inpainted_buf = np.frombuffer(packet_bytes[inpainted_start:], dtype=np.uint8)
                image_dict["img_inpainted"] = cv2.imdecode(inpainted_buf, cv2.IMREAD_COLOR)
            else:
                image_dict["img_inpainted"] = None

            # 4. Reconstitute dictionary mappings to support dot-notation properties safely
            raw_regions = image_dict.get('text_regions') or []
            image_dict['text_regions'] = [ReconstitutedNamespace(r) if isinstance(r, dict) else r for r in raw_regions]

            # ----------------------------------------------------
            # WARM-UP FILTER PASSTHROUGH HOOK
            # ----------------------------------------------------
            is_warmup = (
                "warmup" in str(path_string).lower() or 
                config_data is None or
                image_dict.get("img_inpainted") is None or 
                image_dict.get("img_rgb") is None
            )

            if is_warmup:
                print(f"[IPC Network Server] Completed a pre-warming loop lock frame. Discarding placeholder payload safely.")
                await websocket.send(json.dumps({"status": "acknowledged"}))
                continue

            # ----------------------------------------------------
            # ANTI-BLEED DATA CLEANUP HOOK
            # ----------------------------------------------------
            if _ACTIVE_EDITOR_WIDGET is not None:
                try:
                    if hasattr(_ACTIVE_EDITOR_WIDGET, '_tk_overlay_cache'):
                        _ACTIVE_EDITOR_WIDGET._tk_overlay_cache.clear()
                    
                    normalized_path = os.path.normpath(str(path_string))
                    
                    # 5. Hand the real structural payload directly to the registered UI widget
                    _ACTIVE_EDITOR_WIDGET.append_image(image_dict, path=normalized_path, config=config_data)
                except Exception as append_err:
                    print(f"[IPC Server Error] Failed to execute append_image: {append_err}", file=sys.stderr)
            else:
                print(f"[Server Warning] Context received, but no active editor layout is bound.")

            # Send sync loop status validation acknowledgement frame pulse back
            await websocket.send(json.dumps({"status": "acknowledged"}))
            
            # --- MANDATORY MEMORY PURGE ---
            del image_dict
            del packet_bytes
            
    except Exception as e:
        print(f"[IPC Server Error] Stream processing exception encountered: {e}", file=sys.stderr)

async def main_server_async(host, port):
    """Initializes websockets.serve within an active running loop layout block."""
    async with websockets.serve(
        pure_websocket_handler, 
        host, 
        port, 
        max_size=None,
        ping_interval=None
    ) as server:
        print(f"[IPC Network Server] Server actively listening for GPU worker nodes on ws://{host}:{port}")
        await asyncio.Future()  # Keeps the async server run context open indefinitely

def run_async_server_loop(host, port):
    """Runs the asyncio network loop indefinitely inside a background thread context."""
    global _SERVER_LOOP
    _SERVER_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_SERVER_LOOP)
    
    try:
        _SERVER_LOOP.run_until_complete(main_server_async(host, port))
    except Exception as e:
        print(f"[IPC Server Fatal initialization crash Error]: {e}", file=sys.stderr)


image_stream_queue: queue.Queue = queue.Queue()
shutdown_event: threading.Event = threading.Event()
    
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', '"'}

class Textline:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 1 if text else 0
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = f"{self.text}{delimiter}{word}" if self.text else word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = f"{word}{delimiter}{self.text}" if self.text else word
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def render_lines(
    textlines: List[Textline],
    canvas_h: int,
    canvas_w: int,
    font_size: int,
    stroke_width: int,
    line_spacing: float = 0.01,
    fg: Tuple[int, int, int] = (0, 0, 0),
    bg: Tuple[int, int, int] = (255, 255, 255)
) -> Optional[Image.Image]:
    if not textlines:
        return None

    bg_size = stroke_width
    spacing_y = int(font_size * (line_spacing or 0.01))

    max_len = max(line.length for line in textlines)
    calc_w = max_len + (font_size + bg_size) * 2
    calc_h = font_size * len(textlines) + spacing_y * (len(textlines) - 1) + (font_size + bg_size) * 2

    canvas_text = np.zeros((calc_h, calc_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    pen_orig = [font_size + bg_size, font_size + bg_size]

    for line in textlines:
        pen_line = pen_orig.copy()
        pen_line[0] += line.pos_x
        for char in line.text:
            offset_x = text_render.put_char_horizontal(font_size, char, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    
    line_box = text_render.add_color(canvas_text, fg, canvas_border, bg)

    x, y, width, height = cv2.boundingRect(canvas_border)
    if width <= 0 or height <= 0:
        return None
        
    # Enforce integer casts to fix the slice indices TypeError
    x, y, w, h = int(x), int(y), int(width), int(height)
    return Image.fromarray(line_box[y:y + h, x:x + w])

def seg_eng(text: str) -> List[str]:
    text = text.strip().upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = []

    text_len = len(text)
    for i, char in enumerate(text):
        if char in PUNSET_RIGHT_ENG and i < text_len - 1:
            next_c = text[i + 1]
            processed_text.append(f"{char} " if next_c.isalnum() else char)
        else:
            processed_text.append(char)

    word_list = ''.join(processed_text).split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
    for i, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue

        if len(word) < 3:
            append_left, append_right = False, False
            len_word = len(word)
            len_next = len(word_list[i + 1]) if i < word_num - 1 else -1
            len_prev = len(words[-1]) if words else -1

            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1

            if len_next > 0 and len_prev > 0:
                append_right = cond_next if len_next < len_prev else False
                append_left = cond_prev if not append_right else False
            elif len_next > 0:
                append_right = cond_next
            elif len_prev > 0:
                append_left = cond_prev

            if append_left and words:
                words[-1] = f"{words[-1]} {word}"
            elif append_right:
                words.append(f"{word} {word_list[i + 1]}")
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)

    return words

def layout_lines_aligncenter(
    mask: np.ndarray,
    words: List[str],
    word_lengths: List[int],
    delimiter_len: int,
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf
) -> List[Textline]:
    m = cv2.moments(mask)
    if m['m00'] == 0:
        centroid_x, centroid_y = mask.shape[1] // 2, mask.shape[0] // 2
    else:
        centroid_y = int(m['m01'] / m['m00'])
        centroid_x = int(m['m10'] / m['m00'])

    mask = 255 - mask
    bh, bw = mask.shape[:2]
    num_words = len(words)
    wlst_left, wlst_right = [], []
    len_left, len_right = [], []
    sum_left, sum_right = 0, 0

    if num_words > 1:
        wl_array = np.array(word_lengths, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = int(np.argmin(np.abs(wl_cumsums)))

        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = word_lengths[:central_index]
            sum_left = sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = word_lengths[central_index + 1:]
            sum_right = sum(len_right)
    else:
        central_index = 0

    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - word_lengths[central_index] // 2
    central_line = Textline(words[central_index], pos_x, pos_y, word_lengths[central_index], spacing)
    line_bottom = pos_y + line_height

    while sum_left > 0 or sum_right > 0:
        left_valid, right_valid = False, False
        new_x_l, new_x_r = 0, 0

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if 0 < new_x_l and new_r_l < bw:
                if mask[pos_y:line_bottom, new_x_l].sum() == 0 and mask[pos_y:line_bottom, new_r_l].sum() == 0:
                    left_valid = True

        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2
            new_r_r = new_x_r + new_len_r
            if 0 < new_x_r and new_r_r < bw:
                if mask[pos_y:line_bottom, new_x_r].sum() == 0 and mask[pos_y:line_bottom, new_r_r].sum() == 0:
                    right_valid = True

        insert_left = (sum_left > sum_right) if (left_valid and right_valid) else left_valid
        if not (left_valid or right_valid):
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r

        if central_line.length > max_central_width:
            break

    central_line.strip_spacing()
    lines = [central_line]

    # Layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl

        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len

            valid = (0 < new_x and right_x < bw) and (
                mask[pos_y:line_bottom, new_x].sum() == 0 and mask[pos_y:line_bottom, right_x].sum() == 0
            )

            if valid:
                line.append_right(w, wl + delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line.strip_spacing()
                    break
            else:
                line.strip_spacing()
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.append(line)

    # Layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height * 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl

        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len

            valid = (0 < new_x and right_x < bw) and (
                mask[pos_y:line_bottom, new_x].sum() == 0 and mask[pos_y:line_bottom, right_x].sum() == 0
            )

            if valid:
                line.append_left(w, wl + delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line.strip_spacing()
                    break
            else:
                line.strip_spacing()
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)

    return lines

def render_textblock_list_eng_rendered(
    img: np.ndarray,
    text_regions: List[TextBlock],
    delimiter: str = ' ',
    line_spacing: float = 0.01,
    stroke_width: float = 0.1,
    ballonarea_thresh: float = 2.0,
    downscale_constraint: float = 0.7,
    original_img: Optional[np.ndarray] = None,
    disable_font_border: bool = False,
    font_path: str = discover_global_font_path(),
    force_font_size: int = -1, ) -> Tuple[List, List]:

    rendered_text_list = []
    rendered_text_coords = []
    
    #if (fg_colors != [-1,-1,-1]):
    #    text_regions[0].fg_colors = fg_colors
    #if (bg_colors != [-1,-1,-1]):
    #    text_regions[0].bg_colors = bg_colors   

    def calculate_font_values(font_size: int, words: List[str], f_size=None):
        # Ensure your base font size metrics tracking references are resolved cleanly
        if f_size is None:
            f_size = font_size
        # Assuming your original font rendering subsystem uses a global load or set hook:
        # e.g., from manga_translator.rendering.text_render import set_global_font
        text_render.set_font(font_path)

        sw = int(font_size * stroke_width)
        line_height = int(font_size * 0.8)
        #delimiter_glyph = text_render.get_char_glyph(delimiter, font_size, 0)
        #delimiter_len = delimiter_glyph.advance.x >> 6
        delimiter_glyph = text_render.get_char_glyph(' ', f_size, 0)
        if delimiter_glyph is not None and hasattr(delimiter_glyph, 'advance') and delimiter_glyph.advance is not None:
            delimiter_len = delimiter_glyph.advance.x >> 6
        else:
            # Scale spacing safely relative to the target font sizing parameters
            delimiter_len = max(2, int(f_size * 0.25))

        word_pixel_lengths = []
        base_length = -1
        word_lengths = []
        for word in words:
            word_length = 0
            for cdpt in word:
                glyph = text_render.get_char_glyph(cdpt, f_size, 0)
                
                if glyph is not None and hasattr(glyph, 'metrics') and glyph.metrics is not None:
                    char_offset_x = glyph.metrics.horiAdvance >> 6
                    word_pixel_lengths.append(char_offset_x)
                else:
                    char_offset_x = max(2, int(f_size * 0.55))
                    word_pixel_lengths.append(char_offset_x)
                    # Scale spacing safely relative to the target font sizing parameters
                char_offset_x = glyph.metrics.horiAdvance >> 6
                word_length += char_offset_x
            word_lengths.append(word_length)
            if word_length > base_length:
                base_length = word_length
        #w_len = sum(word_pixel_lengths)
        return font_size, sw, line_height, delimiter_len, base_length, word_lengths

    def update_enlarged_xyxy(reg):
        reg.enlarged_xyxy = reg.xyxy.copy()
        w_diff, h_diff = ((reg.xywh[2:] * reg.enlarge_ratio) - reg.xywh[2:].astype(np.float64)) // 2
        reg.enlarged_xyxy[0] -= w_diff
        reg.enlarged_xyxy[2] += w_diff
        reg.enlarged_xyxy[1] -= h_diff
        reg.enlarged_xyxy[3] += h_diff

    for region in text_regions:
        region.enlarge_ratio = 1
        region.enlarged_xyxy = region.xyxy.copy()

    for region in text_regions:
        if region.enlarge_ratio == 1:
            region.enlarge_ratio = min(max(region.xywh[2] / region.xywh[3], region.xywh[3] / region.xywh[2]) * 1.5, 3)
            update_enlarged_xyxy(region)

        for region2 in text_regions:
            if region is region2:
                continue
            if rect_distance(*region.enlarged_xyxy, *region2.enlarged_xyxy) == 0:
                d = rect_distance(*region.xyxy, *region2.xyxy)
                l1 = (region.xywh[2] + region.xywh[3]) / 2
                l2 = (region2.xywh[2] + region2.xywh[3]) / 2
                region.enlarge_ratio = d / (2 * l1) + 1
                region2.enlarge_ratio = d / (2 * l2) + 1
                update_enlarged_xyxy(region)
                update_enlarged_xyxy(region2)

    for region in text_regions:
        words = seg_eng(region.translation)
        if not words:
            continue
        
        font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(
            region.font_size, words
        )
        
        # Enforce that all coordinates inside xywh are strictly pure integers
        safe_xywh = [int(round(float(c))) for c in region.xywh]

        try:
            # Cast enlarge_ratio to float just in case it's packaged weirdly
            safe_enlarge_ratio = float(region.enlarge_ratio)

            ballon_mask, xyxy = extract_ballon_region(
                original_img.copy(), 
                safe_xywh, 
                enlarge_ratio=safe_enlarge_ratio
            )
        except Exception as e:
            print(f"extract_ballon_region threw {e}")

        ballon_area = (ballon_mask > 0).sum()
        rotated, rx, ry = False, 0.0, 0.0
    
        if abs(region.angle) > 3:
            rotated = True
            angle_rad = np.deg2rad(region.angle)
            angle_sin, angle_cos = np.sin(angle_rad), np.cos(angle_rad)
            rotated_ballon_mask = np.array(Image.fromarray(ballon_mask).rotate(region.angle, expand=True))

            deg = region.angle % 360
            if 0 < deg <= 90:
                ry = abs(ballon_mask.shape[1] * angle_sin)
            elif 90 < deg <= 180:
                rx = abs(ballon_mask.shape[1] * angle_cos)
                ry = rotated_ballon_mask.shape[0]
            elif 180 < deg <= 270:
                ry = abs(ballon_mask.shape[0] * angle_cos)
                rx = rotated_ballon_mask.shape[1]
            else:
                rx = abs(ballon_mask.shape[0] * angle_sin)
            ballon_mask = rotated_ballon_mask
        
        line_width = sum(word_lengths) + delimiter_len * (len(word_lengths) - 1)
        region_area = line_width * line_height + delimiter_len * (len(words) - 1) * line_height
        area_ratio = (ballon_area / region_area) if region_area > 0 else 1.0
        resize_ratio = 1.0

        if area_ratio < ballonarea_thresh and area_ratio > 0:
            scale_factor = ballonarea_thresh / area_ratio
            resize_ratio = min(np.sqrt(scale_factor), (1 / downscale_constraint) ** 2)
            rx *= resize_ratio
            ry *= resize_ratio
            new_shape = (int(resize_ratio * ballon_mask.shape[1]), int(resize_ratio * ballon_mask.shape[0]))
            ballon_mask = cv2.resize(ballon_mask, new_shape)

        nonzero_pts = cv2.findNonZero(ballon_mask)
        if nonzero_pts is None:
            continue
        region_x, region_y, region_w, region_h = cv2.boundingRect(nonzero_pts)

        base_word_idx = max(enumerate(word_lengths), key=lambda x: x[1])[0]
        base_length_word = words[base_word_idx]
        if not base_length_word:
            continue
        
        lines_needed = len(region.translation) / len(base_length_word) 

        # Enforce line_height is an integer for clean division math
        l_height = int(line_height)
        lines_available = int(abs(xyxy[3] - xyxy[1])) // l_height + 1

        font_size_multiplier = max(
            min(region_w / (base_length + 2 * sw), lines_available / lines_needed),
            downscale_constraint
        )

        if font_size_multiplier < 1:
            font_size = int(font_size * font_size_multiplier)
            font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(font_size, words)

        textlines = layout_lines_aligncenter(
            ballon_mask, words, word_lengths, delimiter_len, line_height, delimiter=delimiter
        )
        if not textlines:
            continue

        line_cy = np.mean([line.pos_y for line in textlines]) + line_height / 2
        region_cy = region_y + region_h / 2
        y_offset = int(round(np.clip(region_cy - line_cy, -line_height, line_height)))

        lines_x1 = np.array([line.pos_x for line in textlines])
        lines_x2 = np.array([max(line.pos_x, 0) + line.length for line in textlines])
        canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
        canvas_y1, canvas_y2 = textlines[0].pos_y - sw, textlines[-1].pos_y + line_height + sw
        canvas_h = int(canvas_y2 - canvas_y1)
        canvas_w = int(canvas_x2 - canvas_x1)

        for line in textlines:
            line.pos_x -= canvas_x1
            line.pos_y -= canvas_y1

        
        from .generic2 import color_difference, is_right_to_left_char, is_valuable_char

        # ----------------------------------------------------
        # COLOR MATRIX RE-ALIGNMENT HOOK
        # ----------------------------------------------------
        # Force incoming color vectors to BGR layout for OpenCV rendering matrix layers
        frgb = np.array(region.fg_colors).astype(np.int32).flatten()[:3]
        brgb = np.array(region.bg_colors).astype(np.int32).flatten()[:3]

        if region.adjust_bg_color:
            fg_avg = np.mean(frgb)
            if color_difference(frgb, brgb) < 30:
                brgb = np.array([255, 255, 255]) if fg_avg <= 127 else np.array([0, 0, 0])

        # If text color looks inverted or swapped, convert RGB to BGR for the renderer:
        region_font_color = frgb #(int(frgb[2]), int(frgb[1]), int(frgb[0]))   # R, G, B -> B, G, R
        region_stroke_color = brgb #(int(brgb[2]), int(brgb[1]), int(brgb[0])) # R, G, B -> B, G, R

        region.font_size = force_font_size if force_font_size != -1 else font_size

        textlines_image = render_lines(
            textlines, canvas_h, canvas_w, font_size, sw, line_spacing, region_font_color, region_stroke_color
        )

        if textlines_image is None:
            continue

        rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
        rel_cy = ((canvas_y1 + canvas_y2) / 2 - ry + y_offset) / resize_ratio

        if rotated:
            rcx = rel_cx * angle_cos - rel_cy * angle_sin
            rcy = rel_cx * angle_sin + rel_cy * angle_cos
            rel_cx, rel_cy = rcx, rcy
            textlines_image = textlines_image.rotate(-region.angle, expand=True, resample=Image.Resampling.BILINEAR)
            bbox = textlines_image.getbbox()
            if bbox:
                textlines_image = textlines_image.crop(bbox)

        abs_x = int(rel_cx + xyxy[0] - textlines_image.width / 2)
        abs_y = int(rel_cy + xyxy[1] - textlines_image.height / 2)
        
        rendered_text_list.append(textlines_image)
        rendered_text_coords.append((abs_x, abs_y))

    return rendered_text_list, rendered_text_coords

def render_textblock_list_ch_rendered(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False ) -> np.ndarray:
    from ..rendering import render, resize_regions_to_font_size 
    from tqdm import tqdm
    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # TODO: Maybe remove intersections

    img = img.copy() # Don't overwrite img_inpainted
    text_render_list = []
    text_render_coords = []
    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)  
        new_text_img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
        from PIL import Image
        Image.fromarray(new_text_img).show()
        text_render_coords.append(dst_points)
        text_render_list.append(new_text_img)
    return text_render_list, text_render_coords

class LiveEditor(tk.Frame):
    def __init__(self, master, initial_image_dict=None, **kwargs):
        super().__init__(master, **kwargs)

        self.active_layer_id = None
        self.active_layer_index = None
        self.drag_data = {"x": 0, "y": 0}
        self.current_bg_pil = None
        self.tk_bg_image = None
        self.thumbnail_refs = []
        self.layer_image_refs = {}
        self.bg_width = 0
        self.bg_height = 0

        self.gallery_images = []
        self.current_img_index = -1
        self._tk_overlay_cache = []

        # Closable Property Box widget references
        self.property_box: Optional[tk.Toplevel] = None
        self.layer_font_size_display: Optional[tk.Entry] = None
        self.fg_color_swatch: Optional[tk.Label] = None
        self.fg_color_pick_btn: Optional[tk.Button] = None
        self.bg_color_swatch: Optional[tk.Label] = None
        self.bg_color_pick_btn: Optional[tk.Button] = None
        self.layer_text_display: Optional[tk.Entry] = None
        self.render_btn: Optional[tk.Button] = None
        self.show_bboxes_active = False  # Toggle tracking state for bounding boxes
        self.property_box_has_focus = False  # Blocks canvas events when typing in properties
        self._saved_property_box_geometry = None  # Remembers custom property box window coordinates
        self.is_fullscreen = False  # State tracker for window presentation modes

        self.inpainter_mode_active = False  # True when brush drawing overlay is toggled on
        self.brush_size = 20  # Safe initial default soft brush pixel radius
        self.manual_mask_surface = None  # NumPy tracking array matching native background size
        self._last_brush_coords = None  # Smooth brush interpolator stroke coordinate tracker

        self.current_fg_color: Tuple[int, int, int] = (0, 0, 0)
        self.current_bg_color: Tuple[int, int, int] = (255, 255, 255)

        self.current_font_path = discover_global_font_path()

        self._build_top_controls()
        self._build_main_workspace()
        self.timer = None
        if initial_image_dict:
            self.append_image(initial_image_dict)
        else:
            self._show_placeholder_text()

        # Re-mapped page turn workflows to Alt + Arrow combinations to stay clear of nudge keys
        self.bind_all("<Alt-Left>", lambda e: self._on_arrow_left_press(e) if not self.property_box_has_focus else None)
        self.bind_all("<Alt-Right>", lambda e: self._on_arrow_right_press(e) if not self.property_box_has_focus else None)

        self.zoom_scale = -1.0         # 1.0 represents 100% native size resolution
        self.min_zoom = 0.15           # Minimum 20% downscale limit
        self.max_zoom = 4.0            # Maximum 400% upscale threshold
        
        # Bind the unified Ctrl + Mouse Wheel event listeners cleanly
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel_zoom)
        self.canvas.bind("<Control-Button-4>", self._on_ctrl_mousewheel_zoom)
        self.canvas.bind("<Control-Button-5>", self._on_ctrl_mousewheel_zoom)

        # Bind background canvas clicks to clear selection and close properties
        self.canvas.bind("<Button-1>", self._on_canvas_background_click)
        
        # Intercepts arrow keys to micro-adjust selected text block positions
        self.bind_all("<Up>", lambda e: self._on_keyboard_nudge(0, -1, e))
        self.bind_all("<Down>", lambda e: self._on_keyboard_nudge(0, 1, e))
        self.bind_all("<Left>", lambda e: self._on_keyboard_nudge(-1, 0, e))
        self.bind_all("<Right>", lambda e: self._on_keyboard_nudge(1, 0, e))

        # Shift + Arrow Keys for fast 10-pixel jumps
        self.bind_all("<Shift-Up>", lambda e: self._on_keyboard_nudge(0, -10, e))
        self.bind_all("<Shift-Down>", lambda e: self._on_keyboard_nudge(0, 10, e))
        self.bind_all("<Shift-Left>", lambda e: self._on_keyboard_nudge(-10, 0, e))
        self.bind_all("<Shift-Right>", lambda e: self._on_keyboard_nudge(10, 0, e))
 
    def _build_top_controls(self):
        self.input_frame = tk.Frame(self, bg="#f0f0f0")
        self.input_frame.pack(side="top", fill="x", pady=5)

        # Packed from the right in order:
        # [Run Inpaint] [Brush Size] [Inpaint Mode] [Fullscreen] [Show BBoxes] [Export Image] [Export All] [Fit to Window] [Exit]

        # 1. Exit Button (rightmost)
        self.exit_btn = tk.Button(
            self.input_frame, text="Exit", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", relief="raised", bd=2, padx=10, pady=2, command=self.exit_program
        )
        self.exit_btn.pack(side="right", padx=(5, 15))

        # 2. Fit to Window Button
        self.fit_window_btn = tk.Button(
            self.input_frame, text="Fit to Window", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", relief="raised", bd=2, padx=10, pady=2, command=self.trigger_fit_to_window
        )
        self.fit_window_btn.pack(side="right", padx=10)

        # 3. Export All Button
        self.export_all_btn = tk.Button(
            self.input_frame, text="Export All", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", disabledforeground="#808080", relief="raised", bd=2, padx=10, pady=2, command=self.export_all_images
        )
        self.export_all_btn.pack(side="right", padx=5)

        # 4. Export Image Button
        self.export_img_btn = tk.Button(
            self.input_frame, text="Export Image", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", disabledforeground="#808080", relief="raised", bd=2, padx=10, pady=2, command=self.export_current_image
        )
        self.export_img_btn.pack(side="right", padx=5)

        # 5. Show BBoxes Button
        self.show_bboxes_btn = tk.Button(
            self.input_frame, text="Show BBoxes", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", relief="raised", bd=2, padx=10, pady=2, command=self.toggle_bounding_boxes
        )
        self.show_bboxes_btn.pack(side="right", padx=5)

        # 6. Fullscreen Button
        self.fullscreen_btn = tk.Button(
            self.input_frame, text="Fullscreen", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", relief="raised", bd=2, padx=10, pady=2, command=self.toggle_fullscreen_mode
        )
        self.fullscreen_btn.pack(side="right", padx=5)

        # 7. Inpaint Mode Toggle Button
        self.inpaint_toggle_btn = tk.Button(
            self.input_frame, text="Inpaint Mode", font=("Arial", 10, "bold"), bg="#d9d9d9", fg="black",
            activebackground="#bfbfbf", activeforeground="black", relief="raised", bd=2, padx=10, pady=2, command=self.toggle_inpainter_mode
        )
        self.inpaint_toggle_btn.pack(side="right", padx=5)

        # 8. Brush Size Row Control Spinner Group
        brush_frame = tk.Frame(self.input_frame, bg="#f0f0f0")
        brush_frame.pack(side="right", padx=5)
        tk.Label(brush_frame, text="Brush Size:", font=("Arial", 9, "bold"), bg="#f0f0f0").pack(side="left", padx=2)
        self.brush_size_spinner = tk.Spinbox(brush_frame, from_=5, to=150, width=4, font=("Arial", 9), justify="center", command=self._on_brush_spinner_change)
        self.brush_size_spinner.delete(0, "end")
        self.brush_size_spinner.insert(0, str(self.brush_size))
        self.brush_size_spinner.pack(side="left", padx=2)
        self.brush_size_spinner.bind("<KeyRelease>", lambda e: self._on_brush_spinner_change())

        # 9. Run Inpaint Trigger Action Button
        self.run_inpaint_btn = tk.Button(
            self.input_frame, text="Run Inpaint", font=("Arial", 10, "bold"), bg="#bfbfbf", fg="white",
            activebackground="#d84315", activeforeground="white", state="disabled", relief="raised", bd=2, padx=10, pady=2, command=self.execute_manual_lama_mpe_inpaint
        )
        self.run_inpaint_btn.pack(side="right", padx=5)

    def render_zoomed_view(self, image_scaler = Image.Resampling.BILINEAR):
        """
        Dynamically applies PIL rescaling across the active background 
        and text overlays relative to the current zoom_scale factor.
        Centering calculation is built directly into placement geometry.
        """
        if self.current_img_index == -1 or not self.current_bg_pil:
            return

        data_bundle = self.gallery_images[self.current_img_index]
        w, h = self.current_bg_pil.size
        
        # Calculate fresh scaled spatial dimensions
        scaled_w = max(1, int(w * self.zoom_scale))
        scaled_h = max(1, int(h * self.zoom_scale))

        # 1. Clear out active layers
        self.canvas.delete("all")
        self._tk_overlay_cache.clear()
        self.layer_image_refs.clear()
        
        # 2. Force layout updates to read correct container bounds
        self.update_idletasks()
        view_w = self.canvas_container.winfo_width()
        view_h = self.canvas_container.winfo_height()
        if view_w <= 1: view_w = 1280
        if view_h <= 1: view_h = 720

        # 3. Dynamic Padding Offsets: If image is smaller than window, center it inside the canvas.
        # If it is larger, place it at 0,0 and let the scrollbars handle it.
        offset_x = max(0, (view_w - scaled_w) // 2)
        offset_y = max(0, (view_h - scaled_h) // 2)

        # 4. Rescale and draw the main background image canvas block
        zoomed_bg_pil = self.current_bg_pil.resize((scaled_w, scaled_h), image_scaler)
        self.tk_bg_image = ImageTk.PhotoImage(zoomed_bg_pil)
        self.canvas.create_image(offset_x, offset_y, image=self.tk_bg_image, anchor="nw", tags="bg_layer")
        
        # Configure navigation scroll barriers to map the absolute visible boundaries flawlessly
        scroll_w = max(scaled_w, view_w)
        scroll_h = max(scaled_h, view_h)
        self.canvas.config(scrollregion=(0, 0, scroll_w, scroll_h))

        # 5. Pull text overlays and position them relative to the dynamic offsets
        layers = data_bundle.get("layers", [])
        coords = data_bundle.get("coords", [])

        for i, pil_layer in enumerate(layers):
            if not isinstance(pil_layer, Image.Image):
                continue
            try:
                l_w, l_h = pil_layer.size
                new_l_w = max(1, int(l_w * self.zoom_scale))
                new_l_h = max(1, int(l_h * self.zoom_scale))
                
                scaled_layer_pil = pil_layer.resize((new_l_w, new_l_h), Image.Resampling.BILINEAR)
                tk_layer_img = ImageTk.PhotoImage(scaled_layer_pil)
                self._tk_overlay_cache.append(tk_layer_img)

                # Project coordinate positions out with the dynamic centering offset included
                orig_x, orig_y = coords[i] if i < len(coords) else (50, 50)
                translated_x = int(orig_x * self.zoom_scale) + offset_x
                translated_y = int(orig_y * self.zoom_scale) + offset_y

                layer_tag = f"layer_idx_{i}"
                layer_id = self.canvas.create_image(
                    translated_x, translated_y, image=tk_layer_img, anchor="nw", tags=("draggable_layer", layer_tag)
                )

                self.layer_image_refs[layer_id] = tk_layer_img
                
                self.canvas.tag_bind(layer_id, "<ButtonPress-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_start(event, lid, lidx))
                self.canvas.tag_bind(layer_id, "<B1-Motion>", self._on_dragging)
                self.canvas.tag_bind(layer_id, "<ButtonRelease-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_end(event, lid, lidx))
            except Exception as e:
                print(f"Error scaling overlay item {i}: {e}")
                # Rerender wireframe tracking layer if active

        self.draw_inline_bounding_boxes()

    def toggle_bounding_boxes(self):
        """Toggles the inline canvas layer bounding rectangles visibility state."""
        self.show_bboxes_active = not self.show_bboxes_active
        
        if self.show_bboxes_active:
            self.show_bboxes_btn.config(bg="#a5d6a7", activebackground="#81c784") # Green highlight when active
            self.draw_inline_bounding_boxes()
        else:
            self.show_bboxes_btn.config(bg="#d9d9d9", activebackground="#bfbfbf")
            self.canvas.delete("inline_bbox_rect")
    
    def toggle_fullscreen_mode(self):
        """Toggles window fullscreen mode via pure OS presentation layout abstractions."""
        top_level = self.winfo_toplevel()
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            # Shift button styling to showcase state change
            self.fullscreen_btn.config(bg="#a5d6a7", activebackground="#81c784")
            
            # Cross-platform window expansion protocol
            top_level.attributes("-fullscreen", True)
        else:
            self.fullscreen_btn.config(bg="#d9d9d9", activebackground="#bfbfbf")
            top_level.attributes("-fullscreen", False)
            
            # Recalculate dimensions to return to normal size constraints smoothly
            if self.current_img_index != -1 and self.gallery_images:
                self.load_background_image_by_index(self.current_img_index)
                
        # Proactively re-center viewport layouts matching the changes
        self.master.after(150, self.trigger_fit_to_window)

    def toggle_inpainter_mode(self):
        """Toggles drawing state variables and intercepts layout mouse operations on the main canvas workspace."""
        self.inpainter_mode_active = not self.inpainter_mode_active
        
        if self.inpainter_mode_active:
            self.inpaint_toggle_btn.config(bg="#ffb74d", activebackground="#ffa726") # Orange highlight when drawing mode active
            self.close_property_box()
            self.canvas.config(cursor="pencil")
            
            self.canvas.itemconfig("draggable_layer", state="hidden")
            self.canvas.delete("inline_bbox_rect")

            # Unbind conflicting layer drag operations to keep mouse handling pristine
            self.canvas.tag_unbind("draggable_layer", "")
            self.canvas.tag_unbind("draggable_layer", "")
            
            # Direct brush mapping input loops setup overrides onto the canvas background surface
            self.canvas.bind("<ButtonPress-1>", self._on_brush_stroke_start)
            self.canvas.bind("<B1-Motion>", self._on_brush_stroke_drawing)
            self.canvas.bind("<ButtonRelease-1>", self._on_brush_stroke_end)
        else:
            self.inpaint_toggle_btn.config(bg="#d9d9d9", activebackground="#bfbfbf")
            self.canvas.config(cursor="")
            
            # Wipe out visual stroke previews from view pane
            self.canvas.delete("manual_brush_preview")
            self.canvas.itemconfig("draggable_layer", state="normal")
            # Re-bind layer drag listeners back onto regular execution tracks safely
            self.canvas.bind("<Button-1>", self._on_canvas_background_click)
            self.render_zoomed_view()

    def _on_brush_spinner_change(self):
        try:
            val = int(self.brush_size_spinner.get())
            self.brush_size = max(5, min(val, 150))
        except ValueError:
            pass

    def _on_brush_stroke_start(self, event):
        if not self.inpainter_mode_active or self.current_img_index == -1:
            return
        self._last_brush_coords = (event.x, event.y)
        self._paint_on_mask_surface(event.x, event.y)

    def _on_brush_stroke_drawing(self, event):
        if not self.inpainter_mode_active or self._last_brush_coords is None:
            return
        
        # Line interpolation loop mechanics to provide perfectly gapless soft drawing marks
        x1, y1 = self._last_brush_coords
        x2, y2 = event.x, event.y
        
        distance = int(np.hypot(x2 - x1, y2 - y1))
        steps = max(1, distance // 2)
        
        for i in range(steps + 1):
            t = i / steps
            ix = int(x1 + t * (x2 - x1))
            iy = int(y1 + t * (y2 - y1))
            self._paint_on_mask_surface(ix, iy)
            
        self._last_brush_coords = (x2, y2)

    def _on_brush_stroke_end(self, event):
        self._last_brush_coords = None
        if self.manual_mask_surface is not None and self.manual_mask_surface.sum() > 0:
            self.run_inpaint_btn.config(state="normal")

    def _paint_on_mask_surface(self, win_x, win_y):
        """Projects cursor window tracking markers onto the unscaled matrix tracking surface layer."""
        if self.manual_mask_surface is None or not self.current_bg_pil:
            return

        # Translate raw cursor screen-window coordinates directly into scroll-aware canvas system points
        canvas_x = self.canvas.canvasx(win_x)
        canvas_y = self.canvas.canvasy(win_y)

        # 2. Extract baseline layout workspace view boundaries dynamically
        view_w = self.canvas_container.winfo_width()
        view_h = self.canvas_container.winfo_height()
        if view_w <= 1: view_w = 1280
        if view_h <= 720: view_h = 720

        # Calculate exact zoomed background image resolution footprint dimensions
        scaled_w = int(self.bg_width * self.zoom_scale)
        scaled_h = int(self.bg_height * self.zoom_scale)

        # 3. Dynamic Padding Offsets: Extract centering offsets precisely mirroring render_zoomed_view
        offset_x = max(0, (view_w - scaled_w) // 2)
        offset_y = max(0, (view_h - scaled_h) // 2)

        # 4. Inverse project scrollable canvas coordinates back into native image coordinates
        img_x = int(round((canvas_x - offset_x) / self.zoom_scale))
        img_y = int(round((canvas_y - offset_y) / self.zoom_scale))

        # 5. Bound verification: Only commit drawings if within image area bounds
        if 0 <= img_x < self.bg_width and 0 <= img_y < self.bg_height:
            # Commit mask markings into the unscaled tracking array surface channel
            cv2.circle(self.manual_mask_surface, (img_x, img_y), self.brush_size, 255, -1)
            
            # Render visual brush stroke circle path markers onto the scrollable canvas system space
            r_scaled = max(2, int(self.brush_size * self.zoom_scale))
            self.canvas.create_oval(
                canvas_x - r_scaled, canvas_y - r_scaled, 
                canvas_x + r_scaled, canvas_y + r_scaled,
                fill="#ff1744", outline="", stipple="gray50", tags="manual_brush_preview"
            )

    def draw_inline_bounding_boxes(self):
        """Calculates scale variations dynamically to project red wireframes or solid green selected outlines."""
        self.canvas.delete("inline_bbox_rect")
        # Defensively exit if brush mode is open so it doesn't collide with pixel previews
        if self.inpainter_mode_active:
            return
            
        if not self.show_bboxes_active or self.current_img_index == -1 or not self.current_bg_pil:
            return

        data_bundle = self.gallery_images[self.current_img_index]
        layers = data_bundle.get("layers", [])
        coords = data_bundle.get("coords", [])

        # Read canvas view boundaries dynamically
        self.update_idletasks()
        view_w = self.canvas_container.winfo_width()
        view_h = self.canvas_container.winfo_height()
        if view_w <= 1: view_w = 1280
        if view_h <= 1: view_h = 720

        # Calculate exact zoomed visual background bounds
        scaled_w = max(1, int(self.bg_width * self.zoom_scale))
        scaled_h = max(1, int(self.bg_height * self.zoom_scale))

        # Dynamic Padding Offsets logic must mirror render_zoomed_view centering precisely
        offset_x = max(0, (view_w - scaled_w) // 2)
        offset_y = max(0, (view_h - scaled_h) // 2)

        # Retrieve scroll position fractions to shift coordinates if window scrolls or overflows
        scroll_x_frac = self.canvas.xview()[0]
        scroll_y_frac = self.canvas.yview()[0]
        
        # Convert scroll fractions back to visual viewport pixels to maintain locked alignments
        scroll_w = max(scaled_w, view_w)
        scroll_h = max(scaled_h, view_h)
        scroll_offset_x = scroll_x_frac * scroll_w
        scroll_offset_y = scroll_y_frac * scroll_h

        for i, pil_layer in enumerate(layers):
            if not isinstance(pil_layer, Image.Image):
                continue
            
            orig_x, orig_y = coords[i] if i < len(coords) else (50, 50)
            l_w, l_h = pil_layer.size

            # Project image coordinate positions exactly matching zoom and centering transformations
            x1 = int(orig_x * self.zoom_scale) + offset_x
            y1 = int(orig_y * self.zoom_scale) + offset_y
            x2 = x1 + max(1, int(l_w * self.zoom_scale))
            y2 = y1 + max(1, int(l_h * self.zoom_scale))

            # Check if this item matches the active workspace modification index pointer
            is_selected = (self.active_layer_index is not None and i == self.active_layer_index)

            if is_selected:
                # Selected layers get a solid, thick emerald green outline
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="#00E676",
                    width=3,
                    tags="inline_bbox_rect"
                )
            else:
                # Unselected layers get a dashed red wireframe
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="red",
                    width=2,
                    dash=(4, 4),
                    tags="inline_bbox_rect"
                )
            
        self.canvas.tag_raise("inline_bbox_rect", "bg_layer")

    def exit_program(self):
        """Cleanly releases graphics/memory resources and terminates the program entirely."""
        try:
            self.on_close_cleanup()
            shutdown_event.set()
            subprocess.run(["killall" , "python"])
            os.exit(0)
        except Exception:
            pass
        finally:
            os._exit(0)

    def _build_main_workspace(self):
        self.workspace = tk.Frame(self, bg="#dcdcdc")
        self.workspace.pack(side="top", fill="both", expand=True)

        self.sidebar_container = tk.Frame(self.workspace, width=150, bg="#e0e0e0", bd=1, relief="sunken")
        self.sidebar_container.pack(side="left", fill="y")

        self.gallery_canvas = tk.Canvas(self.sidebar_container, width=130, bg="#e0e0e0", bd=0, highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.gallery_canvas.yview)
        self.scrollable_gallery_frame = tk.Frame(self.gallery_canvas, bg="#e0e0e0")

        self.scrollable_gallery_frame.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        )
        self.gallery_canvas.create_window((0, 0), window=self.scrollable_gallery_frame, anchor="nw")
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        self.gallery_scrollbar.pack(side="right", fill="y")

        self.canvas_container = tk.Frame(self.workspace, bg="gray")
        self.canvas_container.pack(side="right", fill="both", expand=True)

        self.v_scrollbar = ttk.Scrollbar(self.canvas_container, orient="vertical")
        self.h_scrollbar = ttk.Scrollbar(self.canvas_container, orient="horizontal")

        self.canvas = tk.Canvas(
            self.canvas_container, bd=0, highlightthickness=0, bg="gray",
            xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set
        )

        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas_container.rowconfigure(0, weight=1)
        self.canvas_container.columnconfigure(0, weight=1)

        self.gallery_canvas.bind("<MouseWheel>", self._on_sidebar_mousewheel)
        self.canvas.bind("<MouseWheel>", self._on_main_canvas_mousewheel)
        self.gallery_canvas.bind("<Button-4>", self._on_sidebar_mousewheel)
        self.gallery_canvas.bind("<Button-5>", self._on_sidebar_mousewheel)
        self.canvas.bind("<Button-4>", self._on_main_canvas_mousewheel)
        self.canvas.bind("<Button-5>", self._on_main_canvas_mousewheel)

        self.canvas_container.bind("<Configure>", self._on_window_resize_event)

    def _show_placeholder_text(self):
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, 500, 400))
        self.canvas.create_text(
            1280/2, 720/2,
            text="No images loaded.\n\nRunning as server...",
            fill="white", font=("Arial", 12, "italic"), justify="center", tags="placeholder"
        )

    def append_image(self, image_dict: dict, path: str = "", config: Config = None):

        self.config = config

        rendered_text_list, rendered_text_coords = [],[]
        
        rendered_text_list, rendered_text_coords = render_textblock_list_eng_rendered(
            img=image_dict['img_inpainted'],
            text_regions=image_dict['text_regions'],
            original_img=image_dict['img_rgb'],
        )
        
        """rendered_text_list, rendered_text_coords = render_textblock_list_ch_rendered(
            img=image_dict['img_inpainted'],
            text_regions=image_dict['text_regions']
        )"""
        
        rendered_text_translated = [
            " ".join(seg_eng(r.translation)) for r in image_dict['text_regions']
        ]
        gallery_image = Image.fromarray(image_dict['img_inpainted'])
   
        if not isinstance(gallery_image, Image.Image):
            return

        threading.Thread(
            target=self._async_gallery_worker,
            args=(
                gallery_image,
                rendered_text_list,
                rendered_text_coords,
                rendered_text_translated,
                image_dict,
                path,
            ),
            daemon=True,
        ).start() 

    def _async_gallery_worker(
        self,
        gallery_image,
        rendered_text_list,
        rendered_text_coords,
        rendered_text_translated,
        image_dict,
        export_path: str = "",):
        try:
            thumb_pil = Image.fromarray(image_dict['img_rgb']).copy()
            thumb_pil.thumbnail((110, 80))
            thumb_tk = ImageTk.PhotoImage(thumb_pil)

            dest_path = export_path or image_dict.get('export_path', '') or image_dict.get('dest', '')

            data_bundle = {
                "bg": gallery_image,
                "layers": rendered_text_list,
                "coords": rendered_text_coords,
                "text_strings": rendered_text_translated,
                "img_rgb": image_dict['img_rgb'],
                "img_inpainted": image_dict['img_inpainted'],
                "text_regions": image_dict['text_regions'],
                "export_path": dest_path,
            }
            self.master.after(0, self._safe_append_thumbnail, data_bundle, thumb_tk)
        except Exception as e:
            print(f"Background processing error: {e}")

    def _composite_bundle(self, bundle: dict) -> Optional[Image.Image]:
        bg_pil = bundle.get("bg")
        if bg_pil is None:
            return None

        comp = bg_pil.convert("RGBA").copy()
        layers = bundle.get("layers", [])
        coords = bundle.get("coords", [])

        for i, layer in enumerate(layers):
            if not isinstance(layer, Image.Image):
                continue
            x, y = coords[i] if i < len(coords) else (50, 50)
            layer_rgba = layer.convert("RGBA")
            comp.paste(layer_rgba, (int(x), int(y)), mask=layer_rgba)

        return comp

    def _save_composition(self, bundle: dict, export_path: str) -> bool:
        if not export_path:
            return False

        comp = self._composite_bundle(bundle)
        if comp is None:
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
            ext = os.path.splitext(export_path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                comp.convert("RGB").save(export_path, quality=95)
            else:
                comp.save(export_path)
            print(f"Exported composition to: {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting image to {export_path}: {e}")
            return False

    def export_current_image(self):
        if self.current_img_index == -1 or not self.gallery_images:
            return

        bundle = self.gallery_images[self.current_img_index]
        export_path = bundle.get("export_path", "")
        if not export_path:
            export_path = filedialog.asksaveasfilename(
                title="Save Composition",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")]
            )
            if export_path:
                bundle["export_path"] = export_path

        if export_path:
            self._save_composition(bundle, export_path)

    def export_all_images(self):
        if not self.gallery_images:
            return

        success_count = 0
        for i, bundle in enumerate(self.gallery_images):
            export_path = bundle.get("export_path", "")
            if not export_path:
                print(f"Skipping gallery item {i + 1}: No destination path set.")
                continue
            if self._save_composition(bundle, export_path):
                success_count += 1

        print(f"Batch export completed: {success_count}/{len(self.gallery_images)} image(s) saved.")

    def _on_property_box_configure(self, event):
        """Monitors property box position updates to persist custom positions in memory."""
        if self.property_box and self.property_box.winfo_exists():
            # Grab real-time relative pixel window position markers
            x = self.property_box.winfo_x()
            y = self.property_box.winfo_y()
            
            # Avoid logging initialization state flags or minimized viewport artifact values
            if x > -2000 and y > -2000:
                self._saved_property_box_geometry = (x, y)

    def open_or_update_property_box(self, layer_index: int, event=None):
        active_bundle = self.gallery_images[self.current_img_index] if self.current_img_index != -1 else {}
        text_strings = active_bundle.get("text_strings", [])
        text_regions = active_bundle.get("text_regions", [])

        display_text = text_strings[layer_index] if layer_index < len(text_strings) else ""
        region = text_regions[layer_index] if layer_index < len(text_regions) else None
        font_size = str(getattr(region, 'font_size', '')) if region else ""

        # --- EXTRACT UNTRANSLATED OCR ORIGINAL TEXT ---
        original_ocr_text = str(getattr(region, 'text', '')) if region else ""

        current_fg = getattr(region, 'fg_color', None) or getattr(region, 'font_color', None)
        current_bg = getattr(region, 'bg_color', None) or getattr(region, 'stroke_color', None)

        if (current_fg is None or current_bg is None) and hasattr(region, 'get_font_colors'):
            try:
                def_fg, def_bg = region.get_font_colors()
                if current_fg is None: current_fg = def_fg
                if current_bg is None: current_bg = def_bg
            except Exception:
                pass

        if current_fg is None: current_fg = (0, 0, 0)
        if current_bg is None: current_bg = (255, 255, 255)

        if region:
            if not hasattr(region, 'fg_colors') or region.fg_colors is None:
                region.fg_colors = np.array(current_fg, dtype=np.uint8)
            if not hasattr(region, 'bg_colors') or region.bg_colors is None:
                region.bg_colors = np.array(current_bg, dtype=np.uint8)

        fg = region.fg_colors
        bg = region.bg_colors
        fg_flat = np.ravel(fg)
        bg_flat = np.ravel(bg)

        fg_tuple = tuple(int(c) for c in fg_flat[:3])
        bg_tuple = tuple(int(c) for c in bg_flat[:3])

        self.current_fg_color = fg_tuple
        self.current_bg_color = bg_tuple

        fg_hex = f"#{fg_tuple[0]:02x}{fg_tuple[1]:02x}{fg_tuple[2]:02x}"
        bg_hex = f"#{bg_tuple[0]:02x}{bg_tuple[1]:02x}{bg_tuple[2]:02x}"

        if self.property_box is None or not self.property_box.winfo_exists():
            self.property_box = tk.Toplevel(self)
            self.property_box.transient(self.winfo_toplevel())
            self.property_box.resizable(False, False)
            self.property_box.protocol("WM_DELETE_WINDOW", self.close_property_box)
            self.property_box.config(bg="#f4f4f4")

            self.property_box.bind("<FocusIn>", lambda e: setattr(self, 'property_box_has_focus', True))
            self.property_box.bind("<FocusOut>", lambda e: setattr(self, 'property_box_has_focus', False))
            
            self.property_box.bind("<Configure>", self._on_property_box_configure)

            # Adjusted panel height to 270 to naturally comfortably fit the added layout row element
            prop_w = 340
            prop_h = 270

            if self._saved_property_box_geometry:
                self.property_box.geometry(f"{prop_w}x{prop_h}+{self._saved_property_box_geometry[0]}+{self._saved_property_box_geometry[1]}")
            else:
                screen_w = self.winfo_screenwidth()
                screen_h = self.winfo_screenheight()

                main_win = self.winfo_toplevel()
                main_win.update_idletasks()
                main_x = main_win.winfo_x()
                main_y = main_win.winfo_y()
                main_w = main_win.winfo_width()

                target_x = main_x + main_w - prop_w - 20
                target_y = main_y + 60

                if target_x + prop_w > screen_w: target_x = screen_w - prop_w - 10
                if target_y + prop_h > screen_h: target_y = screen_h - prop_h - 40
                if target_x < 0: target_x = 10
                if target_y < 0: target_y = 10

                self.property_box.geometry(f"{prop_w}x{prop_h}+{target_x}+{target_y}")

            content = tk.Frame(self.property_box, bg="#f4f4f4", padx=10, pady=8)
            content.pack(fill="both", expand=True)

            # 1. Font Size Row Layout
            size_frame = tk.Frame(content, bg="#f4f4f4")
            size_frame.pack(fill="x", pady=(0, 4))
            tk.Label(size_frame, text="Font Size:", font=("Arial", 9, "bold"), bg="#f4f4f4", width=10, anchor="w").pack(side="left")
            self.layer_font_size_display = tk.Entry(size_frame, font=("Arial", 9), width=8, bg="#ffffff", fg="#000000")
            self.layer_font_size_display.pack(side="left", padx=2)
            self.layer_font_size_display.bind("<KeyRelease>", self._on_layer_font_size_modified)

            # 2. Font Color Row Layout
            fg_frame = tk.Frame(content, bg="#f4f4f4")
            fg_frame.pack(fill="x", pady=(0, 4))
            tk.Label(fg_frame, text="Font Color:", font=("Arial", 9, "bold"), bg="#f4f4f4", width=10, anchor="w").pack(side="left")
            self.fg_color_swatch = tk.Label(fg_frame, width=4, height=1, relief="solid", bd=1, bg=fg_hex)
            self.fg_color_swatch.pack(side="left", padx=(2, 4))
            self.fg_color_pick_btn = tk.Button(fg_frame, text="🎨 Choose...", font=("Arial", 8), command=self._on_choose_fg_color)
            self.fg_color_pick_btn.pack(side="left")

            # 3. Background Color Row Layout
            bg_frame = tk.Frame(content, bg="#f4f4f4")
            bg_frame.pack(fill="x", pady=(0, 4))
            tk.Label(bg_frame, text="BG Color:", font=("Arial", 9, "bold"), bg="#f4f4f4", width=10, anchor="w").pack(side="left")
            self.bg_color_swatch = tk.Label(bg_frame, width=4, height=1, relief="solid", bd=1, bg=bg_hex)
            self.bg_color_swatch.pack(side="left", padx=(2, 4))
            self.bg_color_pick_btn = tk.Button(bg_frame, text="🎨 Choose...", font=("Arial", 8), command=self._on_choose_bg_color)
            self.bg_color_pick_btn.pack(side="left")

            # 4. Font Face Dropdown Layout
            font_frame = tk.Frame(content, bg="#f4f4f4")
            font_frame.pack(fill="x", pady=(0, 4))
            tk.Label(font_frame, text="Font Face:", font=("Arial", 9, "bold"), bg="#f4f4f4", width=10, anchor="w").pack(side="left")
            self.available_fonts_dict = get_system_and_fallback_fonts()
            font_display_names = sorted(list(self.available_fonts_dict.keys()))
            self.layer_font_combobox = ttk.Combobox(font_frame, values=font_display_names, font=("Arial", 8), state="readonly")
            self.layer_font_combobox.pack(side="left", padx=2, fill="x", expand=True)
            self.layer_font_combobox.bind("<<ComboboxSelected>>", self._on_property_font_dropdown_changed)

            # 5. UNTRANSLATED SOURCE TEXT DISPLAY ROW (Newly Added Layout Segment)
            source_frame = tk.Frame(content, bg="#f4f4f4")
            source_frame.pack(fill="x", pady=(0, 4))
            tk.Label(source_frame, text="Source Text:", font=("Arial", 9, "bold"), bg="#f4f4f4", width=10, anchor="w").pack(side="left")
            
            # Using a flat readonly entry layout context to keep selections copyable without edits
            self.layer_source_display = tk.Entry(source_frame, font=("Arial", 9), bg="#e9e9e9", fg="#555555", bd=1, relief="sunken")
            self.layer_source_display.pack(side="left", padx=2, fill="x", expand=True)

            # 6. Text Input Row Layout
            text_frame = tk.Frame(content, bg="#f4f4f4")
            text_frame.pack(fill="x", pady=(0, 8))
            tk.Label(text_frame, text="Layer Text:", font=("Arial", 9, "bold"), bg="#f4f4f4", width=10, anchor="w").pack(side="left")
            self.layer_text_display = tk.Entry(text_frame, font=("Arial", 9), bg="#ffffff", fg="#000000")
            self.layer_text_display.pack(side="left", padx=2, fill="x", expand=True)
            self.layer_text_display.bind("<KeyRelease>", self._on_layer_text_modified)

            # 7. Action Control Footer Buttons
            btn_frame = tk.Frame(content, bg="#f4f4f4")
            btn_frame.pack(fill="x", side="bottom", pady=(2, 0))

            self.render_btn = tk.Button(
                btn_frame, text="Render Text", font=("Arial", 9, "bold"), bg="#2e7d32", fg="#ffffff",
                activebackground="#1b5e20", activeforeground="#ffffff", padx=10, pady=2, relief="raised", bd=2, cursor="hand2",
                command=self._on_render_text_click
            )
            self.render_btn.pack(side="left")

            close_btn = tk.Button(btn_frame, text="Close", font=("Arial", 9), bg="#e0e0e0", padx=6, pady=2, command=self.close_property_box)
            close_btn.pack(side="right")

        self.property_box.title(f"Layer {layer_index + 1} Properties")
        
        # Update original untranslated text string elements layout
        self.layer_source_display.config(state="normal")
        self.layer_source_display.delete(0, tk.END)
        self.layer_source_display.insert(0, original_ocr_text)
        # Lock back down to read-only state so it acts purely as a copyable highlight element
        self.layer_source_display.config(state="readonly")

        self.layer_text_display.config(state="normal")
        self.layer_text_display.delete(0, tk.END)
        self.layer_text_display.insert(0, display_text)

        self.layer_font_size_display.config(state="normal")
        self.layer_font_size_display.delete(0, tk.END)
        self.layer_font_size_display.insert(0, font_size)

        current_selection = self.layer_font_combobox.get()
        if not current_selection and hasattr(self, 'available_fonts_dict'):
            font_names = sorted(list(self.available_fonts_dict.keys()))
            anime_ace_match = [n for n in font_names if "anime_ace" in n.lower()]
            if anime_ace_match:
                self.layer_font_combobox.set(anime_ace_match[0])
            elif font_names:
                self.layer_font_combobox.current(0)

        self.fg_color_swatch.config(fg=fg_hex)
        self.bg_color_swatch.config(bg=bg_hex)
        self.property_box.lift()

        self.draw_inline_bounding_boxes()

    def close_property_box(self):
        self.property_box_has_focus = False  # Ensure input protection releases safely
        self.active_layer_index = None       # Reset active element tracking
        
        if self.property_box is not None:
            if self.property_box.winfo_exists():
                self.property_box.destroy()
            self.property_box = None
            self.layer_font_size_display = None
            self.fg_color_swatch = None
            self.fg_color_pick_btn = None
            self.bg_color_swatch = None
            self.bg_color_pick_btn = None
            self.layer_text_display = None
            self.render_btn = None
            
        # Clear out target selections formatting right away
        self.draw_inline_bounding_boxes()

    def _on_keyboard_nudge(self, dx: int, dy: int, event=None):
        """
        Micro-adjusts the selected text layer coordinate position via arrow keys.
        Safely aborts calculations if a text property input field is currently active.
        """
        # 1. Defensively exit if property box has input focus, or no layer is selected
        if self.property_box_has_focus or self.active_layer_id is None or self.active_layer_index is None:
            return

        if self.current_img_index == -1:
            return

        # 2. Translate the visual element layer directly on the Tkinter canvas space
        self.canvas.move(self.active_layer_id, dx, dy)

        # 3. Pull the updated visual position and sync back to the master native coordinate dataset
        current_coords = self.canvas.coords(self.active_layer_id)
        if current_coords:
            active_bundle = self.gallery_images[self.current_img_index]
            coords = active_bundle.setdefault("coords", [])
            
            view_w = self.canvas_container.winfo_width()
            view_h = self.canvas_container.winfo_height()
            if view_w <= 1: view_w = 1280
            if view_h <= 1: view_h = 720

            scaled_w = int(self.bg_width * self.zoom_scale)
            scaled_h = int(self.bg_height * self.zoom_scale)
            
            offset_x = max(0, (view_w - scaled_w) // 2)
            offset_y = max(0, (view_h - scaled_h) // 2)
                
            native_x = int(round((current_coords[0] - offset_x) / self.zoom_scale))
            native_y = int(round((current_coords[1] - offset_y) / self.zoom_scale))
            
            if self.active_layer_index < len(coords):
                coords[self.active_layer_index] = (native_x, native_y)

        # 4. Instantly refresh the wireframe outline location matching your shift coordinates
        self.draw_inline_bounding_boxes()

    def _on_choose_fg_color(self):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return

        init_hex = f"#{self.current_fg_color[0]:02x}{self.current_fg_color[1]:02x}{self.current_fg_color[2]:02x}"
        parent_win = self.property_box if (self.property_box and self.property_box.winfo_exists()) else self

        chosen = colorchooser.askcolor(color=init_hex, title="Choose Font Color", parent=parent_win)

        if chosen and chosen[0] is not None:
            rgb = tuple(int(round(c)) for c in chosen[0])
            self.current_fg_color = rgb
            hex_color = chosen[1] or f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            if self.fg_color_swatch and self.fg_color_swatch.winfo_exists():
                self.fg_color_swatch.config(bg=hex_color)

            active_bundle = self.gallery_images[self.current_img_index]
            text_regions = active_bundle.get("text_regions", [])
            if self.active_layer_index < len(text_regions):
                reg = text_regions[self.active_layer_index]
                # Enforce total synchronization across metadata namespaces
                color_arr = np.array(rgb, dtype=np.uint8)
                reg.fg_color = rgb
                reg.font_color = rgb
                reg.fg_colors = color_arr
                if hasattr(reg, 'set_fg_colors'):
                    reg.set_fg_colors(color_arr)

    def _on_choose_bg_color(self):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return

        init_hex = f"#{self.current_bg_color[0]:02x}{self.current_bg_color[1]:02x}{self.current_bg_color[2]:02x}"
        parent_win = self.property_box if (self.property_box and self.property_box.winfo_exists()) else self

        chosen = colorchooser.askcolor(
            color=init_hex,
            title="Choose Background (Stroke) Color",
            parent=parent_win
        )

        if chosen and chosen[0] is not None:
            rgb = tuple(int(round(c)) for c in chosen[0])
            self.current_bg_color = rgb
            hex_color = chosen[1] or f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            if self.bg_color_swatch and self.bg_color_swatch.winfo_exists():
                self.bg_color_swatch.config(bg=hex_color)

            active_bundle = self.gallery_images[self.current_img_index]
            text_regions = active_bundle.get("text_regions", [])
            if self.active_layer_index < len(text_regions):
                reg = text_regions[self.active_layer_index]
                color_arr = np.array(rgb, dtype=np.uint8)
                # Sync standard Python metadata tuples
                reg.bg_color = rgb
                reg.stroke_color = rgb
                reg.bg_colors = color_arr
                # Sync underlying rendering engine NumPy matrix color definitions
                if hasattr(reg, 'set_bg_colors'):
                    reg.set_bg_colors(color_arr)

    def _on_layer_font_size_modified(self, event):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return
        if not self.layer_font_size_display or not self.layer_font_size_display.winfo_exists():
            return

        active_bundle = self.gallery_images[self.current_img_index]
        text_regions = active_bundle.get("text_regions", [])
        val = self.layer_font_size_display.get().strip()

        if self.active_layer_index < len(text_regions):
            try:
                new_size = float(val) if '.' in val else int(val)
                text_regions[self.active_layer_index].font_size = new_size
            except ValueError:
                pass

    def _on_layer_text_modified(self, event):
        if self.current_img_index == -1 or self.active_layer_index is None:
            return
        if not self.layer_text_display or not self.layer_text_display.winfo_exists():
            return

        updated_text = self.layer_text_display.get()
        active_bundle = self.gallery_images[self.current_img_index]
        text_strings = active_bundle.setdefault("text_strings", [])

        while len(text_strings) <= self.active_layer_index:
            text_strings.append("")

        text_strings[self.active_layer_index] = updated_text

    def _on_drag_start(self, event, target_id, layer_index):
        if event.state & 0x0004:
            return

        self.open_or_update_property_box(layer_index, event)

        self.active_layer_id = target_id
        self.active_layer_index = layer_index
        self.canvas.itemconfig(target_id, state="normal")
        
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
        return "break"  # Prevents event bubbling into the background canvas layer

    def _on_render_text_click(self):
        if self.current_img_index == -1:
            return

        active_bundle = self.gallery_images[self.current_img_index]
        original_img_rgb = active_bundle.get("img_rgb")
        img_inpainted = active_bundle.get("img_inpainted")
        text_regions = active_bundle.get("text_regions", [])
        text_strings = active_bundle.get("text_strings", [])

        if img_inpainted is None or original_img_rgb is None:
            return

        for i, region in enumerate(text_regions):
            if i < len(text_strings):
                region.translation = text_strings[i]
            
        target_text_region = text_regions[self.active_layer_index]
        try:
            saved_coords = active_bundle["coords"][self.active_layer_index]
            new_rendered_layers, new_coords = render_textblock_list_eng_rendered(
                img=img_inpainted,
                text_regions=[target_text_region,], 
                original_img=original_img_rgb,
                font_path=self.current_font_path,
                force_font_size=target_text_region.font_size
            )

            processed_layers = [
                layer if isinstance(layer, Image.Image) else Image.fromarray(layer)
                for layer in new_rendered_layers
            ]

            active_bundle["layers"][self.active_layer_index] = processed_layers[0]
            active_bundle["coords"][self.active_layer_index] = (new_coords[0][0], new_coords[0][1])

            self.load_background_image_by_index(self.current_img_index)
        except Exception as e:
            print(f"Error during on-demand text layer rendering: {e}")

    def _safe_append_thumbnail(self, data_bundle, thumb_tk):
        """
        Inserts incoming page frames alphanumerically by their export_path filename
        and cleanly updates the Tkinter gallery widget order.
        """
        # 1. Track the true target path identifier for this incoming file
        target_path = data_bundle.get("export_path", "") or data_bundle.get("dest", "")
        
        # 2. Extract an ordered alphanumeric tracking list of paths currently in our gallery
        existing_paths = [
            b.get("export_path", "") or b.get("dest", "") 
            for b in self.gallery_images
        ]
        
        # 3. Use bisect to discover the exact alphanumeric index insertion point
        # We wrap the evaluation inside our natural sort key to ensure '2' comes before '10'
        sorted_keys = [natural_sort_key(p) for p in existing_paths]
        target_key = natural_sort_key(target_path)
        
        assigned_index = bisect.bisect_left(sorted_keys, target_key)
        
        # 4. Slide the new asset metrics directly into the mapped array index slot
        self.gallery_images.insert(assigned_index, data_bundle)
        self.thumbnail_refs.insert(assigned_index, thumb_tk)
        
        # 5. Redraw the sidebar gallery grid completely to reflect the true sorting order
        self._rebuild_sidebar_gallery_widgets()

        # If this is the first image entering the session pool, boot the canvas view right away
        if len(self.gallery_images) == 1:
            self.load_background_image_by_index(0)
        else:
            # If the user is currently viewing a page that shifted position because a new 
            # alphanumeric file sneaked in before it, update the current tracking pointer
            if self.current_img_index != -1 and assigned_index <= self.current_img_index:
                self.current_img_index += 1

    def _rebuild_sidebar_gallery_widgets(self):
        """Wipes and systematically repacks sidebar thumb buttons in sorted order."""
        # Clear any active button widgets packed inside the sidebar frame container
        for widget in self.scrollable_gallery_frame.winfo_children():
            widget.destroy()
            
        # Re-instantiate buttons targeting their newly updated sorted indices
        for idx, bundle in enumerate(self.gallery_images):
            tk_thumb = self.thumbnail_refs[idx]
            
            btn = tk.Button(
                self.scrollable_gallery_frame,
                image=tk_thumb,
                command=lambda current_idx=idx: self.load_background_image_by_index(current_idx),
                bd=2,
                relief="groove"
            )
            btn.pack(pady=8, padx=10)

    def load_background_image_by_index(self, target_index: int):
        if not self.gallery_images or not (0 <= target_index < len(self.gallery_images)):
            return

        # Capture the current viewport position fractions before changing the image index
        prev_x_view = self.canvas.xview() if hasattr(self, 'canvas') else (0.0, 1.0)
        prev_y_view = self.canvas.yview() if hasattr(self, 'canvas') else (0.0, 1.0)
        
        self.close_property_box()
        self.current_img_index = target_index
        
        # Pass the viewport tracking state into the loader
        self.load_background_image(self.gallery_images[target_index], prev_x_view[0], prev_y_view[0])

    def load_background_image(self, data_bundle: dict, restore_x: float = 0.0, restore_y: float = 0.0):
        pil_image = data_bundle["bg"]
        try:
            self.current_bg_pil = pil_image.copy()
            self.bg_width, self.bg_height = self.current_bg_pil.size
            target_width = self.bg_width + 150
            target_height = self.bg_height + 50

            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            if platform.system() == "Linux":
                try:
                    output = subprocess.check_output(
                        "xrandr | grep -E ' connected (primary )?[0-9]+x[0-9]+'",
                        shell=True,
                        stderr=subprocess.DEVNULL
                    ).decode("utf-8")
                    for part in output.split():
                        if "x" in part and "+" in part:
                            res_str = part.split("+")
                            dims = res_str[0].split("x")
                            screen_width = int(dims[0])
                            screen_height = int(dims[1])
                            break
                except Exception:
                    pass

            top_level = self.winfo_toplevel()
            self.canvas_container.pack_propagate(True)
            
            # Max dimensions are dynamically capped at 85% of display hardware configurations
            max_safe_w = int(screen_width * 0.85) if screen_width > 0 else 1280
            max_safe_h = int(screen_height * 0.85) if screen_height > 0 else 720
            
            final_width = min(target_width, max_safe_w)
            final_height = min(target_height, max_safe_h)
            
            # Only apply regular bounds geometries changes if not intentionally toggled into Fullscreen
            if not self.is_fullscreen:
                top_level.geometry(f"{final_width}x{final_height}")
            
            # If zoom_scale is -1.0, this is the first boot initialization: trigger an auto-fit layout
            if self.zoom_scale == -1.0:
                view_h = final_height - 50
                self.zoom_scale = max(0.15, view_h / self.bg_height)
                should_center_fresh = True
            else:
                should_center_fresh = False
            
            self.canvas.delete("all")
            self.layer_image_refs.clear()
            self.active_layer_id = None
            
            self.current_bg_pil = pil_image.copy()
            self.bg_width, self.bg_height = self.current_bg_pil.size
            
            self.render_zoomed_view()

            # Allocates a single-channel tracking array completely mapped to base resolution pixels
            self.manual_mask_surface = np.zeros((self.bg_height, self.bg_width), dtype=np.uint8)
            self.run_inpaint_btn.config(state="disabled")

            if should_center_fresh:
                self.master.after(100, self.center_canvas_viewport)
            else:
                self.master.after(100, lambda: self._apply_viewport_scroll(restore_x, restore_y))

            self.update_window_title_with_filename(data_bundle)

        except Exception as e:
            print(f"Error executing background transition: {e}")

    def _apply_viewport_scroll(self, scroll_x: float, scroll_y: float):
        """Restores explicit scroll fractions directly to the canvas axes."""
        try:
            self.canvas.xview_moveto(scroll_x)
            self.canvas.yview_moveto(scroll_y)
        except Exception:
            pass

    def load_draggable_text_image_layers(self, data_bundle: dict):
        self._tk_overlay_cache.clear()
        layers = data_bundle.get("layers", [])
        coords = data_bundle.get("coords", [])

        for i, pil_layer in enumerate(layers):
            if not isinstance(pil_layer, Image.Image):
                continue
            try:
                tk_layer_img = ImageTk.PhotoImage(pil_layer)
                self._tk_overlay_cache.append(tk_layer_img)

                x, y = coords[i] if i < len(coords) else (50, 50)
                layer_tag = f"layer_idx_{i}"
                layer_id = self.canvas.create_image(
                    x, y, image=tk_layer_img, anchor="nw", tags=("draggable_layer", layer_tag)
                )

                self.layer_image_refs[layer_id] = tk_layer_img
                self.active_layer_id = layer_id

                self.canvas.tag_bind(layer_id, "<ButtonPress-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_start(event, lid, lidx))
                self.canvas.tag_bind(layer_id, "<B1-Motion>", self._on_dragging)
                self.canvas.tag_bind(layer_id, "<ButtonRelease-1>", lambda event, lid=layer_id, lidx=i: self._on_drag_end(event, lid, lidx))
            except Exception as e:
                print(f"Error rendering layer {i}: {e}")

    def update_window_title_with_filename(self, data_bundle: dict):
        """Extracts the base file name from the image metadata and updates the root window title."""
        try:
            # 1. Grab whatever path string identifier is available in the payload bundle
            path_str = data_bundle.get("export_path", "") or data_bundle.get("dest", "")
            
            if path_str:
                # 2. Extract just the file name (e.g., 'page_01.png' from '/path/to/page_01.png')
                base_name = os.path.basename(os.path.normpath(str(path_str)))
                new_title = f"Live Editor - {base_name}"
            else:
                new_title = f"Live Editor - Image {self.current_img_index + 1}"
                
            # 3. Apply the string directly to the root window top level title bar
            self.winfo_toplevel().title(new_title)
        except Exception as e:
            print(f"Error updating window title: {e}", file=sys.stderr)

    def _on_arrow_left_press(self, event):
        if not self.gallery_images or self.current_img_index == -1:
            return
        new_index = (self.current_img_index - 1) % len(self.gallery_images)
        self.load_background_image_by_index(new_index)

    def _on_arrow_right_press(self, event):
        if not self.gallery_images or self.current_img_index == -1:
            return
        new_index = (self.current_img_index + 1) % len(self.gallery_images)
        self.load_background_image_by_index(new_index)

    def _on_sidebar_mousewheel(self, event):
        # Base speed modifier for the sidebar layout
        scroll_speed = 1
        
        if getattr(event, 'num', None) in (4, 5):
            delta = -1 * scroll_speed if event.num == 4 else 1 * scroll_speed
        else:
            delta = -1 * scroll_speed if event.delta > 0 else 1 * scroll_speed
        self.gallery_canvas.yview_scroll(delta, "units")

    def _on_main_canvas_mousewheel(self, event):
        # ----------------------------------------------------
        # SPEED TUNING PARAMETER
        # ----------------------------------------------------
        # Increase this number to scroll faster, decrease it to scroll slower.
        scroll_speed = 1 

        if getattr(event, 'num', None) in (4, 5):
            delta = -1 * scroll_speed if event.num == 4 else 1 * scroll_speed
        elif hasattr(event, 'delta') and event.delta != 0:
            # Flatten Windows/macOS high-resolution delta counts down to base steps
            base_step = int(event.delta / abs(event.delta)) # Yields a clean 1 or -1
            delta = -1 * base_step * scroll_speed
        else:
            delta = -1 * scroll_speed if event.delta > 0 else 1 * scroll_speed

        # Execute smooth, minor unit shifts along the active axis
        if (event.state & 0x1) != 0:
            self.canvas.xview_scroll(delta, "units")
        else:
            self.canvas.yview_scroll(delta, "units")

    def _on_property_font_dropdown_changed(self, event=None):
        """
        Triggered dynamically whenever a user updates a font face option 
        inside the open properties layout dialogue context window.
        """
        chosen_display_name = self.layer_font_combobox.get()
        font_file_path = self.available_fonts_dict.get(chosen_display_name)

        if font_file_path and os.path.exists(font_file_path):
            print(f"[UI Layout Engine] Swapping typography selection to: '{font_file_path}'")
            
            # 1. Update the core FreeType font selections cache mapping list configurations 
            # and flush the associated inner loop LRU glyph measurement caching dictionaries
            text_render.set_font(font_file_path)
            self.current_font_path = font_file_path
            # 2. Re-trigger layout calculations across all currently indexed image blocks
            '''if self.current_img_index != -1 and self.gallery_images:
                active_bundle = self.gallery_images[self.current_img_index]
                
                # Fetch original source path matching this specific image block
                path_string = active_bundle.get("export_path", "")
                
                # Re-invoke append_image to dynamically typeset all boxes on the active canvas viewport
                self.append_image(active_bundle, path=path_string)'''

    def scheduled_render(self):
        self.current_bg_pil = self.gallery_images[self.current_img_index]["bg"].copy()
        self.render_zoomed_view()

    def schedule_render(self):
        """
        Debounces the high-quality BILINEAR render. 
        Resets every time the user scrolls and fires 300ms after scrolling ends.
        """
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        # 300 milliseconds feels instantaneous to the user but groups scroll ticks perfectly
        self.timer = threading.Timer(0.3, self.scheduled_render)
        self.timer.start()
        
    def _on_ctrl_mousewheel_zoom(self, event):
        """
        Intercepts Control + Scroll events to handle smooth real-time image scaling.
        Guarantees mouse focus points anchor accurately to prevent canvas drifting.
        """
        if self.current_img_index == -1:
            return "break"

        # 1. Trace the precise location on the image map directly underneath the cursor tip
        mouse_canvas_x = self.canvas.canvasx(event.x)
        mouse_canvas_y = self.canvas.canvasy(event.y)

        # Convert canvas position coordinates down to 100% stable base pixel numbers
        native_target_x = mouse_canvas_x / self.zoom_scale
        native_target_y = mouse_canvas_y / self.zoom_scale

        # 2. Determine mouse wheel rotation steps direction
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.03
        else:
            zoom_factor = 0.97
            
        proposed_scale = self.zoom_scale * zoom_factor

        # Bound scaling limits defensively
        if proposed_scale < self.min_zoom:
            proposed_scale = self.min_zoom
        elif proposed_scale > self.max_zoom:
            proposed_scale = self.max_zoom

        if proposed_scale == self.zoom_scale:
            return "break"

        # Apply updated scale configuration tracking metrics
        self.zoom_scale = proposed_scale

        # 3. Fire fast NEAREST resampling while the user is actively zooming
        self.render_zoomed_view(Image.Resampling.NEAREST)
        
        # 4. Debounce the high-quality BILINEAR sharpen pass
        self.schedule_render()

        # 5. Readjust canvas viewport windows to keep your cursor locked over the original location
        # This prevents the image elements from drifting across the view panel during adjustments
        new_mouse_canvas_x = native_target_x * self.zoom_scale
        new_mouse_canvas_y = native_target_y * self.zoom_scale
        
        # Calculate exact pixel drift differentials
        shift_x = new_mouse_canvas_x - mouse_canvas_x
        shift_y = new_mouse_canvas_y - mouse_canvas_y
        
        # Scroll the viewport window positions dynamically to cancel out position drift
        # self.canvas.xview_scroll(int(shift_x), "units")
        # self.canvas.yview_scroll(int(shift_y), "units")

        if self.active_layer_index is not None:
            self.open_or_update_property_box(self.active_layer_index)

        return "break"

    def _update_zoom_scroll_region(self):
        """Calculates and applies updated navigation constraints based on the active scale factor."""
        if not self.current_bg_pil:
            return
        w, h = self.current_bg_pil.size
        # Stretch target tracking boundaries relative to our zoom multiplier vector dimensions
        scaled_w = int(w * self.zoom_scale)
        scaled_h = int(h * self.zoom_scale)
        
        # Add a light padding margin border to keep canvas handling clean at high zoom values
        #self.canvas.config(scrollregion=(-100, -100, scaled_w + 100, scaled_h + 100))

    def register_close_protocol(self):
        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_close_cleanup)

    def on_close_cleanup(self):
        try:
            self.close_property_box()
            self.canvas.delete("all")
            self.gallery_canvas.delete("all")
            self.layer_image_refs.clear()
            self._tk_overlay_cache.clear()
            self.thumbnail_refs.clear()
            self.tk_bg_image = None
            self.current_bg_pil = None
            self.winfo_toplevel().destroy()
        except Exception as e:
            print(f"Error during widget resource cleanup: {e}")

    def _on_dragging(self, event):
        # 1. Defensively guard against missing active object references
        if self.active_layer_id is None or self.active_layer_index is None:
            return

        # 2. Calculate raw window pixel distance variations
        delta_window_x = event.x - self.drag_data["x"]
        delta_window_y = event.y - self.drag_data["y"]
        
        # 3. Physically move the active canvas visual object layer
        self.canvas.move(self.active_layer_id, delta_window_x, delta_window_y)
        
        # Update mouse anchoring indices continuously for the next drag frame delta
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        # 4. CRITICAL FIX: Extract current visual coordinates and sync directly to memory mid-drag
        current_coords = self.canvas.coords(self.active_layer_id)
        if current_coords and self.current_img_index != -1:
            active_bundle = self.gallery_images[self.current_img_index]
            coords = active_bundle.setdefault("coords", [])
            
            # Read real viewport alignment offsets dynamically
            view_w = self.canvas_container.winfo_width()
            view_h = self.canvas_container.winfo_height()
            if view_w <= 1: view_w = 1280
            if view_h <= 1: view_h = 720

            scaled_w = int(self.bg_width * self.zoom_scale)
            scaled_h = int(self.bg_height * self.zoom_scale)
            
            offset_x = max(0, (view_w - scaled_w) // 2)
            offset_y = max(0, (view_h - scaled_h) // 2)
                
            # Inverse project visual canvas pixel positions back into native image coordinates
            native_x = int(round((current_coords[0] - offset_x) / self.zoom_scale))
            native_y = int(round((current_coords[1] - offset_y) / self.zoom_scale))
            
            if self.active_layer_index < len(coords):
                coords[self.active_layer_index] = (native_x, native_y)

        # 5. Redraw the wireframe bounding layout boxes matching the new synced position coordinates
        self.draw_inline_bounding_boxes()

    def _on_drag_start(self, event, target_id, layer_index):
        if event.state & 0x0004:
            return

        self.open_or_update_property_box(layer_index, event)

        self.active_layer_id = target_id
        self.active_layer_index = layer_index
        self.canvas.itemconfig(target_id, state="normal")
        
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
        return "break"  # Prevents event bubbling into the background canvas layer

    def _on_drag_end(self, event, target_id, layer_index):
        """Processes final positioning cleanup once manual drag operation completes."""
        if self.current_img_index == -1:
            return

        current_coords = self.canvas.coords(target_id)
        if current_coords:
            active_bundle = self.gallery_images[self.current_img_index]
            coords = active_bundle.setdefault("coords", [])
            while len(coords) <= layer_index:
                coords.append((50, 50))
                
            view_w = self.canvas_container.winfo_width()
            view_h = self.canvas_container.winfo_height()
            if view_w <= 1: view_w = 1280
            if view_h <= 1: view_h = 720

            scaled_w = int(self.bg_width * self.zoom_scale)
            scaled_h = int(self.bg_height * self.zoom_scale)
            
            offset_x = max(0, (view_w - scaled_w) // 2)
            offset_y = max(0, (view_h - scaled_h) // 2)
                
            native_x = int(round((current_coords[0] - offset_x) / self.zoom_scale))
            native_y = int(round((current_coords[1] - offset_y) / self.zoom_scale))
            coords[layer_index] = (native_x, native_y)
            
        # Update the property window immediately on drop to display precise destination points
        if self.active_layer_index is not None:
            self.open_or_update_property_box(self.active_layer_index)

    def center_canvas_viewport(self):
        """
        Adjusts scrollbars to center the active area when the image overflows the window view.
        """
        if self.current_img_index == -1 or not self.current_bg_pil:
            return

        self.update_idletasks()
        canvas_visible_w = self.canvas_container.winfo_width()
        canvas_visible_h = self.canvas_container.winfo_height()

        scaled_img_w = int(self.bg_width * self.zoom_scale)
        scaled_img_h = int(self.bg_height * self.zoom_scale)

        # Center Horizontally (Only if image overflows visible pane)
        if scaled_img_w > canvas_visible_w:
            center_x_fraction = ((scaled_img_w - canvas_visible_w) / 2) / scaled_img_w
            self.canvas.xview_moveto(center_x_fraction)
        else:
            self.canvas.xview_moveto(0)

        # Center Vertically (Only if image overflows visible pane)
        if scaled_img_h > canvas_visible_h:
            center_y_fraction = ((scaled_img_h - canvas_visible_h) / 2) / scaled_img_h
            self.canvas.yview_moveto(center_y_fraction)
        else:
            self.canvas.yview_moveto(0)

    def _on_window_resize_event(self, event):
        """Passively tracking changes without breaking canvas zoom layouts."""
        pass
    
    def trigger_fit_to_window(self):
        """Calculates the best zoom factor to fit the image to the window seamlessly."""
        if self.current_img_index == -1 or not self.current_bg_pil:
            return

        window_w = self.canvas_container.winfo_width()
        window_h = self.canvas_container.winfo_height()
        if window_w <= 1 or window_h <= 1:
            return

        view_w, view_h = max(window_w - 4, 50), max(window_h - 4, 50)
        img_w, img_h = self.current_bg_pil.size
        
        # Calculate aspect ratio boundaries
        img_aspect = img_w / img_h
        view_aspect = view_w / view_h

        if img_aspect > view_aspect:
            proposed_scale = view_w / img_w
        else:
            proposed_scale = view_h / img_h

        # Enforce bounds limits cleanly onto your core zoom tracker
        self.zoom_scale = max(self.min_zoom, min(proposed_scale, self.max_zoom))
        
        # Rerender using standard zoom paths so layers are fully active and editable
        self.render_zoomed_view()
        self.center_canvas_viewport()

        if self.active_layer_index is not None:
            self.open_or_update_property_box(self.active_layer_index)
    
    def _on_canvas_background_click(self, event):
        """
        Triggered when clicking empty space or the background image canvas.
        Clears the selection state, reverts highlighting, and closes the property window.
        """
        # Ignore clicks if the user was using a control modifier (like Ctrl+Click zoom steps)
        if event.state & 0x0004:
            return

        # Use canvas coordinate transformations to determine what object is underneath the cursor
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        clicked_items = self.canvas.find_overlapping(canvas_x - 1, canvas_y - 1, canvas_x + 1, canvas_y + 1)

        # Check if any overlapping item belongs to a text layer element
        for item_id in clicked_items:
            tags = self.canvas.gettags(item_id)
            if "draggable_layer" in tags:
                return  # A text block was clicked! Let drag initialization routines handle it safely.

        # Clear active selection indexing trackers if clicking true empty space
        self.active_layer_id = None
        self.active_layer_index = None

        # Cleanly dismantle and close the active property box frame workflow
        self.close_property_box()

    def execute_manual_lama_mpe_inpaint(self):
        """Pulls drawing contours and processes a background worker thread to execute LamaMPEInpainter."""
        if self.current_img_index == -1 or self.manual_mask_surface is None:
            return
        if self.manual_mask_surface.sum() == 0:
            return

        self.run_inpaint_btn.config(state="disabled", text="Inpainting...")
        self.inpaint_toggle_btn.config(state="disabled")

        # Gather background matrix parameters cleanly
        bundle = self.gallery_images[self.current_img_index]
        native_rgb = bundle["img_inpainted"]  # Original clean source frame array matrix

        threading.Thread(
            target=self._async_lama_mpe_worker_pipeline,
            args=(native_rgb, self.manual_mask_surface.copy(), self.current_img_index),
            daemon=True
        ).start()

    def _async_lama_mpe_worker_pipeline(self, rgb_matrix: np.ndarray, mask_matrix: np.ndarray, target_idx: int):
        try:
            from manga_translator.inpainting.inpainting_lama_mpe import LamaMPEInpainter
            from manga_translator.utils import Context
            import asyncio
            
            print("[Inpainter Pipeline] Initializing LamaMPEInpainter runtime hardware structures...")
            
            # 1. Instantiate the module interface cleanly
            inpainter_instance = LamaMPEInpainter()
            
            # 2. Package property parameters block mapping layouts natively
            dummy_ctx = Context()
            dummy_ctx.img_rgb = rgb_matrix
            
            # Define an internal coroutine wrapper to execute the async load and run pipelines sequentially
            async def run_inpainting_lifecycle():
                # Explicitly load the model layers and weights into CPU architecture
                print("[Inpainter Pipeline] Loading model weights and computing neural network fields...")
                await inpainter_instance.load('cpu')
                
                # Perform the forward pass calculation over the targeted hand-drawn elements matrices
                return await inpainter_instance.inpaint(rgb_matrix, mask_matrix, dummy_ctx)

            # 3. Spin up the thread-isolated asyncio engine loop execution context
            inpainted_result_matrix = asyncio.run(run_inpainting_lifecycle())

            if inpainted_result_matrix is not None:
                self.master.after(0, self._safe_commit_inpainted_surface_callback, inpainted_result_matrix, target_idx)
            else:
                raise RuntimeError("Inpainter framework layer returned empty matrix reference array blocks.")
                
        except Exception as err:
            print(f"[IPC Inpainter Thread Crash Fatal Error]: {err}", file=sys.stderr)
            self.master.after(0, lambda: self.run_inpaint_btn.config(state="normal", text="Run Inpaint"))
            self.master.after(0, lambda: self.inpaint_toggle_btn.config(state="normal"))

    def _safe_commit_inpainted_surface_callback(self, fresh_inpainted_rgb: np.ndarray, target_idx: int):
        """Callback thread sync handler loop to safely write fresh image data back into the gallery bundle."""
        if target_idx < len(self.gallery_images):
            bundle = self.gallery_images[target_idx]
            
            # 1. Overwrite the baseline image matrix frames with the fresh inpainted results data
            bundle["img_inpainted"] = fresh_inpainted_rgb
            new_bg_pil = Image.fromarray(fresh_inpainted_rgb)
            bundle["bg"] = new_bg_pil
            
            # 2. Ensure the currently displayed PIL image reference
            # is updated immediately if the user is working on this active page index
            if target_idx == self.current_img_index:
                self.current_bg_pil = new_bg_pil.copy()
            
            # 3. Reset drawing tracking data surfaces entirely
            self.manual_mask_surface = np.zeros((self.bg_height, self.bg_width), dtype=np.uint8)
            
            # 4. Restore interface button state parameters
            self.run_inpaint_btn.config(state="disabled", text="Run Inpaint")
            self.inpaint_toggle_btn.config(state="normal")
            
            # 5. Cleanly shut down inpaint mode state parameters
            # This handles the cache clearing and runs render_zoomed_view() for you
            if self.inpainter_mode_active:
                self.toggle_inpainter_mode()
            else:
                if hasattr(self, '_tk_overlay_cache'):
                    self._tk_overlay_cache.clear()
                self.render_zoomed_view()
                
            print(f"[Inpainter Pipeline] Manual LamaMPE clean loop finished on frame index #{target_idx + 1} successfully.")

