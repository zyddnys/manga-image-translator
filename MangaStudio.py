# ===================================================================================
# Manga Translation Studio - v6.0 (The Workshop Update)
#
# Author: User & Gemini Collaboration
#
# Description: This major update (v6) transforms the studio into a powerful workshop
#              by introducing a dedicated "Extra Settings" tab for advanced tasks
#              and project management.
#
# Key Features of v6:
#   - Preset Manager: Save, load, and manage your favorite setting configurations.
#   - Special Tasks Workshop: Perform standalone tasks like RAW Output, Upscaling,
#     or Colorization with their own dedicated and independent settings.
#   - Smart Colorization: Features an intelligent process that automatically
#     calculates the optimal upscale ratio to restore original image dimensions.
#   - Centralized File Structure: All application data (profiles, configs, temp files)
#     is now neatly organized into a single "MangaStudio_Data" directory.
#   - v5 Foundation: All stable features from v5, including the full configuration
#     UI, Model Manager, and Visual Compare tab, are retained and enhanced.
#
# ===================================================================================

import os
import json
import time
import shutil
import threading
import subprocess
import traceback
import copy
from tkinter import filedialog, messagebox, colorchooser, font
import customtkinter as ctk
from CTkToolTip import CTkToolTip
from PIL import Image, ImageTk
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    messagebox.showerror(
        "Missing Library",
        "The tkinterdnd2 library was not found.\n"
        "Please install it using: 'pip install tkinterdnd2'"
    )
    exit()

# --- PROJECT CONSTANTS ---

# Model and option lists derived from the config-help.txt output.
# We use "Disabled" as a user-friendly key, which is later converted
# to 'none' or null when generating the JSON config.
MODEL_OPTIONS = {
    "detector": ["paddle", "dbconvnext", "ctd", "craft", "Disabled"],
    "inpainter": ["lama_large", "lama_mpe", "sd", "original", "Disabled"],
    "ocr": ["mocr", "48px", "32px", "48px_ctc"], # mocr is now the default and recommended
    "translator": [
        # Separators are handled in the UI to be non-selectable
        "--- OFFLINE MODELS (No API Key) ---",
        "sugoi",
        "m2m100",
        "m2m100_big",
        "nllb",
        "nllb_big",
        "mbart50",
        "jparacrawl",
        "jparacrawl_big",
        "qwen2",
        "qwen2_big",
        "offline",
        "--- API-BASED (Requires Setup) ---",
        "deepl",
        "gemini",
        "deepseek",
        "groq",
        "youdao",
        "baidu",
        "caiyun",
        "sakura",
        "papago",
        "openai", # Alias for chatgpt
        "custom_openai",
        "--- OTHER ACTIONS ---",
        "original", # Keep original text
        "none",     # Create textless (raw) output
    ],
    "upscaler": ["esrgan", "waifu2x", "4xultrasharp", "Disabled"],
    "renderer": ["default", "manga2eng", "manga2eng_pillow", "Disabled"],
    "alignment": ["auto", "left", "center", "right"],
    "direction": ["auto", "horizontal", "vertical"],
    "inpainting_precision": ["fp32", "fp16", "bf16"],
    "colorizer": ["none", "mc2"],
}


# In MangaStudio.py

# --- REPLACE THE ENTIRE LANGUAGES DICTIONARY WITH THIS ---
LANGUAGES = {
    "Auto-Detect": "auto",
    "English": "ENG",
    "Turkish": "TRK",
    "Japanese": "JPN",
    "Korean": "KOR",
    "Simplified Chinese": "CHS",
    "Traditional Chinese": "CHT",
    "Spanish": "ESP",
    "French": "FRA",
    "German": "DEU",
    "Russian": "RUS",
    "Portuguese (Brazilian)": "PTB",
    "Italian": "ITA",
    "Polish": "PLK",
    "Dutch": "NLD",
    "Czech": "CSY",
    "Hungarian": "HUN",
    "Romanian": "ROM",
    "Ukrainian": "UKR",
    "Vietnamese": "VIN",
    "Arabic": "ARA",
    "Serbian": "SRP",
    "Croatian": "HRV",
    "Thai": "THA",
    "Indonesian": "IND",
    "Filipino (Tagalog)": "FIL"
}

TRANSLATOR_CAPABILITIES = {
    # --- OFFLINE MODELS ---
    "sugoi": {"source": ["JPN"], "target": ["ENG"]},
    "jparacrawl": {"source": ["JPN"], "target": ["ENG"]},
    "jparacrawl_big": {"source": ["JPN"], "target": ["ENG"]},
    "m2m100": {"source": ["all"], "target": ["all"]},
    "m2m100_big": {"source": ["all"], "target": ["all"]},
    "nllb": {"source": ["all"], "target": ["all"]},
    "nllb_big": {"source": ["all"], "target": ["all"]},
    "mbart50": {"source": ["all"], "target": ["all"]},
    "qwen2": {"source": ["all"], "target": ["all"]},
    "qwen2_big": {"source": ["all"], "target": ["all"]},
    "offline": {"source": ["all"], "target": ["all"]},
    # --- API-BASED MODELS ---
    "deepl": {"source": ["all"], "target": ["all"]},
    "gemini": {"source": ["all"], "target": ["all"]},
    "deepseek": {"source": ["all"], "target": ["all"]},
    "groq": {"source": ["all"], "target": ["all"]},
    "youdao": {"source": ["all"], "target": ["all"]},
    "baidu": {"source": ["all"], "target": ["all"]},
    "caiyun": {"source": ["all"], "target": ["all"]},
    "sakura": {"source": ["all"], "target": ["all"]},
    "papago": {"source": ["KOR"], "target": ["JPN", "ENG", "CHS", "CHT"]},
    "openai": {"source": ["all"], "target": ["all"]},
    "custom_openai": {"source": ["all"], "target": ["all"]},
    # --- OTHER ACTIONS (Always available) ---
    "original": {"source": ["any"], "target": ["any"]},
    "none": {"source": ["any"], "target": ["any"]},
}

# Default settings used when a new job is added.
# In class TranslatorStudioApp, at the top of the file:

# This dictionary now contains ALL possible settings with their default values.
FACTORY_SETTINGS = {
    # General & Translator
    "processing_device": "CPU", "source_lang": "auto", "target_lang": "ENG", "translator": "sugoi",
    "no_text_lang_skip": False, "skip_lang": "", "translator_chain": "",
    # Detector & OCR
    "detector": "default", "detection_size": 2048, "text_threshold": 0.5, "box_threshold": 0.7, "unclip_ratio": 2.3,
    "det_rotate": False, "det_auto_rotate": False, "det_invert": False, "det_gamma_correct": False,
    "ocr": "48px", "use_mocr_merge": False, "min_text_length": 0, "ignore_bubble": 0, "prob": None,
    # Image & Inpainter
    "inpainter": "lama_large", "inpainting_size": 2048, "inpainting_precision": "bf16",
    "upscaler": "esrgan", "upscale_ratio": "Disabled", "revert_upscaling": False,
    "colorizer": "none", "colorization_size": 576, "denoise_sigma": 30,
    # Render & Output
    "renderer": "default", "font_family": "Sans-serif", "font_color": "000000", "font_size": None,
    "font_size_offset": 0, "font_size_minimum": -1, "line_spacing": None, "alignment": "auto",
    "direction": "auto", "disable_font_border": False, "uppercase": False, "lowercase": False,
    "no_hyphenation": False, "rtl": False,
    "auto_rename": True, "backup_original": False, "overwrite_output": False, "filter_text": "",
    # Root level advanced
    "kernel_size": 3, "mask_dilation_offset": 0
}

SPECIAL_TASK_DEFAULTS = {
    "processing_device": "CPU",
    "raw": {
        "detection_size": 2048,
        "text_threshold": 0.5,
        "box_threshold": 0.7,
        "unclip_ratio": 2.3,
        "ignore_bubble": 0,
        # NOTE: We only need defaults for values controlled by UI widgets.
        "inpainter_model": "lama_large",
        "inpainting_size": 1024
    },
    "upscale": {
        "model": "esrgan",
        "ratio": 2.0,
        "revert_size": False
    },
    "colorize": {
        "model": "mc2",
        "size": 576,
        "denoise": 30,
        "upscaler_model": "esrgan",
        "final_ratio": 1.0,
        "ignore_bubble": 0,
        "text_threshold": 0.9 
    }
}

APP_DATA_DIR = os.path.join(os.path.dirname(__file__), "MangaStudio_Data")

CONFIG_FILE = os.path.join(APP_DATA_DIR, "studio_config.json")
PROFILES_DIR = os.path.join(APP_DATA_DIR, "profiles")
TEMP_DIR = os.path.join(APP_DATA_DIR, "temp")

# --- MAIN APPLICATION CLASSES ---

class TkinterDnD_App(ctk.CTk, TkinterDnD.DnDWrapper):
    """Base CTk class with added Drag & Drop functionality."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

# ===================================================================================
# Model Manager Class (Hybrid Approach)
# ===================================================================================
class ModelManager:
    """
    Manages all models by providing a centralized info dictionary,
    triggering the backend for downloads, and handling local file operations.
    """
    def __init__(self, app_instance, models_base_dir="models"):
        self.app = app_instance
        self.base_dir = models_base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.MODELS_INFO = {
            "Detector": {
                "Default Detector": {
                    "config_value": "default", "file_path": "detector/comic-text-detector.pt", "size": "98 MB",
                    "description": "General-purpose detector, recommended for most comics."
                },
                "CTD (Comic Text Detector)": {
                    "config_value": "ctd", "file_path": "detector/ctd.pt", "size": "145 MB",
                    "description": "Alternative detector, may perform better on non-standard layouts."
                },
                "Paddle OCR Detector": { "config_value": "paddle", "source": "Built-in", "description": "Fast detector, comes with the library." },
                "Craft Detector": { "config_value": "craft", "source": "Built-in", "description": "Another built-in option, less common for manga." }
            },
            "Translator": {
                "Sugoi Translator": {
                    "config_value": "sugoi", "check_path": "translator/sugoi-models", "size": "622 MB",
                    "description": "High-quality Japanese to English offline translator."
                },
                "JParacrawl (Base)": {
                    "config_value": "jparacrawl", "check_path": "translator/jparacrawl-base-models", "size": "523 MB",
                    "description": "Japanese to English model from the JParaCrawl project."
                },
                "JParacrawl (Big)": {
                    "config_value": "jparacrawl_big", "check_path": "translator/jparacrawl-big-models", "size": "1.51 GB",
                    "description": "A larger, more accurate version of the JParaCrawl model."
                },
                "M2M100 (418M)": {
                    "config_value": "m2m100", "check_path": "translator/m2m100_418m_ct2", "size": "1.09 GB",
                    "description": "Facebook's M2M100 model (418M parameters) for many-to-many translation."
                },
                "M2M100 (1.2B)": {
                    "config_value": "m2m100_big", "check_path": "translator/m2m100_1.2b_ct2", "size": "4.6 GB",
                    "description": "A larger, more accurate version of M2M100 (1.2B parameters)."
                },
                "NLLB-200 (Distilled 600M)": {
                    "config_value": "nllb", "source": "Hugging Face Cache", "description": "Facebook's NLLB model for over 200 languages. Downloads to a central cache, not 'models' folder."
                },
                 "Qwen2 (7B)": {
                    "config_value": "qwen2", "source": "Hugging Face Cache", "description": "Alibaba's powerful Qwen2 model (7B parameters). Downloads to a central cache."
                }
            },
            "Inpainter": {
                "Lama (Large)": {
                    "config_value": "lama_large", "file_path": "inpainter/lama-large.pt", "size": "300 MB",
                    "description": "High-quality text removal. The recommended default."
                },
                "Lama (MPE)": {
                    "config_value": "lama_mpe", "file_path": "inpainter/lama-mpe.pt", "size": "104 MB",
                    "description": "A smaller and faster version of the Lama inpainter."
                }
            },
            "OCR": {
                "Default OCR (48px)": {
                    "config_value": "48px", "file_path": "ocr/ocr-48px.pt", "size": "105 MB",
                    "description": "The standard OCR model, good for most languages."
                },
                "Manga OCR (mocr)": {
                    "config_value": "mocr", "file_path": "ocr/mocr.pt", "size": "120 MB",
                    "description": "Specialized OCR for manga, can be more accurate for Japanese."
                },
                "CTC-Based OCR": {
                    "config_value": "48px_ctc", "file_path": "ocr/ocr-ctc.pt", "size": "138 MB",
                    "description": "Alternative OCR model using CTC loss."
                }
            },
            "Upscaler": {
                "Real-ESRGAN (Anime)": {
                    "config_value": "esrgan", "file_path": "upscaler/RealESRGAN_x4plus_anime_6B.pth", "size": "65 MB",
                    "description": "High-quality 4x upscaler optimized for anime-style art."
                },
                 "4x-UltraSharp": {
                    "config_value": "4xultrasharp", "file_path": "upscaler/4x-UltraSharp.pth", "size": "64 MB",
                    "description": "Alternative sharp upscaler, good for detailed images."
                },
                "Waifu2x": { "config_value": "waifu2x", "source": "Built-in", "description": "A popular upscaler included with the library." }
            },
            "Colorizer": {
                 "Manga Colorization v2": {
                    "config_value": "mc2", "file_path": "colorizer/manga-colorization-v2-generator.zip", "size": "123 MB",
                    "description": "AI model to colorize black and white manga pages."
                 }
            }
        }

    def get_model_info(self, display_name):
        """Finds a model's info dictionary by its display name."""
        for category in self.MODELS_INFO.values():
            if display_name in category:
                return category[display_name]
        return None

    def get_target_path(self, display_name):
        """Gets the local path for a model, whether it's a file or a directory."""
        info = self.get_model_info(display_name)
        if not info: return None
        # Prioritize 'check_path' (for directories) then 'file_path'
        path_key = info.get("check_path") or info.get("file_path")
        return os.path.join(self.base_dir, path_key) if path_key else None

    def check_model_exists(self, display_name):
        """Checks if a model file or directory exists locally."""
        path = self.get_target_path(display_name)
        return os.path.exists(path) if path else False

    def trigger_download(self, display_name):
        """Triggers the backend tool to download a model."""
        info = self.get_model_info(display_name)
        if not info or not (info.get("file_path") or info.get("check_path")):
            self.app.log("ERROR", f"Cannot download '{display_name}': Model info not found or not downloadable.")
            return

        if self.check_model_exists(display_name):
            self.app.log("INFO", f"Model '{display_name}' already exists.")
            return

        def _run_download():
            self.app.log("PIPELINE", f"Attempting to download '{display_name}'...")
            dummy_image_path = os.path.join(os.path.dirname(self.app.python_executable), "dummy.png")
            with open(dummy_image_path, "w") as f: f.write("")

            trigger_arg = ""
            for cat_name, models in self.MODELS_INFO.items():
                if display_name in models:
                    trigger_arg = f"--{cat_name.lower()}"
                    break
            
            command = [
                self.app.python_executable, "-m", "manga_translator", "local",
                "-i", dummy_image_path, "--use-cpu", "--translator", "none",
                trigger_arg, info['config_value']
            ]

            try:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                for line in iter(process.stdout.readline, ''):
                    if line.strip(): self.app.log("RAW", line.strip())
                process.wait()
                self.app.log("SUCCESS", f"Download process for '{display_name}' finished.")
            except Exception as e:
                self.app.log("ERROR", f"Failed to trigger download for '{display_name}': {e}")
            finally:
                if os.path.exists(dummy_image_path): os.remove(dummy_image_path)
                self.app.after(0, self.app.refresh_model_manager_ui)

        threading.Thread(target=_run_download, daemon=True).start()

    def delete_model(self, display_name):
        """Deletes a model file OR directory."""
        if not self.check_model_exists(display_name):
            self.app.log("INFO", f"Model '{display_name}' not found, nothing to delete.")
            return

        path_to_delete = self.get_target_path(display_name)
        try:
            if os.path.isdir(path_to_delete):
                shutil.rmtree(path_to_delete) # Use shutil to delete a directory
                self.app.log("SUCCESS", f"Successfully deleted directory '{display_name}'.")
            elif os.path.isfile(path_to_delete):
                os.remove(path_to_delete) # Use os.remove for a single file
                self.app.log("SUCCESS", f"Successfully deleted file '{display_name}'.")
        except OSError as e:
            self.app.log("ERROR", f"Error deleting '{display_name}': {e}")
        finally:
            self.app.after(0, self.app.refresh_model_manager_ui)

class TranslatorStudioApp(TkinterDnD_App):
    """The main application class for Manga Translation Studio."""
    
    SETTING_GROUPS = {
        "General & Translator": ["processing_device", "source_lang", "target_lang", "translator", 
                                 "no_text_lang_skip", "translator_chain", "skip_lang"],
        "Detector & OCR": ["detector", "detection_size", "text_threshold", "box_threshold", 
                           "unclip_ratio", "det_rotate", "det_auto_rotate", "det_invert", 
                           "det_gamma_correct", "ocr", "use_mocr_merge", "min_text_length", 
                           "ignore_bubble", "prob"],
        "Image & Inpainter": ["inpainter", "inpainting_size", "inpainting_precision", "upscaler", 
                              "upscale_ratio", "revert_upscaling", "colorizer", 
                              "colorization_size", "denoise_sigma"],
        "Render & Output": ["renderer", "font_family", "font_color", "font_size", "font_size_offset", 
                            "font_size_minimum", "line_spacing", "alignment", "direction", 
                            "disable_font_border", "uppercase", "lowercase", "no_hyphenation", "rtl", 
                            "auto_rename", "backup_original", "overwrite_output", "filter_text", 
                            "kernel_size", "mask_dilation_offset"]
    }

    # 1. Initialization and Setup
    
    def __init__(self):
        try:
            super().__init__()
            self._initialize_app()
        except Exception:
            # Catch critical errors during startup
            print("---! APPLICATION FAILED TO START !---")
            traceback.print_exc()
            print("---------------------------------------")
            messagebox.showerror(
                "Fatal Startup Error",
                "The application could not start. Check the console for details."
            )
            if self:
                self.after(100, self.destroy)

    def _initialize_app(self):
        """Sets up the core components and states of the application."""
        os.makedirs(APP_DATA_DIR, exist_ok=True)
        self._setup_variables()
        self._load_global_settings() # This will now handle the initial theme setting.
        self._setup_window()
        self._create_main_layout()
        self.log("INFO", "UI layout created successfully.")
        self._populate_settings_panel()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.log("INFO", "Application startup complete. Ready for jobs.")
        self.bind("<Configure>", self._on_resize)
        self._setup_profiles()

    def _setup_variables(self):
        """Initializes all application-wide variables."""
        self.job_queue = []
        self.selected_job_id = None
        self.setting_widgets = {}
        self.is_running_pipeline = False
        self.pipeline_process = None
        self.enable_advanced_debug = ctk.BooleanVar(value=False)
        self._app_settings = {}
        
        self.special_task_widgets = {}
        # This will hold the values from the special task widgets
        self.special_task_settings = copy.deepcopy(SPECIAL_TASK_DEFAULTS)
        
        self.python_executable = os.path.join(os.path.dirname(__file__), 'venv', 'Scripts', 'python.exe')
        if not os.path.exists(self.python_executable):
            self.python_executable = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
        
        self.default_job_settings = FACTORY_SETTINGS.copy()
        self.model_manager = ModelManager(self)
        try:
            self.system_fonts = sorted(list(font.families()))
        except:
            self.system_fonts = ["Arial", "Times New Roman", "Courier New", "Sans-serif"]

        # --- VISUAL COMPARE STATE VARIABLES ---
        self.test_image_path = None
        self.original_pil_image = None
        self.translated_pil_image = None
        
        # This dictionary will hold persistent references to the CTkImage objects
        # This is the definitive fix for the garbage collection issue.
        self.image_references = {}

        # Camera state for both canvases
        self.canvas_zoom_level = 1.0  # Start at 100%
        self.canvas_pan_offset = [0, 0]  # [x, y]
        
        # For drag-to-pan functionality
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # For smart resizing
        self.last_known_size = (0, 0)
        self.resize_timer = None
        
        # --- ADD/UPDATE THESE LINES ---
        self.image_references = {} # For garbage collection fix
        self.original_canvas = None
        self.translated_canvas = None
        self._current_layout = None # To track layout changes
        # --- END OF ADDITION/UPDATE ---
        

    def _load_global_settings(self):
        """Loads application settings from the config file."""
        self.log("INFO", "Loading global settings...")
        self.default_job_settings = FACTORY_SETTINGS.copy()
        
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if "window_geometry" in config:
                    self.geometry(config.get("window_geometry", "1280x720"))

                if "default_job_settings" in config:
                    self.default_job_settings.update(config["default_job_settings"])
                    
                self.log("SUCCESS", f"Settings loaded from '{CONFIG_FILE}'.")
        except Exception as e:
            self.log("ERROR", f"Failed to load settings: {e}")

    def _setup_window(self):
        """Configures the main window settings."""
        self.title("🎌 Manga Translation Studio V6")
        self.geometry("1280x720")
        self.minsize(1100, 700)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        # Register the window for drop events
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self._handle_drop)

    # 2. Main UI Layout Creation
    
    def _create_main_layout(self):
        """Creates the main UI layout (left, right, bottom panels)."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Ensure methods are called on the correct 'self' instance
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()

    def _create_right_panel(self):
        """Creates the main right-side TabView for all major sections."""
        main_tabs = ctk.CTkTabview(self, corner_radius=10)
        main_tabs.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        # Add the main sections as tabs
        tab_config = main_tabs.add("Configuration ⚙️")
        tab_compare = main_tabs.add("Visual Compare 👁️") 
        tab_manager = main_tabs.add("Model Manager 📦")
        tab_log = main_tabs.add("Live Log 📊")
        
        # Populate each main tab
        self._create_settings_tab(tab_config)
        self._create_visual_compare_tab(tab_compare) 
        self._create_model_manager_tab(tab_manager)
        self._create_log_tab(tab_log)

    def _create_left_panel(self):
        """Creates the left panel containing the job list and controls."""
        left_panel = ctk.CTkFrame(self, width=350, corner_radius=0)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left_panel, text="Workflow Queue", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.job_list_frame = ctk.CTkScrollableFrame(left_panel, label_text="Drag & drop folders here")
        self.job_list_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # --- MODIFIED SECTION START ---
        self.job_controls_frame = ctk.CTkFrame(left_panel) # <-- MODIFIED: Added 'self.'
        self.job_controls_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew") # <-- MODIFIED
        self.job_controls_frame.grid_columnconfigure((0, 1, 2), weight=1) # <-- MODIFIED
        ctk.CTkButton(self.job_controls_frame, text="➕ Add", command=self._add_job).grid(row=0, column=0, padx=2, pady=5, sticky="ew") # <-- MODIFIED
        ctk.CTkButton(self.job_controls_frame, text="🗑️ Remove", command=self._remove_selected_job).grid(row=0, column=1, padx=2, pady=5, sticky="ew") # <-- MODIFIED
        ctk.CTkButton(self.job_controls_frame, text="🧹 Clear", command=self._clear_all_jobs).grid(row=0, column=2, padx=2, pady=5, sticky="ew") # <-- MODIFIED
        
        self.reorder_frame = ctk.CTkFrame(left_panel) # <-- MODIFIED: Added 'self.'
        self.reorder_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew") # <-- MODIFIED
        self.reorder_frame.grid_columnconfigure((0, 1), weight=1) # <-- MODIFIED
        ctk.CTkButton(self.reorder_frame, text="↑ Move Up", command=lambda: self._move_job("up")).grid(row=0, column=0, padx=2, pady=5, sticky="ew") # <-- MODIFIED
        ctk.CTkButton(self.reorder_frame, text="↓ Move Down", command=lambda: self._move_job("down")).grid(row=0, column=1, padx=2, pady=5, sticky="ew") # <-- MODIFIED
        # --- MODIFIED SECTION END ---

    def _create_bottom_panel(self):
        """Creates the bottom panel with the progress bar and start/stop buttons."""
        bottom_bar = ctk.CTkFrame(self, height=80, fg_color="transparent")
        bottom_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        bottom_bar.grid_columnconfigure(0, weight=1)
        
        progress_frame = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        progress_frame.grid(row=0, column=0, sticky="ew", padx=20)
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.progress_label.grid(row=0, column=0, sticky="w")
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, sticky="ew")
        
        controls_frame = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        controls_frame.grid(row=0, column=1, sticky="e")
        self.start_button = ctk.CTkButton(controls_frame, text="▶️ START PIPELINE", height=40, font=ctk.CTkFont(size=14, weight="bold"), command=self._start_pipeline_thread)
        self.start_button.pack(side="left", padx=10)
        self.stop_button = ctk.CTkButton(controls_frame, text="⏹️ STOP", height=40, state="disabled", fg_color="#D32F2F", hover_color="#B71C1C", command=self._stop_pipeline)
        self.stop_button.pack(side="left", padx=10)

    # 3. Tab Creation Methods

    def _create_settings_tab(self, tab):
        """Creates the main tabbed interface for all settings."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.settings_tabs = ctk.CTkTabview(tab, corner_radius=10)
        self.settings_tabs.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tab_general = self.settings_tabs.add("General & Translator")
        tab_ocr = self.settings_tabs.add("Detector & OCR")
        tab_image = self.settings_tabs.add("Image & Inpainter")
        tab_render = self.settings_tabs.add("Render & Output")
        tab_extra = self.settings_tabs.add("Extra Settings")

        self.setting_widgets = {}

        self._create_general_tab(tab_general)
        self._create_ocr_tab(tab_ocr)
        self._create_image_tab(tab_image)
        self._create_render_tab(tab_render)
        self._create_extra_settings_tab(tab_extra)
        
        reset_frame = ctk.CTkFrame(tab)
        reset_frame.grid(row=1, column=0, sticky="e", padx=5, pady=(10, 5))

        reset_tab_button = ctk.CTkButton(reset_frame, text="Reset Current Tab", command=self._reset_current_tab_settings)
        reset_tab_button.pack(side="left", padx=(0, 5))

        reset_all_button = ctk.CTkButton(reset_frame, text="Reset All Settings...", fg_color="#D32F2F", hover_color="#B71C1C", command=self._reset_all_settings)
        reset_all_button.pack(side="left")

    def _create_general_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(1, weight=1)
        row_idx = {'row': 0}

        self._create_setting_row(scroll_frame, 'processing_device', "Processing Device:", "segmented", values=["CPU", "NVIDIA GPU"], tooltip="CPU is universal but slower. NVIDIA GPU is much faster but requires a compatible card and setup.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'source_lang', "Source Language:", "option", values=list(LANGUAGES.keys()), command=lambda v: self._on_setting_change('source_lang', LANGUAGES[v]), row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'target_lang', "Target Language:", "option", values=[k for k in LANGUAGES if k != "Auto-Detect"], command=lambda v: self._on_setting_change('target_lang', LANGUAGES[v]), row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'translator', "Translator Model:", "option", values=MODEL_OPTIONS['translator'], command=self._on_translator_select, row_idx_ref=row_idx)
        
        ctk.CTkLabel(scroll_frame, text="Advanced Translator Settings", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(15,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'no_text_lang_skip', "Translate Same Language:", "checkbox", tooltip="If checked, the translator will run even if the source and target languages are the same.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'translator_chain', "Translator Chain:", "entry", special_props={"placeholder_text": "e.g., sugoi > deepl"}, tooltip="Chain multiple translators. The output of the first is the input to the second.", row_idx_ref=row_idx)
        
        skip_frame = ctk.CTkFrame(scroll_frame)
        skip_frame.grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=10, sticky="nsew"); row_idx['row'] += 1
        ctk.CTkLabel(skip_frame, text="Skip Translating from these Languages:").pack(anchor="w", padx=10, pady=(5,0))
        lang_scroll_frame = ctk.CTkScrollableFrame(skip_frame, fg_color="transparent", height=100)
        lang_scroll_frame.pack(fill="x", expand=True, padx=5)
        lang_scroll_frame.grid_columnconfigure((0, 1), weight=1)
        self.setting_widgets['skip_lang_checkboxes'] = {}
        lang_items = [lang for lang in LANGUAGES.items() if lang[1] != 'auto']
        for i, (name, code) in enumerate(lang_items):
            cb = ctk.CTkCheckBox(lang_scroll_frame, text=name)
            cb.configure(command=lambda c=code, widget=cb: self._on_setting_change('skip_lang', (c, widget.get())))
            cb.grid(row=i // 2, column=i % 2, padx=10, pady=5, sticky="w")
            self.setting_widgets['skip_lang_checkboxes'][code] = cb

    def _create_ocr_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(1, weight=1)
        row_idx = {'row': 0}

        ctk.CTkLabel(scroll_frame, text="Core Settings", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(10,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'detector', "Detector Model:", "option", values=MODEL_OPTIONS['detector'], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'detection_size', "Detection Size:", "slider", special_props={"from_": 512, "to": 4096, "number_of_steps": 56}, tooltip="Image resolution for text detection.\nLarger values find smaller text but are slower.\nDefault: 2048", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'ocr', "OCR Model:", "option", values=MODEL_OPTIONS['ocr'], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'ignore_bubble', "Ignore Small Bubbles:", "slider", special_props={"from_": 0, "to": 500, "number_of_steps": 50}, tooltip="Ignores any detected text bubbles smaller than this pixel area. Useful for filtering out noise.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'use_mocr_merge', "Use mOCR Merge:", "checkbox", tooltip="Reads text with multiple OCR engines and merges the results for higher accuracy. Great for difficult fonts, but slightly slower.", row_idx_ref=row_idx)

        ctk.CTkLabel(scroll_frame, text="Advanced Settings", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(15,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'text_threshold', "Text Threshold:", "slider", special_props={"from_": 10, "to": 90, "number_of_steps": 80}, command=lambda v: self._on_setting_change('text_threshold', round(float(v)/100.0, 2)), tooltip="How confident the model must be to mark an area as text.\nLower: Finds more text, may include noise.\nHigher: Finds only clear text.\nDefault: 0.5", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'box_threshold', "Box Threshold:", "slider", special_props={"from_": 10, "to": 90, "number_of_steps": 80}, command=lambda v: self._on_setting_change('box_threshold', round(float(v)/100.0, 2)), tooltip="How confident the model must be to finalize the bounding box.\nLower: Looser boxes.\nHigher: Tighter, more precise boxes.\nDefault: 0.7", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'unclip_ratio', "Unclip Ratio:", "slider", 
                                 special_props={"from_": 10, "to": 50, "number_of_steps": 40}, 
                                 command=lambda v: self._on_setting_change('unclip_ratio', round(float(v)/10.0, 1)), 
                                 tooltip="How much to expand the detected text box. \n- Lower values (e.g., 1.5) create tighter boxes. \n- Higher values (e.g., 3.0) capture more of the outline. \n- Too high can cause boxes to merge.\nDefault: 2.3", 
                                 row_idx_ref=row_idx)        
        det_checks_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        det_checks_frame.grid(row=row_idx['row'], column=0, columnspan=2, padx=5, pady=0, sticky="w"); row_idx['row'] += 1
        self._create_setting_row(det_checks_frame, 'det_rotate', "Detect Rotated Text", "checkbox", row_idx_ref={'row':0})
        self._create_setting_row(det_checks_frame, 'det_auto_rotate', "Auto-Rotate Image", "checkbox", row_idx_ref={'row':1})
        self._create_setting_row(det_checks_frame, 'det_invert', "Invert Image Colors", "checkbox", row_idx_ref={'row':2})
        self._create_setting_row(det_checks_frame, 'det_gamma_correct', "Gamma Correction", "checkbox", row_idx_ref={'row':3})
        self._create_setting_row(scroll_frame, 'min_text_length', "Min. Text Length:", "entry", special_props={"placeholder_text": "e.g. 2"}, tooltip="Ignore any recognized text shorter than this length.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'prob', "OCR Probability Threshold:", "entry", special_props={"placeholder_text": "Blank = auto"}, tooltip="Minimum confidence for an OCR result to be accepted. (e.g. 0.85)", row_idx_ref=row_idx)

    def _create_image_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(1, weight=1)
        row_idx = {'row': 0}

        ctk.CTkLabel(scroll_frame, text="Inpainter (Text Remover)", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(10,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'inpainter', "Inpainter Model:", "option", values=MODEL_OPTIONS['inpainter'], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'inpainting_precision', "Inpainting Precision:", "option", values=MODEL_OPTIONS['inpainting_precision'], tooltip="Calculation precision. 'fp32' is safe. 'fp16'/'bf16' are faster but require modern GPU support.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'inpainting_size', "Inpainting Size:", "slider", special_props={"from_": 512, "to": 4096, "number_of_steps": 56}, tooltip="The resolution for the inpainting process.\nHigher values can improve quality but are much slower and use more memory.\nDefault: 2048", row_idx_ref=row_idx)
        
        ctk.CTkLabel(scroll_frame, text="Upscaler", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(15,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'upscaler', "Upscaler Model:", "option", values=MODEL_OPTIONS['upscaler'], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'upscale_ratio', "Upscale Ratio:", "segmented", values=["Disabled", "2x", "3x", "4x"], tooltip="How many times to multiply the image resolution.", command=lambda v: self._on_setting_change('upscale_ratio', v.replace('x', '') if 'x' in v else "Disabled"), row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'revert_upscaling', "Revert Upscaling:", "checkbox", tooltip="If checked, the image is upscaled for better processing, then scaled back down to its original size.", row_idx_ref=row_idx)

        ctk.CTkLabel(scroll_frame, text="Colorizer", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(15,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'colorizer', "Colorizer Model:", "option", values=MODEL_OPTIONS['colorizer'], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'colorization_size', "Colorization Size:", "slider", special_props={"from_": 256, "to": 2048, "number_of_steps": 16}, tooltip="The resolution for the colorization process.\nHigher values improve quality but are slower.\nDefault: 576", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'denoise_sigma', "Denoise Sigma:", "slider", special_props={"from_": 0, "to": 100, "number_of_steps": 100}, tooltip="Strength of the denoising filter applied during colorization.\nHigher values remove more noise but can soften details.", row_idx_ref=row_idx)

    def _create_render_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(1, weight=1)
        row_idx = {'row': 0}

        ctk.CTkLabel(scroll_frame, text="Font & Text Style", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(10,0), sticky="w"); row_idx['row'] += 1
        
        project_fonts = self._get_project_fonts()
        font_list = []
        if project_fonts:
            font_list.extend(["--- Project Fonts ---"])
            font_list.extend(project_fonts)
        font_list.extend(["--- System Fonts ---"])
        font_list.extend(self.system_fonts)
        
        self._create_setting_row(scroll_frame, 'gimp_font', "Font:", "combo", values=font_list, 
                                 command=lambda v: self._on_setting_change('gimp_font', v) if not v.startswith("---") else None,
                                 tooltip="Select a font from the project's 'fonts' folder or your system.", 
                                 row_idx_ref=row_idx)
        
        font_color_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        font_color_frame.grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        font_color_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(font_color_frame, text="Font Color:").grid(row=0, column=0, sticky="w")
        color_entry = ctk.CTkEntry(font_color_frame, placeholder_text="000000")
        color_entry.grid(row=0, column=1, padx=(10,5), sticky="ew")
        color_entry.bind("<KeyRelease>", lambda e: self._on_setting_change('font_color', color_entry.get()))
        color_button = ctk.CTkButton(font_color_frame, text="...", width=40, command=self._choose_font_color)
        color_button.grid(row=0, column=2, padx=(0,0), sticky="e")
        self.setting_widgets['font_color'] = color_entry
        self.setting_widgets['color_picker_button'] = color_button
        row_idx['row'] += 1
        
        self._create_setting_row(scroll_frame, 'alignment', "Text Alignment:", "segmented", values=["auto", "left", "center", "right"], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'font_size_offset', "Font Size Offset:", "slider", special_props={"from_": -20, "to": 20, "number_of_steps": 40}, tooltip="Fine-tune the automatically calculated font size.\nPositive values make text bigger, negative values make it smaller.", row_idx_ref=row_idx)

        ctk.CTkLabel(scroll_frame, text="Output & Filtering", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(15,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'auto_rename', "Auto-rename output folder", "checkbox", tooltip="Automatically renames the output folder with a language suffix (e.g., 'manga-ENG').", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'overwrite_output', "Overwrite existing output folder", "checkbox", tooltip="If checked, any existing output folder with the same name will be deleted before processing.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'backup_original', "Create backup of original folder", "checkbox", tooltip="Saves a copy of the original, untouched folder before processing.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'filter_text', "Filter Text (comma-sep):", "entry", special_props={"placeholder_text": "e.g., (SFX),[NOISE]"}, tooltip="Any text matching these patterns will be ignored by the OCR.\nUseful for sound effects or watermarks.", row_idx_ref=row_idx)

        ctk.CTkLabel(scroll_frame, text="Advanced Render & Inpainter Settings", font=ctk.CTkFont(weight="bold")).grid(row=row_idx['row'], column=0, columnspan=2, padx=10, pady=(15,0), sticky="w"); row_idx['row'] += 1
        self._create_setting_row(scroll_frame, 'font_size', "Font Size (Override):", "slider", special_props={"from_": 0, "to": 150, "number_of_steps": 151}, command=lambda v: self._on_setting_change('font_size', int(v) if v > 0 else None), tooltip="Manually set a fixed font size. '0' means automatic sizing.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'font_size_minimum', "Min. Font Size:", "slider", special_props={"from_": -1, "to": 50, "number_of_steps": 52}, command=lambda v: self._on_setting_change('font_size_minimum', int(v)), tooltip="The smallest font size allowed. Prevents text from becoming unreadably small. '-1' disables this.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'line_spacing', "Line Spacing:", "slider", special_props={"from_": 0, "to": 30, "number_of_steps": 31}, command=lambda v: self._on_setting_change('line_spacing', int(v) if v > 0 else None), tooltip="Manually set the space between lines of text. '0' means automatic.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'direction', "Text Direction:", "option", values=MODEL_OPTIONS['direction'], row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'mask_dilation_offset', "Mask Dilation Offset:", "entry", special_props={"placeholder_text": "e.g., 10"}, tooltip="Expands or shrinks the mask around the text before inpainting.", row_idx_ref=row_idx)
        self._create_setting_row(scroll_frame, 'kernel_size', "Kernel Size:", "entry", special_props={"placeholder_text": "e.g., 3"}, tooltip="Size of the kernel used in mask processing. Must be an odd number.", row_idx_ref=row_idx)
        font_checks_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        font_checks_frame.grid(row=row_idx['row'], column=0, columnspan=2, padx=5, pady=0, sticky="w"); row_idx['row'] += 1
        self._create_setting_row(font_checks_frame, 'disable_font_border', "Disable Font Border", "checkbox", row_idx_ref={'row':0})
        self._create_setting_row(font_checks_frame, 'uppercase', "Force Uppercase", "checkbox", row_idx_ref={'row':1})
        self._create_setting_row(font_checks_frame, 'lowercase', "Force Lowercase", "checkbox", row_idx_ref={'row':2})
        self._create_setting_row(font_checks_frame, 'no_hyphenation', "No Hyphenation", "checkbox", row_idx_ref={'row':3})
        self._create_setting_row(font_checks_frame, 'rtl', "Right-to-Left (RTL)", "checkbox", tooltip="Enable for languages like Arabic. Disable for most others.", row_idx_ref={'row':4})

    def _create_visual_compare_tab(self, tab):
        """Creates a stable UI for Visual Compare with a fixed layout that does not flash."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # --- Top Controls Frame ---
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        controls_frame.grid_columnconfigure(3, weight=1)

        load_button = ctk.CTkButton(controls_frame, text="Load Test Image...", command=self._load_test_image)
        load_button.pack(side="left", padx=10, pady=10)
        
        self.fast_preview_check = ctk.CTkCheckBox(controls_frame, text="Fast Preview")
        self.fast_preview_check.pack(side="left", padx=10, pady=10)
        self.fast_preview_check.select()
        
        self.run_test_button = ctk.CTkButton(controls_frame, text="Run Test", state="disabled", command=self._run_visual_test)
        self.run_test_button.pack(side="right", padx=10, pady=10)
        
        reset_button = ctk.CTkButton(controls_frame, text="Reset View", command=self._reset_canvas_view)
        reset_button.pack(side="right", padx=10, pady=10)

        self.zoom_label = ctk.CTkLabel(controls_frame, text="Zoom: 100%")
        self.zoom_label.pack(side="right", padx=10, pady=10)

        # --- Informational Text Label ---
        info_text = ("This tab is for quick visual tests on a single image. "
                    "Load an image to see how your current settings will affect the output before running a full job.\n"
                    "Note: High zoom levels on large images may impact performance. This is normal behavior.")
        info_label = ctk.CTkLabel(tab, text=info_text, wraplength=500, justify="left", text_color="gray60")
        info_label.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")

        # --- Image Display Area (This frame will hold the canvases) ---
        self.image_area_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.image_area_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Initialize canvas variables to None. They will be created by _redraw_images.
        self.original_canvas = None
        self.translated_canvas = None
        self.original_label = None
        self.output_label = None
        self._current_layout = None  # Layout takibi için

    def _create_extra_settings_tab(self, tab):
        """Creates the UI for presets, appearance, and other extra settings."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(0, weight=1)
        
        # --- 1. Preset Manager ---
        preset_frame = ctk.CTkFrame(scroll_frame)
        preset_frame.pack(fill="x", padx=10, pady=10, anchor="n")
        preset_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(preset_frame, text="Preset Manager", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(5,10), sticky="w")
        
        self.profile_combobox = ctk.CTkComboBox(preset_frame, values=[], command=lambda choice: self.profile_name_entry.delete(0, 'end') or self.profile_name_entry.insert(0, choice))
        self.profile_combobox.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.profile_name_entry = ctk.CTkEntry(preset_frame, placeholder_text="Enter new preset name")
        self.profile_name_entry.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        button_frame = ctk.CTkFrame(preset_frame, fg_color="transparent")
        button_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        button_frame.grid_columnconfigure((0,1,2), weight=1)

        ctk.CTkButton(button_frame, text="Save", command=self._save_profile).grid(row=0, column=0, padx=2, sticky="ew")
        ctk.CTkButton(button_frame, text="Load", command=self._load_profile).grid(row=0, column=1, padx=2, sticky="ew")
        ctk.CTkButton(button_frame, text="Delete", command=self._delete_profile, fg_color="#D32F2F", hover_color="#B71C1C").grid(row=0, column=2, padx=2, sticky="ew")

        # --- 2. Advanced Debug Checkbox (Moved here) ---
        debug_frame = ctk.CTkFrame(scroll_frame)
        debug_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        debug_checkbox = ctk.CTkCheckBox(debug_frame, text="Enable Advanced Debugging",
                                         variable=self.enable_advanced_debug)
        debug_checkbox.pack(side="left", padx=10, pady=5)

        debug_tooltip_text = ("WARNING: This is for advanced users or for troubleshooting specific issues.\n\n"
                              "- When enabled, the tool saves all intermediate processing steps (masks, OCR boxes, etc.) "
                              "into a 'debug_output' subfolder within each job's final output directory.\n"
                              "- This will consume a VERY LARGE amount of disk space (e.g., >20MB for a single image) "
                              "and will significantly slow down the entire process.\n"
                              "- A new process will overwrite the previous debug files.\n\n"
                              "It is STRONGLY recommended to keep this OFF for normal use.")
        
        tooltip_icon = ctk.CTkLabel(debug_frame, text=" (?)", text_color="orange red", cursor="hand2", font=ctk.CTkFont(weight="bold"))
        tooltip_icon.pack(side="left")
        CTkToolTip(tooltip_icon, message=debug_tooltip_text, delay=0.1)
        
        # --- 3. Special Tasks ---
        main_tasks_frame = ctk.CTkFrame(scroll_frame)
        main_tasks_frame.pack(fill="x", padx=10, pady=10, anchor="n")
        main_tasks_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(main_tasks_frame, text="Special Tasks Workshop", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(5,10))

        # --- Common Setting: Processing Device ---
        device_frame = ctk.CTkFrame(main_tasks_frame, fg_color="transparent")
        device_frame.pack(fill="x", padx=10, pady=5)
        device_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(device_frame, text="Processing Device:").grid(row=0, column=0, sticky="w", padx=(0,10))
        device_selector = ctk.CTkSegmentedButton(device_frame, 
                                                 values=["CPU", "NVIDIA GPU"],
                                                 command=lambda v: self._on_special_task_setting_change("processing_device", v))
        device_selector.set(self.special_task_settings["processing_device"])
        device_selector.grid(row=0, column=1, sticky="ew")
        self.special_task_widgets['processing_device'] = device_selector
        
        self._create_raw_task_module(main_tasks_frame)
        self._create_upscale_task_module(main_tasks_frame)
        self._create_colorize_task_module(main_tasks_frame)
        
        api_placeholder_frame = ctk.CTkFrame(scroll_frame)
        api_placeholder_frame.pack(fill="x", padx=10, pady=10, anchor="n")
        
        ctk.CTkLabel(api_placeholder_frame, text="API Key Management (Future Implementation)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5,5), padx=10, anchor="w")
        
        api_note = ("This section is reserved for managing API keys for services like DeepL, OpenAI, etc. "
                    "This feature will be implemented in a future update.")
        ctk.CTkLabel(api_placeholder_frame, text=api_note, wraplength=450, justify="left", text_color="gray60").pack(pady=(0,10), padx=10, anchor="w")

    def _create_raw_task_module(self, parent_frame):
        """Builds the UI module for the 'RAW Output' special task, including tooltips and a reset button."""
        raw_frame = ctk.CTkFrame(parent_frame)
        raw_frame.pack(fill="x", expand=True, padx=10, pady=(10,10))
        raw_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(raw_frame, text="Task: RAW Output (Clean Text)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=3, pady=(5,10), padx=10, sticky="w")
        
        # --- Local helper function to create a setting row with a slider and a tooltip ---
        def create_special_slider_row(row, text, key, from_, to, steps, tooltip_text, number_format="{:.2f}"):
            # --- Label and Tooltip ---
            label_frame = ctk.CTkFrame(raw_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkLabel(label_frame, text=text).pack(side="left")

            default_val = self.special_task_settings['raw'][key]
            full_tooltip = f"{tooltip_text}\nDefault: {default_val}"
            
            tooltip_icon = ctk.CTkLabel(label_frame, text=" (?)", text_color="cyan", cursor="hand2")
            tooltip_icon.pack(side="left", padx=2)
            CTkToolTip(tooltip_icon, message=full_tooltip, delay=0.1)

            # --- Slider and Value Label ---
            slider_frame = ctk.CTkFrame(raw_frame, fg_color="transparent")
            slider_frame.grid(row=row, column=1, sticky="ew", padx=10)
            slider_frame.grid_columnconfigure(0, weight=1)
            
            value_label = ctk.CTkLabel(slider_frame, width=45, anchor="e")
            value_label.grid(row=0, column=1, padx=(10,0))
            
            slider = ctk.CTkSlider(slider_frame, from_=from_, to=to, number_of_steps=steps, 
                                   command=lambda v, k=key, lbl=value_label, fmt=number_format: (lbl.configure(text=fmt.format(v)), self._on_special_task_setting_change(k, v, "raw")))
            slider.grid(row=0, column=0, sticky="ew")
            
            initial_value = self.special_task_settings['raw'][key]
            slider.set(initial_value)
            value_label.configure(text=number_format.format(initial_value))

            self.special_task_widgets[f"raw_{key}"] = (slider, value_label)

        # --- Create the UI elements using the helper function ---
        create_special_slider_row(1, "Detection Size:", "detection_size", 512, 4096, (4096-512)//64, 
                                  "Image resolution for text detection.\nHigher: Finds smaller text, but is slower.\nLower: Faster, but may miss small text.", number_format="{:.0f}")
        create_special_slider_row(2, "Text Threshold:", "text_threshold", 0.1, 0.9, 80, 
                                  "How confident the model must be to mark an area as text.\nHigher: Finds only very clear text.\nLower: Finds more text, but may include noise.")
        create_special_slider_row(3, "Box Threshold:", "box_threshold", 0.1, 0.9, 80, 
                                  "How confident the model must be to finalize the text box.\nHigher: Tighter, more precise boxes.\nLower: Looser, more forgiving boxes.")
        create_special_slider_row(4, "Unclip Ratio:", "unclip_ratio", 1.0, 5.0, 80, 
                                  "How much to expand the detected text box to capture outlines.\nHigher: Expands more.\nLower: Tighter boxes.")
        create_special_slider_row(5, "Ignore Bubbles < (px):", "ignore_bubble", 0, 500, 50, 
                                  "Ignores any detected text bubbles smaller than this pixel area.\nUseful for filtering out noise and small sfx.", number_format="{:.0f}")
        ctk.CTkLabel(raw_frame, text="Inpainter Model:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        inpainter_menu = ctk.CTkOptionMenu(raw_frame, 
                                           values=["lama_large", "lama_mpe","sd","original"],
                                           command=lambda v: self._on_special_task_setting_change("inpainter_model", v, "raw"))
        inpainter_menu.set(self.special_task_settings['raw']['inpainter_model'])
        inpainter_menu.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
        self.special_task_widgets['raw_inpainter_model'] = inpainter_menu

        create_special_slider_row(7, "Inpainting Size:", "inpainting_size", 512, 2048, (4096-512)//64,
                                  "Resolution for the inpainting process.\nHigher: Better quality, but much slower.\nLower: Faster, but may be less precise.", number_format="{:.0f}")
        
        # --- Bottom section with Reset and Generate Buttons ---
        button_frame = ctk.CTkFrame(raw_frame, fg_color="transparent")
        button_frame.grid(row=9, column=1, columnspan=2, sticky="sew", pady=(10,5), padx=10)
        button_frame.grid_columnconfigure((0, 1), weight=1) # Let columns expand
        
        ctk.CTkButton(button_frame, text="Reset", command=self._reset_raw_module_settings, fg_color="gray50", hover_color="gray40").grid(row=0, column=0, sticky="w", padx=2)
        ctk.CTkButton(button_frame, text="Generate RAW Output...", command=lambda: self._start_special_task_pipeline('raw')).grid(row=0, column=1, sticky="e", padx=2)

    def _reset_raw_module_settings(self):
        """Resets all controls in the RAW task module to their default values by reading from the defaults constant."""
        self.log("INFO", "Resetting RAW Output module settings to defaults...")
        
        # A small helper dictionary to format the labels correctly.
        format_map = {
            "detection_size": "{:.0f}",
            "text_threshold": "{:.2f}",
            "box_threshold": "{:.2f}",
            "unclip_ratio": "{:.2f}",
            "ignore_bubble": "{:.0f}"
        }

        # Iterate through the keys defined in the DEFAULTS constant for the 'raw' module.
        for key in SPECIAL_TASK_DEFAULTS['raw']:
            # This part handles sliders
            if key in format_map:
                default_value = SPECIAL_TASK_DEFAULTS['raw'][key]
                self.special_task_settings['raw'][key] = default_value
                slider, label = self.special_task_widgets[f"raw_{key}"]
                slider.set(default_value)
                label.configure(text=format_map[key].format(default_value))
                
        default_model = SPECIAL_TASK_DEFAULTS['raw']['inpainter_model']
        self.special_task_settings['raw']['inpainter_model'] = default_model
        self.special_task_widgets['raw_inpainter_model'].set(default_model)
        
        self.log("SUCCESS", "RAW Output module has been reset.")
    
    def _create_upscale_task_module(self, parent_frame):
        """Builds the UI module for the 'Upscale' special task."""
        upscale_frame = ctk.CTkFrame(parent_frame)
        upscale_frame.pack(fill="x", expand=True, padx=10, pady=(10,10))
        upscale_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(upscale_frame, text="Task: Upscale Image", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=3, pady=(5,10), padx=10, sticky="w")
        
        # --- Upscaler Model ---
        label_frame = ctk.CTkFrame(upscale_frame, fg_color="transparent")
        label_frame.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkLabel(label_frame, text="Upscaler Model:").pack(side="left")
        tooltip_icon = ctk.CTkLabel(label_frame, text=" (?)", text_color="cyan", cursor="hand2")
        tooltip_icon.pack(side="left", padx=2)
        CTkToolTip(tooltip_icon, message="The AI model used for upscaling.\n'esrgan' and '4xultrasharp' are high quality, 'waifu2x' is faster.", delay=0.1)
        model_menu = ctk.CTkOptionMenu(upscale_frame, 
                                       values=["esrgan", "waifu2x", "4xultrasharp"],
                                       command=lambda v: self._on_special_task_setting_change("model", v, "upscale"))
        model_menu.set(self.special_task_settings['upscale']['model'])
        model_menu.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        self.special_task_widgets['upscale_model'] = model_menu

        # --- Upscale Ratio Slider ---
        ctk.CTkLabel(upscale_frame, text="Upscale Ratio:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        
        slider_frame = ctk.CTkFrame(upscale_frame, fg_color="transparent")
        slider_frame.grid(row=2, column=1, sticky="ew", padx=10)
        slider_frame.grid_columnconfigure(0, weight=1)
        
        value_label = ctk.CTkLabel(slider_frame, width=45, anchor="e")
        value_label.grid(row=0, column=1, padx=(10,0))

        # This slider logic is complex, so we handle it directly
        ratio_slider = ctk.CTkSlider(slider_frame, from_=1, to=16, number_of_steps=15,
                                 command=lambda v: (value_label.configure(text=f"{int(v)}x"), self._on_special_task_setting_change("ratio", int(v), "upscale")))
        initial_ratio = self.special_task_settings['upscale']['ratio']
        ratio_slider.set(initial_ratio)
        value_label.configure(text=f"{initial_ratio:.2f}x")
        self.special_task_widgets['upscale_ratio'] = (ratio_slider, value_label)
        ratio_slider.grid(row=0, column=0, sticky="ew")

        # --- Revert Size Checkbox ---
        revert_checkbox = ctk.CTkCheckBox(upscale_frame, text="Revert to original size after processing",
                                          command=lambda: self._on_special_task_setting_change("revert_size", revert_checkbox.get(), "upscale"))
        if self.special_task_settings['upscale']['revert_size']:
            revert_checkbox.select()
        revert_checkbox.grid(row=3, column=1, sticky="w", padx=10, pady=10)
        self.special_task_widgets['upscale_revert_size'] = revert_checkbox

        # --- Bottom section with Reset and Generate Buttons ---
        button_frame = ctk.CTkFrame(upscale_frame, fg_color="transparent")
        button_frame.grid(row=4, column=1, columnspan=2, sticky="sew", pady=(10,5), padx=10)
        button_frame.grid_columnconfigure((0, 1), weight=1)
        
        ctk.CTkButton(button_frame, text="Reset", command=self._reset_upscale_module_settings, fg_color="gray50", hover_color="gray40").grid(row=0, column=0, sticky="w", padx=2)
        ctk.CTkButton(button_frame, text="Generate Upscaled Image...", command=lambda: self._start_special_task_pipeline('upscale')).grid(row=0, column=1, sticky="e", padx=2)
    
    def _reset_upscale_module_settings(self):
        """Resets all controls in the Upscale task module to their default values."""
        self.log("INFO", "Resetting Upscale module settings to defaults...")

        # --- Reset Model ---
        default_model = SPECIAL_TASK_DEFAULTS['upscale']['model']
        self.special_task_settings['upscale']['model'] = default_model
        self.special_task_widgets['upscale_model'].set(default_model)

        # --- Reset Ratio Slider ---
        default_ratio = SPECIAL_TASK_DEFAULTS['upscale']['ratio']
        self.special_task_settings['upscale']['ratio'] = default_ratio
        slider, label = self.special_task_widgets['upscale_ratio']
        slider.set(default_ratio)
        label.configure(text=f"{int(default_ratio)}x")

        # --- Reset Revert Checkbox ---
        default_revert = SPECIAL_TASK_DEFAULTS['upscale']['revert_size']
        self.special_task_settings['upscale']['revert_size'] = default_revert
        checkbox = self.special_task_widgets['upscale_revert_size']
        if default_revert:
            checkbox.select()
        else:
            checkbox.deselect()

        self.log("SUCCESS", "Upscale module has been reset.")
        
    def _create_colorize_task_module(self, parent_frame):
        """Builds the UI module for the 'Colorize' special task with smart ratio and text detection controls."""
        s_settings = self.special_task_settings['colorize']
        
        colorize_frame = ctk.CTkFrame(parent_frame)
        colorize_frame.pack(fill="x", expand=True, padx=10, pady=(10,10))
        colorize_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(colorize_frame, text="Task: Colorize Image", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=2, pady=(5,10), padx=10, sticky="w")
        
        note_text = ("NOTE: Despite our best efforts to allow for precise decimal scaling (e.g., 2.56x), "
                     "the underlying upscaler models stubbornly refuse to accept anything but whole numbers. "
                     "Therefore, to make this work, the final calculated ratio will be rounded to the nearest integer (e.g., 3x) before processing. "
                     "The result will be very close to what you requested, but not exact. We tried.")
        ctk.CTkLabel(colorize_frame, text=note_text, wraplength=450, justify="left", font=ctk.CTkFont(slant="italic"), text_color="gray60").grid(row=1, column=0, columnspan=2, pady=(0, 10), padx=10, sticky="w")
        
        # --- Helper for creating sliders, NOW WITH TOOLTIP SUPPORT ---
        def create_slider(row, text, key, from_, to, steps, number_format="{:.0f}", tooltip_text=None):
            label_frame = ctk.CTkFrame(colorize_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkLabel(label_frame, text=text).pack(side="left")
            # If tooltip_text is provided, create the icon and tooltip
            if tooltip_text:
                tooltip_icon = ctk.CTkLabel(label_frame, text=" (?)", text_color="cyan", cursor="hand2")
                tooltip_icon.pack(side="left", padx=2)
                CTkToolTip(tooltip_icon, message=tooltip_text, delay=0.1)
            
            slider_frame = ctk.CTkFrame(colorize_frame, fg_color="transparent")
            slider_frame.grid(row=row, column=1, sticky="ew", padx=10)
            slider_frame.grid_columnconfigure(0, weight=1)
            value_label = ctk.CTkLabel(slider_frame, width=45, anchor="e")
            value_label.grid(row=0, column=1, padx=(10,0))
            slider = ctk.CTkSlider(slider_frame, from_=from_, to=to, number_of_steps=steps, 
                                   command=lambda v: (value_label.configure(text=number_format.format(v)), 
                                                      self._on_special_task_setting_change(key, v, "colorize")))
            
            slider.grid(row=0, column=0, sticky="ew")
            initial_value = s_settings[key]
            slider.set(initial_value)
            value_label.configure(text=number_format.format(initial_value))
            self.special_task_widgets[f"colorize_{key}"] = (slider, value_label)

        # --- Update slider creation calls to include the new 'tooltip_text' parameter ---
        ctk.CTkLabel(colorize_frame, text="--- Colorization Settings ---", text_color="gray60").grid(row=2, column=0, columnspan=2, pady=(5,0))
        create_slider(3, "Colorization Quality:", "size", 256, 1024, (1024-256)//16, 
                      tooltip_text="The detail level for colorization. Best results are often near the default 576.")
        create_slider(4, "Denoise Sigma:", "denoise", 0, 100, 100, 
                      tooltip_text="Strength of the noise reduction filter applied during colorization.")
        
        ctk.CTkLabel(colorize_frame, text="--- Text Detection (to prevent cleaning) ---", text_color="gray60").grid(row=5, column=0, columnspan=2, pady=(10,0))
        create_slider(6, "Ignore Bubbles < (px):", "ignore_bubble", 0, 500, 50, 
                      tooltip_text="Prevents cleaning of text bubbles smaller than this pixel area.")
        create_slider(7, "Text Threshold:", "text_threshold", 0.1, 0.9, 80, number_format="{:.2f}", 
                      tooltip_text="Set high (e.g., 0.9) to prevent cleaning anything that isn't clearly text.")

        # --- Create controls for Final Output (with manual tooltips) ---
        ctk.CTkLabel(colorize_frame, text="--- Final Output Size ---", text_color="gray60").grid(row=8, column=0, columnspan=2, pady=(10,0))
        
        # Upscaler Model (manual tooltip)
        label_frame_model = ctk.CTkFrame(colorize_frame, fg_color="transparent")
        label_frame_model.grid(row=9, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkLabel(label_frame_model, text="Upscaler Model:").pack(side="left")
        tooltip_icon_model = ctk.CTkLabel(label_frame_model, text=" (?)", text_color="cyan", cursor="hand2")
        tooltip_icon_model.pack(side="left", padx=2)
        CTkToolTip(tooltip_icon_model, message="The AI model used for the final upscale step.", delay=0.1)
        
        model_menu = ctk.CTkOptionMenu(colorize_frame,
                                       values=["esrgan", "waifu2x", "4xultrasharp"],
                                       command=lambda v: self._on_special_task_setting_change("upscaler_model", v, "colorize"))
        model_menu.grid(row=9, column=1, sticky="ew", padx=10, pady=5)
        self.special_task_widgets['colorize_upscaler_model'] = model_menu

        # Final Upscale Ratio (manual tooltip)
        label_frame_ratio = ctk.CTkFrame(colorize_frame, fg_color="transparent")
        label_frame_ratio.grid(row=10, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkLabel(label_frame_ratio, text="Final Upscale Ratio:").pack(side="left")
        tooltip_icon_ratio = ctk.CTkLabel(label_frame_ratio, text=" (?)", text_color="cyan", cursor="hand2")
        tooltip_icon_ratio.pack(side="left", padx=2)
        CTkToolTip(tooltip_icon_ratio, message="The final size of the output image relative to the original.\n1.0x = Match original size.", delay=0.1)
        
        slider_frame = ctk.CTkFrame(colorize_frame, fg_color="transparent")
        slider_frame.grid(row=10, column=1, sticky="ew", padx=10)
        slider_frame.grid_columnconfigure(0, weight=1)
        value_label = ctk.CTkLabel(slider_frame, width=45, anchor="e")
        value_label.grid(row=0, column=1, padx=(10,0))
        ratio_slider = ctk.CTkSlider(slider_frame, from_=1, to=16, command=lambda v: self._on_colorize_ratio_slider_change(v, value_label))
        
        ratio_slider.grid(row=0, column=0, sticky="ew")
        self.special_task_widgets['colorize_final_ratio'] = (ratio_slider, value_label)

        # Bottom section with Buttons
        button_frame = ctk.CTkFrame(colorize_frame, fg_color="transparent")
        button_frame.grid(row=11, column=1, sticky="sew", pady=(10,5), padx=10)
        button_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(button_frame, text="Reset", command=self._reset_colorize_module_settings, fg_color="gray50", hover_color="gray40").grid(row=0, column=0, sticky="w", padx=2)
        ctk.CTkButton(button_frame, text="Generate Colorized Image...", command=lambda: self._start_special_task_pipeline('colorize')).grid(row=0, column=1, sticky="e", padx=2)

        self._reset_colorize_module_settings(log_output=False)

    def _reset_colorize_module_settings(self, log_output=True):
        """Resets all controls in the Colorize task module to their default values."""
        if log_output:
            self.log("INFO", "Resetting Colorize module settings to defaults...")
        
        defaults = SPECIAL_TASK_DEFAULTS['colorize']
        s_settings = self.special_task_settings['colorize']
        widgets = self.special_task_widgets

        # Reset all simple key-value pairs
        for key, default_value in defaults.items():
            s_settings[key] = default_value
            widget_key = f"colorize_{key}"

            if widget_key in widgets:
                if key == "upscaler_model":
                    widgets[widget_key].set(default_value)
                elif key == "final_ratio":
                    slider, label = widgets[widget_key]
                    slider.set(default_value)
                    label.configure(text=f"{default_value:.2f}x")
                elif key == "text_threshold":
                    slider, label = widgets[widget_key]
                    slider.set(default_value)
                    label.configure(text=f"{default_value:.2f}")
                else: # Handles size, denoise, ignore_bubble
                    slider, label = widgets[widget_key]
                    slider.set(default_value)
                    label.configure(text=f"{default_value:.0f}")

        if log_output:
            self.log("SUCCESS", "Colorize module has been reset.")
    
    def _on_special_task_setting_change(self, key, value, category=None):
        """Updates the special_task_settings dictionary when a control is changed."""
        
        processed_value = value
        try:
            # Always try to process as a number first. This handles all slider inputs.
            # We round to 3 decimal places for sufficient precision.
            processed_value = round(float(value), 3)
        except (ValueError, TypeError):
            # If it fails, it's a string (like 'esrgan') or a boolean. Keep the original value.
            pass 
        
        if category:
            self.special_task_settings[category][key] = processed_value
        else:
            self.special_task_settings[key] = processed_value

    def _on_colorize_ratio_slider_change(self, raw_value, label_widget):
        """
        Custom handler specifically for the Colorize module's ratio slider
        to implement non-linear steps.
        """
        # 1.0 to 5.0: 0.05 increments
        if raw_value <= 5:
            final_value = round(raw_value, 2)
        # 5.0 to 10.0: 1.0 increments
        elif raw_value <= 10:
            final_value = round(raw_value)
        # 10.0 to 16.0: 1.0 increments
        else:
            final_value = round(raw_value)
            
        label_widget.configure(text=f"{final_value:.2f}x")
        # Update the correct key in the settings dictionary
        self._on_special_task_setting_change("final_ratio", final_value, "colorize")
        
    def _start_special_task_pipeline(self, task_type: str):
        """Prepares and starts the pipeline for a special task."""
        if self.is_running_pipeline:
            messagebox.showwarning("Process Busy [Code: UI-01]", "A process is already running. Please wait for it to finish.")
            return
        
        job = next((j for j in self.job_queue if j["id"] == self.selected_job_id), None)
        if not job:
            messagebox.showwarning("No Job Selected [Code: UI-02]", "Please select a job from the queue on the left to apply this task to.")
            return

        self._toggle_ui_state(True)
        task_thread = threading.Thread(target=self._run_special_task_pipeline, args=(task_type, job), daemon=True)
        task_thread.start()

    def _run_special_task_pipeline(self, task_type: str, job):
        """Runs a single job with the configuration from a special task module."""
        self.log("PIPELINE", f"Special task '{task_type}' started for: {os.path.basename(job['source_path'])}")
        
        # [PROGRESS BAR FIX] Update progress bar at the start of the task
        self.after(0, self._update_progress, (0.1, f"Preparing task: {task_type.capitalize()}..."))

        s_settings = self.special_task_settings
        config_data = {}
        task_succeeded = False # Flag to track if the process completes without errors

        if task_type == 'raw':
            config_data = {
              "detector": { "detector": "default", "detection_size": int(s_settings['raw']['detection_size']), "text_threshold": s_settings['raw']['text_threshold'], "box_threshold": s_settings['raw']['box_threshold'], "unclip_ratio": s_settings['raw']['unclip_ratio'] },
              "ocr": { "ocr": "mocr", "ignore_bubble": int(s_settings['raw']['ignore_bubble']) },
              "inpainter": { "inpainter": s_settings['raw']['inpainter_model'], "inpainting_size": int(s_settings['raw']['inpainting_size']) },
              "translator": { "translator": "none" }
            }
        
        elif task_type == 'upscale':
            # [UPSCALE CONFIG FIX] Explicitly disable other steps to prevent text detection
            config_data = {
              "detector": { "detector": "none" },
              "translator": { "translator": "none" },
              "upscale": {
                "upscaler": s_settings['upscale']['model'],
                "revert_upscaling": s_settings['upscale']['revert_size'],
                "upscale_ratio": int(s_settings['upscale']['ratio'])
              }
            }
        
        elif task_type == 'colorize':
            # =================================================================================
            # DEVELOPER NOTE: Advanced "Ultra Quality" Colorization Workflow
            #
            # The current implementation uses a "smart input size" method to work around the
            # underlying tool's limitation of only accepting integer upscale ratios.
            # A theoretically higher-quality (but more complex) approach would be:
            #
            # 1. First Pass (AI Upscale): Run the process with a fixed, high upscale ratio (e.g., 4x).
            #    This creates a large, high-detail, colorized intermediate file.
            #    >> config = {"colorizer": {...}, "upscale": {"upscale_ratio": 4}}
            #    >> run_command(config)
            #
            # 2. Second Pass (Resize): Use the Pillow library to open the large intermediate file
            #    and resize it down to the user's final desired size (e.g., original_width * 2.0x).
            #    This two-step process (AI upscale + standard resize) often yields the best quality.
            #
            # To implement this, this function would need to be made asynchronous or have a
            # callback system to manage the two separate processes. The current single-pass
            # method is a reliable and simpler alternative.
            # =================================================================================

            # --- Current Implementation: Smart Ratio Calculation with Rounding ---
            final_ratio_to_apply = 1.0
            try:
                first_image_path = next((os.path.join(job['source_path'], f) for f in os.listdir(job['source_path']) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))), None)
                if not first_image_path: raise ValueError("No image found in source directory")

                with Image.open(first_image_path) as img:
                    original_width, _ = img.size
                                
                required_ratio_to_match_original = original_width / s_settings['colorize']['size']
                ideal_ratio = required_ratio_to_match_original * s_settings['colorize']['final_ratio']
                
                # --- THIS IS THE KEY CHANGE: Round to the nearest integer ---
                final_ratio_to_apply = round(ideal_ratio)
                
                # Ensure the ratio is at least 1, as 0 is not a valid upscale ratio
                if final_ratio_to_apply < 1:
                    final_ratio_to_apply = 1
                
                self.log("INFO", f"Ideal ratio calculated: {ideal_ratio:.3f}x. Rounded to nearest integer: {final_ratio_to_apply}x")

            except Exception as e:
                self.log("ERROR", f"Could not calculate upscale ratio: {e}. Defaulting to 1x.")
                final_ratio_to_apply = 1

            config_data = {
              "detector": {
                  "detector": "default",
                  "text_threshold": s_settings['colorize']['text_threshold']
              },
              "ocr": {
                  "ignore_bubble": int(s_settings['colorize']['ignore_bubble'])
              },
              "inpainter": {
                  "inpainter": "none" # Crucial to prevent actual cleaning, we only use detection results.
              },
              "colorizer": {
                "colorizer": s_settings['colorize']['model'],
                "colorization_size": int(s_settings['colorize']['size']),
                "denoise_sigma": int(s_settings['colorize']['denoise'])
              },
              "upscale": {
                "upscaler": s_settings['colorize']['upscaler_model'],
                "upscale_ratio": round(final_ratio_to_apply, 3)
              },
              "translator": { "translator": "none" }
            }

        # --- [ERROR HANDLING FIX] The rest of the function with the success flag logic ---
        config_path = ""
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            config_path = os.path.join(TEMP_DIR, f"temp_config_{job['id']}_special.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
                        
            output_path = f"{job['source_path']}_{task_type.upper()}"
            command = [ self.python_executable, "-m", "manga_translator", "local", "-i", job['source_path'], "-o", output_path, "--config-file", config_path ]
            if self.enable_advanced_debug.get(): command.append("-v")
            if s_settings['processing_device'] == 'NVIDIA GPU': command.append("--use-gpu")

            self.pipeline_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            for line in iter(self.pipeline_process.stdout.readline, ''):
                if line.strip(): self.log("RAW", line.strip())
            
            return_code = self.pipeline_process.wait()

            if return_code == 0:
                self.log("SUCCESS", f"Special task '{task_type}' finished successfully.")
                task_succeeded = True
            else:
                self.log("ERROR", f"Special task '{task_type}' failed. See log for details.")
        
        except Exception as e:
            self.log("ERROR", f"A critical Python error occurred during the special task: {e}")
            traceback.print_exc()

        finally:
            self.pipeline_process = None
            if os.path.exists(config_path):
                try: os.remove(config_path)
                except Exception: pass
            
            self.log("PIPELINE", "Task ended.")
            
            final_status = "Ready" if task_succeeded else "Finished with Errors"
            self.after(0, self._update_progress, (1.0, final_status))
            
            self.after(0, self._toggle_ui_state, False)
        
    def _setup_profiles(self):
        """Ensures the profiles directory exists and loads the list."""
        os.makedirs(PROFILES_DIR, exist_ok=True)
        self.after(100, self._refresh_profile_list)

    def _refresh_profile_list(self):
        """Scans the profiles directory and updates the combobox."""
        try:
            profiles = [f.replace(".json", "") for f in os.listdir(PROFILES_DIR) if f.endswith(".json")]
            sorted_profiles = sorted(profiles)
            self.profile_combobox.configure(values=sorted_profiles if profiles else ["No profiles found"])
            if not profiles:
                self.profile_combobox.set("No profiles found")
            else:
                self.profile_combobox.set(sorted_profiles[0])
        except Exception as e:
            self.log("ERROR", f"Could not refresh profile list: {e}")

    def _save_profile(self):
        """Saves the current default settings as a new profile."""
        profile_name = self.profile_name_entry.get().strip()
        if not profile_name or profile_name == "No profiles found":
            messagebox.showwarning("Warning", "Please enter a valid name for the profile.")
            return

        file_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
        if os.path.exists(file_path):
            if not messagebox.askyesno("Confirm Overwrite", f"Profile '{profile_name}' already exists. Overwrite it?"):
                return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.default_job_settings, f, indent=4)
            self.log("SUCCESS", f"Profile '{profile_name}' saved successfully.")
        except Exception as e:
            self.log("ERROR", f"Failed to save profile: {e}")
        finally:
            self._refresh_profile_list()
            self.profile_combobox.set(profile_name)

    def _load_profile(self):
        """Loads a profile and applies its settings."""
        profile_name = self.profile_combobox.get()
        if not profile_name or profile_name == "No profiles found":
            messagebox.showwarning("Warning", "No profile selected to load.")
            return

        file_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
        if not os.path.exists(file_path):
            self.log("ERROR", f"Profile '{profile_name}' not found.")
            self._refresh_profile_list()
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
            
            self.default_job_settings.update(loaded_settings)
            
            job = next((j for j in self.job_queue if j.get("id") == self.selected_job_id), None)
            if job:
                job['settings'].update(loaded_settings)
            
            self._populate_settings_panel()
            self.log("SUCCESS", f"Profile '{profile_name}' loaded.")

        except Exception as e:
            self.log("ERROR", f"Failed to load profile '{profile_name}': {e}")

    def _delete_profile(self):
        """Deletes the selected profile from the disk."""
        profile_name = self.profile_combobox.get()
        if not profile_name or profile_name == "No profiles found":
            messagebox.showwarning("Warning", "No profile selected to delete.")
            return

        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to permanently delete the profile '{profile_name}'?"):
            return
            
        file_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.log("SUCCESS", f"Profile '{profile_name}' deleted.")
            except Exception as e:
                self.log("ERROR", f"Failed to delete profile: {e}")
        else:
            self.log("WARNING", f"Profile '{profile_name}' was not found on disk.")
        
        self.profile_name_entry.delete(0, 'end')
        self._refresh_profile_list()

    def _change_color_theme(self, new_theme: str):
        """Changes the application's color theme."""
        ctk.set_default_color_theme(new_theme)
        self._app_settings['color_theme'] = new_theme
        self.log("INFO", f"Color theme changed to {new_theme}.")
               
    def _create_model_manager_tab(self, tab):
        """Creates the UI for the Model Manager tab using a consistent grid layout."""
        # Configure the grid layout for the parent tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)  # Allow the scrollable frame (at row 1) to expand

        # Top descriptive label
        description_text = ("Manage common models here. Missing models will be downloaded automatically when a job starts.\n\n"
                            "Note: For a full list of all supported models, please see the original project's documentation.")
        
        ctk.CTkLabel(tab, text=description_text,
                     wraplength=tab.winfo_reqwidth() - 40, justify="left").grid(
            row=0, column=0, padx=10, pady=10, sticky="ew")

        # Create a scrollable frame to list all the models
        self.model_list_frame = ctk.CTkScrollableFrame(tab, label_text="Available Models")
        self.model_list_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.model_list_frame.grid_columnconfigure(0, weight=1)

        # Add a refresh button at the bottom
        refresh_button = ctk.CTkButton(tab, text="🔄 Refresh Status", command=self.refresh_model_manager_ui)
        refresh_button.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="e")

        # Initial population of the list
        self.after(200, self.refresh_model_manager_ui)

    def _create_log_tab(self, tab):
        """Creates the content for the 'Live Log' tab."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(header, text="Clear Log", width=100, command=self._clear_log).pack(side="right")
        self.log_textbox = ctk.CTkTextbox(tab, state="disabled", wrap="word", font=("Consolas", 12))
        self.log_textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def _create_setting_row(self, parent, key, label, widget_type, values=None, tooltip=None, command=None, special_props=None, row_idx_ref=None):
        """Helper function to create a labeled setting widget row, with robust support for all slider types."""
        if special_props is None: special_props = {}
        
        row_idx = row_idx_ref['row']
        label_frame = ctk.CTkFrame(parent, fg_color="transparent")
        label_frame.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(label_frame, text=label).pack(side="left")

        if tooltip:
            tooltip_icon = ctk.CTkLabel(label_frame, text=" (?)", text_color="cyan", cursor="hand2")
            tooltip_icon.pack(side="left")
            CTkToolTip(tooltip_icon, message=tooltip)

        widget = None
        final_command = command if command else lambda v, k=key: self._on_setting_change(k, v)

        if widget_type in ["option", "combo", "entry", "checkbox", "segmented"]:
            # This part for non-slider widgets is correct and remains unchanged.
            if widget_type == "option": widget = ctk.CTkOptionMenu(parent, values=values, command=final_command, **special_props)
            elif widget_type == "combo": widget = ctk.CTkComboBox(parent, values=values, command=final_command, **special_props); widget.bind("<KeyRelease>", lambda e, k=key: self._on_setting_change(k, e.widget.get()))
            elif widget_type == "entry": widget = ctk.CTkEntry(parent, **special_props); widget.bind("<KeyRelease>", lambda e, k=key: self._on_setting_change(k, e.widget.get()))
            elif widget_type == "checkbox": widget = ctk.CTkCheckBox(parent, text="", command=lambda k=key: self._on_setting_change(k, self.setting_widgets[k].get()), **special_props)
            elif widget_type == "segmented": widget = ctk.CTkSegmentedButton(parent, values=values, command=final_command, **special_props)
            
            if widget:
                widget.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
                self.setting_widgets[key] = widget
        
        elif widget_type == "slider":
            slider_frame = ctk.CTkFrame(parent, fg_color="transparent")
            slider_frame.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
            slider_frame.grid_columnconfigure(0, weight=1)

            value_label = ctk.CTkLabel(slider_frame, text="", width=45, anchor="e")
            value_label.grid(row=0, column=1, padx=(10, 0))

            def update_command(slider_value):
                # This inner function now correctly handles the display text for all slider types
                display_text = ""
                # The 'command' passed to the row creator handles the actual value change
                final_command(slider_value)
                
                # This block only handles what the user SEES
                if key == 'unclip_ratio':
                    display_text = f"{float(slider_value)/10.0:.1f}"
                elif key in ['text_threshold', 'box_threshold']:
                    display_text = f"{float(slider_value)/100.0:.2f}"
                elif key == 'font_size' and int(slider_value) == 0:
                    display_text = "auto"
                elif key == 'line_spacing' and int(slider_value) == 0:
                    display_text = "auto"
                elif key == 'font_size_minimum' and int(slider_value) == -1:
                    display_text = "disabled"
                else: # For all other standard integer sliders
                    display_text = f"{int(slider_value)}"
                
                value_label.configure(text=display_text)

            widget = ctk.CTkSlider(slider_frame, command=update_command, **special_props)
            widget.grid(row=0, column=0, sticky="ew")
            
            self.setting_widgets[key] = (widget, value_label)
        
        row_idx_ref['row'] += 1
        return widget

    # 4. Core Application Logic

    def _add_job(self):
        """Adds a new job via a file dialog."""
        folder_path = filedialog.askdirectory(title="Select Manga/Image Folder")
        if folder_path:
            self._add_job_from_path(folder_path)

    def _add_job_from_path(self, path):
        """Creates a new job from a given path and adds it to the queue."""
        job_id = f"job_{int(time.time() * 1000)}_{len(self.job_queue)}"
        job_data = {
            "id": job_id,
            "source_path": path,
            "settings": self.default_job_settings.copy()
        }
        self.job_queue.append(job_data)
        self._update_job_list_display()
        self.log("INFO", f"Job added: {os.path.basename(path)}")
        if len(self.job_queue) == 1:
            self._select_job(job_id)

    def _remove_selected_job(self):
        """Removes the currently selected job from the queue."""
        if not self.selected_job_id:
            messagebox.showwarning("Warning", "You must select a job to remove.")
            return
        
        self.job_queue = [j for j in self.job_queue if j['id'] != self.selected_job_id]
        self.selected_job_id = None
        self._update_job_list_display()

        if self.job_queue:
            # Hala iş varsa, ilkini seç (bu _select_job UI'ı etkinleştirir).
            self._select_job(self.job_queue[0]['id'])
        else:
            # Hiç iş kalmadıysa, varsayılanları göster ve UI'ı devre dışı bırak.
            self._populate_settings_panel()

    def _clear_all_jobs(self):
        """Clears all jobs from the queue."""
        if self.job_queue and messagebox.askyesno("Confirm", "Are you sure you want to remove all jobs?"):
            self.job_queue.clear()
            self.selected_job_id = None
            self._update_job_list_display()
            
            # İş listesi boşaldığı için varsayılanları göster ve UI'ı devre dışı bırak.
            self._populate_settings_panel()

    def _move_job(self, direction):
        """Moves the selected job up or down in the list."""
        if not self.selected_job_id: return
        try:
            index = next(i for i, job in enumerate(self.job_queue) if job["id"] == self.selected_job_id)
        except StopIteration: return
            
        if direction == "up" and index > 0:
            new_index = index - 1
        elif direction == "down" and index < len(self.job_queue) - 1:
            new_index = index + 1
        else: return

        self.job_queue.insert(new_index, self.job_queue.pop(index))
        self._update_job_list_display()

    def _select_job(self, job_id):
        """Marks a job as selected, populates the settings panel, and ENABLES the UI."""
        self.selected_job_id = job_id
        self._update_job_list_display()

        # 1. Ayarları seçili işin verileriyle doldur.
        self._populate_settings_panel()
        # 2. Ayar panelini kullanıcı için aktif hale getir.
        self._toggle_setting_widgets_state(True)

    def _start_pipeline_thread(self):
        """Starts the pipeline in a separate thread."""
        if self.is_running_pipeline: return
        if not self.job_queue:
            messagebox.showinfo("Information", "Please add a job first.")
            return
        if not os.path.exists(self.python_executable):
            self.log("ERROR", f"Python executable not found at: {self.python_executable}")
            messagebox.showerror("Error", "Python executable in the virtual environment (venv) could not be found!")
            return
            
        self._toggle_ui_state(True)
        self.pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.pipeline_thread.start()

    def _run_pipeline(self):
        """Processes the job queue sequentially."""
        self.log("PIPELINE", "Translation pipeline started...")
        total_jobs = len(self.job_queue)

        for i, job in enumerate(list(self.job_queue)):
            if not self.is_running_pipeline:
                self.log("PIPELINE", "Pipeline stopped by user.")
                break

            self.after(0, self._update_progress, (i / total_jobs, f"Processing {i+1}/{total_jobs}: {os.path.basename(job['source_path'])}"))
            
            if job['settings'].get('backup_original'): self._backup_folder(job['source_path'])
            
            config_path, command = self._build_command_for_job(job)
            self.log("DEBUG", f"Executing command: {' '.join(command)}")

            try:
                # Store the process object in the CORRECT variable
                self.pipeline_process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                # Read from the CORRECT variable
                for line in iter(self.pipeline_process.stdout.readline, ''):
                    if line.strip(): self.log("RAW", line.strip())
                
                # Wait for the CORRECT variable to finish
                return_code = self.pipeline_process.wait()

                if return_code == 0:
                    self.log("SUCCESS", f"Job finished successfully: {os.path.basename(job['source_path'])}")
                    if job['settings'].get('auto_rename'): self._rename_output_folder(job)
                else:
                    self.log("ERROR", f"Job '{os.path.basename(job['source_path'])}' failed with exit code: {return_code}.")
            
            except FileNotFoundError:
                self.log("ERROR", f"Command not found: '{self.python_executable}'. Please check your venv setup.")
                break
            except Exception as e:
                self.log("ERROR", f"A critical pipeline error occurred: {e}")
                traceback.print_exc()
                break
            finally:
                # Reset the process variable after the job is done or fails
                self.pipeline_process = None
                if os.path.exists(config_path):
                    try:
                        os.remove(config_path)
                    except Exception as e:
                        self.log("ERROR", f"Could not remove temp config file: {e}")
        
        self.log("PIPELINE", "All jobs completed!")
        self.after(0, self._update_progress, (1.0, "Ready"))
        self.after(0, self._toggle_ui_state, False)

    def _stop_pipeline(self):
        """Stops the running pipeline."""
        self.log("PIPELINE", "Stop command received. The process will halt after the current job finishes.")
        self._toggle_ui_state(False)

    def _build_command_for_job(self, job, output_path=None):
        """
        Builds a clean, optimized, and nested config from all UI settings,
        omitting sections that are disabled (set to 'none').
        """
        settings = job['settings']

        def _get(key, type_func=str, default=None):
            val = settings.get(key)
            if val is None or val == '': val = default
            if val is None: return None
            try:
                if isinstance(val, str) and val.lower() == "disabled":
                    return None if key == 'upscale_ratio' else "none"
                return type_func(val)
            except (ValueError, TypeError):
                return default

        config_data = {
            "translator": {"translator": _get("translator"), "target_lang": _get("target_lang"), "source_lang": _get("source_lang"), "no_text_lang_skip": _get("no_text_lang_skip", bool), "skip_lang": _get("skip_lang"), "translator_chain": _get("translator_chain")},
            "detector": {"detector": _get("detector"), "detection_size": _get("detection_size", int), "text_threshold": _get("text_threshold", float), "box_threshold": _get("box_threshold", float), "unclip_ratio": _get("unclip_ratio", float), "det_rotate": _get("det_rotate", bool), "det_auto_rotate": _get("det_auto_rotate", bool), "det_invert": _get("det_invert", bool), "det_gamma_correct": _get("det_gamma_correct", bool)},
            "ocr": {"ocr": _get("ocr"), "use_mocr_merge": _get("use_mocr_merge", bool), "min_text_length": _get("min_text_length", int), "ignore_bubble": _get("ignore_bubble", int), "prob": _get("prob", float)},
            "inpainter": {"inpainter": _get("inpainter"), "inpainting_size": _get("inpainting_size", int), "inpainting_precision": _get("inpainting_precision")},
            "colorizer": {"colorizer": _get("colorizer"), "colorization_size": _get("colorization_size", int), "denoise_sigma": _get("denoise_sigma", int)},
            "upscale": {"upscaler": _get("upscaler"), "revert_upscaling": _get("revert_upscaling", bool), "upscale_ratio": _get("upscale_ratio", int)},
            "render": {"renderer": _get("renderer"), "gimp_font": _get("font_family"), "font_color": _get("font_color"), "font_size": _get("font_size", int), "font_size_offset": _get("font_size_offset", int), "font_size_minimum": _get("font_size_minimum", int), "line_spacing": _get("line_spacing", int), "alignment": _get("alignment"), "direction": _get("direction"), "disable_font_border": _get("disable_font_border", bool), "uppercase": _get("uppercase", bool), "lowercase": _get("lowercase", bool), "no_hyphenation": _get("no_hyphenation", bool), "rtl": _get("rtl", bool)},
            "filter_text": _get("filter_text"), "kernel_size": _get("kernel_size", int), "mask_dilation_offset": _get("mask_dilation_offset", int)
        }
        
        # --- OPTIMIZATION LOGIC ---
        final_config = {}
        for section, content in config_data.items():
            if isinstance(content, dict):
                main_key = section
                # The main model key is usually the same as the section name
                if content.get(main_key) and content[main_key] != 'none':
                    final_config[section] = {k: v for k, v in content.items() if v is not None}
            else: # For root-level settings
                if content is not None:
                    final_config[section] = content
        
        # Special case: If translator is 'none', it implies render should also be 'none'
        if final_config.get("translator", {}).get("translator") == "none":
            final_config["render"] = {"renderer": "none"}
        # --- END OF OPTIMIZATION ---

        os.makedirs(TEMP_DIR, exist_ok=True)
        config_path = os.path.join(TEMP_DIR, f"temp_config_{job['id']}.json")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=4)
                
        if output_path is None:
            output_path = f"{job['source_path']}_translated"
        
        command = [self.python_executable, "-m", "manga_translator", "local", "-i", job['source_path'], "-o", output_path, "--config-file", config_path]
        if self.enable_advanced_debug.get():
            # Eğer işaretliyse, komutun sonuna "-v" ekle
            command.append("-v")
        if settings.get('processing_device') == 'NVIDIA GPU':
            command.append("--use-gpu")
        
        return config_path, command

    # 5. Visual Compare Interactive Logic

    def _load_test_image(self):
        """Loads a test image and forces a full recreation of the canvas layout."""
        file_path = filedialog.askopenfilename(title="Select a Test Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if not file_path: return
        
        self.test_image_path = file_path
        self.log("INFO", f"Loaded test image: {os.path.basename(file_path)}")
        
        try:
            self.original_pil_image = Image.open(file_path)
            self.translated_pil_image = None
            
            # Reset view and force the layout to be recreated for the new image
            self._reset_canvas_view(force_recreate=True)
            
            self.run_test_button.configure(state="normal")
        except Exception as e:
            self.log("ERROR", f"Failed to load image file: {e}")

    def _run_visual_test(self):
        """
        Runs the translation pipeline on the single loaded test image
        by creating temporary input/output directories for robust processing.
        """
        if not hasattr(self, 'test_image_path') or not self.test_image_path:
            messagebox.showwarning("No Image", "Please load a test image first.")
            return

        self.log("PIPELINE", "Starting visual test on single image...")
        self._toggle_setting_widgets_state(False) # Disable settings during test
        self.run_test_button.configure(state="disabled", text="Testing...")
        
        # --- NEW ROBUST SETUP ---
        # 1. Define temporary directories
        temp_input_dir = os.path.join(TEMP_DIR, "visual_test_input")
        temp_output_dir = os.path.join(TEMP_DIR, "visual_test_output")

        # 2. Clean up and create fresh directories
        if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        os.makedirs(temp_input_dir)
        os.makedirs(temp_output_dir)

        # 3. Copy the selected test image into the temporary input directory
        try:
            shutil.copy(self.test_image_path, temp_input_dir)
        except Exception as e:
            self.log("ERROR", f"Failed to copy test image to temp directory: {e}")
            self._toggle_setting_widgets_state(True)
            self.run_test_button.configure(state="normal", text="Run Test with Current Settings")
            return
        # --- END OF NEW SETUP ---

        # Use current default settings for the test
        test_settings = self.default_job_settings.copy()
        
        # --- NEW: Aggressive Fast Preview Logic ---
        if self.fast_preview_check.get():
            self.log("INFO", "Fast Preview enabled. Overriding settings for speed.")
            # Lower resolutions to reduce memory usage significantly
            test_settings['detection_size'] = 1024
            test_settings['inpainting_size'] = 1024
            # Use a faster precision mode if on GPU
            if test_settings['processing_device'] == 'NVIDIA GPU':
                test_settings['inpainting_precision'] = 'fp16'
        # --- END NEW ---

        test_job = {
            "id": "visual_test",
            "source_path": temp_input_dir,
            "settings": test_settings # Use the (potentially overridden) settings
        }

        def _run_test_thread():
            config_path, command = self._build_command_for_job(test_job, output_path=temp_output_dir)
            
            try:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                for line in iter(process.stdout.readline, ''):
                    if line.strip(): self.log("RAW", line.strip())
                
                return_code = process.wait()

                if return_code == 0:
                    self.log("SUCCESS", "Visual test completed successfully.")
                    result_files = os.listdir(temp_output_dir)
                    if result_files:
                        result_path = os.path.join(temp_output_dir, result_files[0])
                        self.after(0, self._display_test_result, result_path)
                    else:
                        self.log("ERROR", "Test ran, but no output file was found in the temp directory.")
                else:
                    self.log("ERROR", f"Visual test failed with exit code {return_code}.")

            except Exception as e:
                self.log("ERROR", f"A critical error occurred during the visual test: {e}")
            finally:
                if os.path.exists(config_path): os.remove(config_path)
                # Re-enable UI elements on the main thread
                self.after(0, self._toggle_setting_widgets_state, True)
                self.after(0, self.run_test_button.configure, {"state": "normal", "text": "Run Test with Current Settings"})

        threading.Thread(target=_run_test_thread, daemon=True).start()

    def _display_test_result(self, image_path):
        """Loads the result image and triggers a redraw."""
        try:
            self.translated_pil_image = Image.open(image_path)
            # Redraw without forcing layout recreation
            self.after(0, self._redraw_images)
        except Exception as e:
            self.log("ERROR", f"Failed to load result image: {e}")

    def _redraw_images(self, event=None, force_recreate=False):
        """
        The main drawing manager. It dynamically sets the layout only when necessary
        and then redraws the content inside the canvases.
        """
        if not self.original_pil_image: return

        img_w, img_h = self.original_pil_image.size
        is_horizontal_layout = img_w > img_h
        layout_changed = self._current_layout != is_horizontal_layout
        
        if force_recreate or layout_changed or not self.original_canvas:
            self._current_layout = is_horizontal_layout
            
            for widget in self.image_area_frame.winfo_children():
                widget.destroy()

            if is_horizontal_layout:
                self.image_area_frame.grid_rowconfigure((1, 3), weight=1); self.image_area_frame.grid_columnconfigure(0, weight=1); self.image_area_frame.grid_columnconfigure(1, weight=0)
                r1, c1, r2, c2 = (1, 0, 3, 0)
            else:
                self.image_area_frame.grid_rowconfigure(1, weight=1); self.image_area_frame.grid_rowconfigure(3, weight=0); self.image_area_frame.grid_columnconfigure((0, 1), weight=1)
                r1, c1, r2, c2 = (1, 0, 1, 1)

            ctk.CTkLabel(self.image_area_frame, text="Original (Ctrl+Scroll=Zoom, Drag=Pan)").grid(row=r1-1, column=c1, pady=(0, 5))
            self.original_canvas = ctk.CTkCanvas(self.image_area_frame, background="#2b2b2b", highlightthickness=0)
            self.original_canvas.grid(row=r1, column=c1, padx=(0, 5) if not is_horizontal_layout else 0, pady=(0,5) if is_horizontal_layout else 0, sticky="nsew")

            ctk.CTkLabel(self.image_area_frame, text="Output").grid(row=r2-1, column=c2, pady=(0, 5))
            self.translated_canvas = ctk.CTkCanvas(self.image_area_frame, background="#2b2b2b", highlightthickness=0)
            self.translated_canvas.grid(row=r2, column=c2, padx=(5, 0) if not is_horizontal_layout else 0, pady=(5,0) if is_horizontal_layout else 0, sticky="nsew")

            for canvas in [self.original_canvas, self.translated_canvas]:
                canvas.bind("<Control-MouseWheel>", self._on_zoom)
                canvas.bind("<ButtonPress-1>", self._on_pan_start)
                canvas.bind("<B1-Motion>", self._on_pan_move)
        
        # Always redraw the image content on the canvases
        if self.original_canvas and self.original_pil_image:
            self._display_image_in_canvas(self.original_canvas, self.original_pil_image, "original")
        if self.translated_canvas and self.translated_pil_image:
            self._display_image_in_canvas(self.translated_canvas, self.translated_pil_image, "translated")
        elif self.translated_canvas:
            self.translated_canvas.delete("all")

    def _display_image_in_canvas(self, canvas, pil_image, key):
        # This function remains correct from the previous version.
        canvas.delete("all")
        if not pil_image: return
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return
        zoomed_w = int(pil_image.width * self.canvas_zoom_level)
        zoomed_h = int(pil_image.height * self.canvas_zoom_level)
        if zoomed_w > 4096 or zoomed_h > 4096:
             pil_image.thumbnail((4096,4096)); zoomed_w=pil_image.width; zoomed_h=pil_image.height
        if zoomed_w <= 0 or zoomed_h <= 0: return
        resized_image = pil_image.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)
        ctk_image = ImageTk.PhotoImage(resized_image)
        self.image_references[key] = ctk_image
        x = (canvas_width / 2) + self.canvas_pan_offset[0]
        y = (canvas_height / 2) + self.canvas_pan_offset[1]
        canvas.create_image(x, y, image=self.image_references[key], anchor="center")

    def _reset_canvas_view(self, force_recreate=False):
        """Resets pan and fits the image to the screen."""
        self.canvas_pan_offset = [0, 0]
        if force_recreate:
            self._redraw_images(force_recreate=True)
        self.after(50, self._fit_image_to_view)

    def _fit_image_to_view(self):
        # This function remains correct. It calculates the best zoom level.
        if not self.original_pil_image or not self.original_canvas: return
        canvas_width = self.original_canvas.winfo_width() - 20
        canvas_height = self.original_canvas.winfo_height() - 20
        if canvas_width <= 1 or canvas_height <= 1: return
        img_w, img_h = self.original_pil_image.size
        zoom_w = canvas_width / img_w
        zoom_h = canvas_height / img_h
        self.canvas_zoom_level = min(zoom_w, zoom_h)
        self.zoom_label.configure(text=f"Zoom: {self.canvas_zoom_level*100:.0f}%")
        self._redraw_images()

    # 6. Event Handlers

    def _on_setting_change(self, key, value):
        """
        Callback for when a setting is changed. 
        It updates the settings dictionary of the currently selected job.
        If no job is selected, it updates the application's default settings template.
        """
        settings_source = None
        job = next((j for j in self.job_queue if j.get("id") == self.selected_job_id), None)

        if job:
            # If a job is selected, get its specific settings dictionary
            settings_source = job.get('settings', {})
            log_prefix = f"Job '{os.path.basename(job['source_path'])}'"
        else:
            # If NO job is selected, get the global default settings dictionary
            settings_source = self.default_job_settings
            log_prefix = "Default settings"

        # --- SPECIAL HANDLING FOR SKIP_LANG CHECKBOXES ---
        if key == 'skip_lang':
            current_skips = set(settings_source.get('skip_lang', '').split(','))
            current_skips.discard('')
            lang_code, is_checked = value
            if is_checked:
                current_skips.add(lang_code)
            else:
                current_skips.discard(lang_code)
            
            # Update the setting with the new comma-separated string
            settings_source[key] = ",".join(sorted(list(current_skips)))
        else:
            # --- NORMAL BEHAVIOR ---
            # Apply the new value directly to the determined settings source
            settings_source[key] = value
        
        # If no job is selected, we have just modified the defaults.
        # If a job IS selected, we should also update the defaults for the next job to use.
        if job:
            self.default_job_settings[key] = value

        # If a language was changed, the translator options might need to be re-evaluated.
        # This part requires the new _update_translator_options logic.
        if key in ['source_lang', 'target_lang']:
            self._update_translator_options()

    def _on_translator_select(self, value):
        """Special handler for the translator dropdown."""
        if value.startswith("---"): # User clicked a separator
            current_job = next((j for j in self.job_queue if j["id"] == self.selected_job_id), None)
            if current_job:
                self.setting_widgets['translator'].set(current_job['settings'].get('translator', 'sugoi'))
            return
        self._on_setting_change('translator', value)
        self._populate_settings_panel() # Repopulate to handle UI state changes
 
    def _on_resize(self, event=None):
        current_size = (self.winfo_width(), self.winfo_height())
        if current_size != self.last_known_size:
            self.last_known_size = current_size
            if hasattr(self, '_resize_timer'):
                self.after_cancel(self._resize_timer)
            self._resize_timer = self.after(150, self._fit_image_to_view)

    def _on_zoom(self, event):
        if not self.original_pil_image: return "break"
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        new_zoom = max(0.1, min(self.canvas_zoom_level * factor, 8.0))
        if new_zoom != self.canvas_zoom_level:
            self.canvas_zoom_level = new_zoom
            self.zoom_label.configure(text=f"Zoom: {self.canvas_zoom_level*100:.0f}%")
            self._redraw_images()
        return "break"

    def _on_pan_start(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def _on_pan_move(self, event):
        if not self.original_pil_image: return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.canvas_pan_offset[0] += dx
        self.canvas_pan_offset[1] += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self._redraw_images()
        
    def _handle_drop(self, event):
        """Handles the drop event for adding new jobs."""
        paths = self.tk.splitlist(event.data)
        for path in paths:
            if os.path.isdir(path):
                self._add_job_from_path(path)
            else:
                self.log("WARNING", f"Only folders are accepted: {path}")
    
    def _on_closing(self):
        """Handles the window closing event, managing running processes."""
        # Check if a pipeline process is currently active
        if self.is_running_pipeline and self.pipeline_process:
            if messagebox.askyesno("Confirm Exit", "A translation job is currently running. Are you sure you want to stop it and exit?"):
                # If user confirms, terminate the process
                try:
                    self.log("PIPELINE", "Termination command received from user. Shutting down...")
                    self.pipeline_process.kill() # Forcefully stops the subprocess
                except Exception as e:
                    self.log("ERROR", f"Failed to terminate the running process: {e}")
                
                # Proceed with closing
                self._save_settings_and_destroy()
            else:
                # If user cancels, do nothing and keep the window open
                return
        else:
            # If no process is running, just close normally
            self._save_settings_and_destroy()
        
    # 7. Helper & Utility Functions  

    def _update_job_list_display(self):
        """Refreshes the job list display on the left panel."""
        for widget in self.job_list_frame.winfo_children():
            widget.destroy()
            
        if not self.job_queue:
            self.job_list_frame.configure(label_text="Drag & drop folders here")
            return
        
        self.job_list_frame.configure(label_text="")
        for i, job in enumerate(self.job_queue):
            summary = f"{i + 1}. {os.path.basename(job['source_path'])}"
            color = ctk.ThemeManager.theme["CTkButton"]["fg_color"] if job['id'] == self.selected_job_id else "transparent"
            btn = ctk.CTkButton(self.job_list_frame, text=summary, fg_color=color, anchor="w", command=lambda j_id=job['id']: self._select_job(j_id))
            btn.pack(fill="x", padx=5, pady=3)

    def _populate_settings_panel(self):
        settings_source = None
        job = next((j for j in self.job_queue if j.get("id") == self.selected_job_id), None)

        if job:
            settings_source = job.get('settings', {})
        else:
            settings_source = self.default_job_settings

        for key, widget_or_group in self.setting_widgets.items():
            if key in ['skip_lang_checkboxes', 'color_picker_button']:
                continue

            value = settings_source.get(key, FACTORY_SETTINGS.get(key))

            if isinstance(widget_or_group, tuple):
                slider, label = widget_or_group
                
                slider_pos = 0
                label_text = ""

                # This block now correctly sets the initial state for all slider types
                if key == 'unclip_ratio':
                    slider_pos = float(value * 10) if value is not None else 23
                    label_text = f"{value:.1f}" if value is not None else "2.3"
                elif key in ['text_threshold', 'box_threshold']:
                    slider_pos = float(value * 100) if value is not None else 50
                    label_text = f"{value:.2f}" if value is not None else "0.50"
                elif key == 'font_size':
                    slider_pos = float(value) if value is not None else 0
                    label_text = str(int(slider_pos)) if value is not None else "auto"
                elif key == 'line_spacing':
                    slider_pos = float(value) if value is not None else 0
                    label_text = str(int(slider_pos)) if value is not None else "auto"
                elif key == 'font_size_minimum':
                    slider_pos = float(value) if value is not None else -1
                    label_text = str(int(slider_pos)) if value != -1 else "disabled"
                else: # For all other standard integer sliders
                    slider_pos = float(value) if value is not None else 0
                    label_text = f"{int(slider_pos)}"
                
                slider.set(slider_pos)
                label.configure(text=label_text)
            
            elif isinstance(widget_or_group, (ctk.CTkOptionMenu, ctk.CTkComboBox, ctk.CTkSegmentedButton)):
                if key in ['source_lang', 'target_lang']:
                    display_value = next((k for k, v in LANGUAGES.items() if v == value), value)
                    widget_or_group.set(display_value)
                elif key == 'upscale_ratio':
                    val_to_set = f"{value}x" if value and str(value) != 'Disabled' else 'Disabled'
                    widget_or_group.set(val_to_set)
                else:
                    widget_or_group.set(str(value) if value is not None else "")
            
            elif isinstance(widget_or_group, ctk.CTkEntry):
                widget_or_group.delete(0, 'end')
                widget_or_group.insert(0, str(value) if value is not None else "")
            
            elif isinstance(widget_or_group, ctk.CTkCheckBox):
                if value: widget_or_group.select()
                else: widget_or_group.deselect()
        
        skipped_langs = set(settings_source.get('skip_lang', '').split(','))
        if 'skip_lang_checkboxes' in self.setting_widgets:
            for code, checkbox in self.setting_widgets['skip_lang_checkboxes'].items():
                if code in skipped_langs: checkbox.select()
                else: checkbox.deselect()

        self._update_translator_options()
        if 'translator' in self.setting_widgets:
             self.setting_widgets['translator'].set(settings_source.get('translator', 'sugoi'))
        
        is_raw_mode = settings_source.get('translator', '') == 'none'
        render_state = "disabled" if is_raw_mode else "normal"
        render_keys = ['renderer', 'font_family', 'font_size_offset', 'alignment', 'color_picker_button', 'font_color']
        for r_key in render_keys:
            if r_key in self.setting_widgets:
                widget_or_group = self.setting_widgets[r_key]
                if isinstance(widget_or_group, tuple):
                    for item in widget_or_group: item.configure(state=render_state)
                else:
                    widget_or_group.configure(state=render_state)
        if 'target_lang' in self.setting_widgets:
            self.setting_widgets['target_lang'].configure(state="disabled" if is_raw_mode else "normal")

    def _toggle_ui_state(self, is_running):
        """Toggles the state of UI elements ONLY when the pipeline is running or stopping."""
        self.is_running_pipeline = is_running
        pipeline_state = "disabled" if is_running else "normal"
        
        # Disable/Enable job list and its controls
        for frame in [self.job_controls_frame, self.reorder_frame]:
            for widget in frame.winfo_children():
                if isinstance(widget, ctk.CTkButton):
                    widget.configure(state=pipeline_state)
        
        for widget in self.job_list_frame.winfo_children():
             if isinstance(widget, ctk.CTkButton):
                widget.configure(state=pipeline_state)

        # Toggle the main pipeline buttons
        self.start_button.configure(state=pipeline_state)
        self.stop_button.configure(state="normal" if is_running else "disabled")
        
        # --- SIMPLIFIED LOGIC ---
        # The settings panel is ONLY disabled when the pipeline is running.
        # Otherwise, it's always enabled.
        self._toggle_setting_widgets_state(not is_running)

    def _toggle_setting_widgets_state(self, is_enabled):
        """
        Sets the state of all widgets, correctly handling nested dictionaries (for skip_lang)
        and tuples (for sliders with value labels).
        """
        state = "normal" if is_enabled else "disabled"
        
        for key, widget_or_group in self.setting_widgets.items():
            if key == 'skip_lang_checkboxes':
                for checkbox_widget in widget_or_group.values(): checkbox_widget.configure(state=state)
            elif isinstance(widget_or_group, tuple):
                for item in widget_or_group: item.configure(state=state)
            else:
                if widget_or_group: widget_or_group.configure(state=state)

        if is_enabled:
            job = next((j for j in self.job_queue if j["id"] == self.selected_job_id), None)
            if not job: return

            is_raw_mode = str(job['settings'].get('translator', '')).lower() == 'none'
            render_state = "disabled" if is_raw_mode else "normal"
            render_keys = ['renderer', 'font_family', 'font_size_offset', 'alignment', 'color_picker_button', 'font_color', 'font_size', 'font_size_minimum', 'line_spacing', 'direction', 'disable_font_border', 'uppercase', 'lowercase', 'no_hyphenation', 'rtl']
            
            for r_key in render_keys:
                if r_key in self.setting_widgets:
                    widget_or_group = self.setting_widgets[r_key]
                    if isinstance(widget_or_group, tuple):
                        for item in widget_or_group: item.configure(state=render_state)
                    else:
                        widget_or_group.configure(state=render_state)

            if 'target_lang' in self.setting_widgets:
                self.setting_widgets['target_lang'].configure(state="disabled" if is_raw_mode else "normal")

    def _update_translator_options(self):
        """Dynamically filters the translator list based on selected languages."""
        if not self.selected_job_id: return
        job = next((j for j in self.job_queue if j["id"] == self.selected_job_id), None)
        if job:
            settings_source = job.get('settings', {})
        else:
            settings_source = self.default_job_settings

        source_lang = job['settings'].get('source_lang', 'Auto-Detect')
        target_lang = job['settings'].get('target_lang', 'English')

        valid_translators = []
        
        # This logic determines which translators to show in the dropdown
        for name, caps in TRANSLATOR_CAPABILITIES.items():
            # Always include special actions
            if caps["source"] == ["any"]:
                valid_translators.append(name)
                continue
            
            # Check for multilingual models
            is_universal_source = "all" in caps["source"]
            is_universal_target = "all" in caps["target"]
            
            # Check for specific language support
            source_supported = is_universal_source or source_lang in caps["source"] or source_lang == 'auto'
            target_supported = is_universal_target or target_lang in caps["target"]
            
            if source_supported and target_supported:
                valid_translators.append(name)
        
        # Add separators for readability
        final_list = []
        groups = {
            "--- OFFLINE MODELS (No API Key) ---": ["sugoi", "m2m100", "m2m100_big", "nllb", "nllb_big", "mbart50", "jparacrawl", "jparacrawl_big", "qwen2", "qwen2_big", "offline"],
            "--- API-BASED (Requires Setup) ---": ["deepl", "gemini", "deepseek", "groq", "youdao", "baidu", "caiyun", "sakura", "papago", "openai", "custom_openai"],
            "--- OTHER ACTIONS ---": ["original", "none"]
        }

        for group_name, members in groups.items():
            group_members_in_list = [m for m in members if m in valid_translators]
            if group_members_in_list:
                final_list.append(group_name)
                final_list.extend(group_members_in_list)

        self.setting_widgets['translator'].configure(values=final_list)

        # If the currently selected translator is no longer valid, select the first valid one
        current_translator = job['settings'].get('translator', 'sugoi')
        if current_translator not in final_list:
            # Find the first non-separator item
            new_default = next((item for item in final_list if not item.startswith("---")), "sugoi")
            self.setting_widgets['translator'].set(new_default)
            self._on_setting_change('translator', new_default)

    def _reset_current_tab_settings(self):
        """Resets only the settings for the currently selected tab to their factory defaults."""
        current_tab_name = self.settings_tabs.get()
        
        if current_tab_name not in self.SETTING_GROUPS:
            self.log("INFO", f"No resettable settings found for tab: {current_tab_name}")
            return
            
        keys_to_reset = self.SETTING_GROUPS[current_tab_name]
        
        # Find the correct settings dictionary to modify (selected job or global defaults)
        job = next((j for j in self.job_queue if j.get("id") == self.selected_job_id), None)
        if job:
            settings_source = job['settings']
        else:
            # If no job is selected, reset the global default settings instead
            settings_source = self.default_job_settings

        if not messagebox.askyesno("Confirm Reset", 
                                   f"Are you sure you want to reset all settings in the '{current_tab_name}' tab to their factory defaults?"):
            return

        self.log("PIPELINE", f"Resetting settings for tab: '{current_tab_name}'")
        for key in keys_to_reset:
            if key in FACTORY_SETTINGS:
                # Revert the key's value back to the one in FACTORY_SETTINGS
                settings_source[key] = FACTORY_SETTINGS[key]
        
        # Repopulate the entire settings panel to reflect the changes
        self._populate_settings_panel()
        self.log("SUCCESS", f"Tab '{current_tab_name}' has been reset.")

    def _reset_all_settings(self):
        """Resets ALL settings to their factory defaults after a confirmation."""
        if not messagebox.askyesno("Confirm Reset", 
                                   "Are you sure you want to reset ALL settings to their factory defaults?\n\n"
                                   "This action cannot be undone."):
            return

        # Ayarları güncellemek için bir kaynak bul (seçili iş veya varsayılanlar)
        job = next((j for j in self.job_queue if j.get("id") == self.selected_job_id), None)
        settings_source = job['settings'] if job else self.default_job_settings
        
        self.log("PIPELINE", "Resetting ALL settings to factory defaults!")
        for key in FACTORY_SETTINGS:
            settings_source[key] = FACTORY_SETTINGS[key]
            
        # Değişiklikleri göstermek için arayüzü yeniden doldur
        self._populate_settings_panel()

    def _get_project_fonts(self):
        """Scans the local 'fonts' directory and returns a list of font files."""
        project_fonts = []
        fonts_dir = "fonts"
        if os.path.isdir(fonts_dir):
            try:
                for filename in os.listdir(fonts_dir):
                    if filename.lower().endswith(('.ttf', '.otf')):
                        # Return the relative path for the config
                        project_fonts.append(os.path.join(fonts_dir, filename).replace("\\", "/"))
            except Exception as e:
                self.log("ERROR", f"Could not read project fonts folder: {e}")
        return sorted(project_fonts)

    def _choose_font_color(self):
        """Opens a color picker dialog and updates the font color setting."""
        current_color = f"#{self.setting_widgets['font_color'].get()}"
        chosen_color = colorchooser.askcolor(color=current_color, title="Choose Font Color")
        
        if chosen_color and chosen_color[1]: # Check if a color was selected
            hex_color = chosen_color[1].lstrip('#').upper()
            self.setting_widgets['font_color'].delete(0, 'end')
            self.setting_widgets['font_color'].insert(0, hex_color)
            self._on_setting_change('font_color', hex_color)

    def refresh_model_manager_ui(self):
        """Clears and rebuilds the model list in the Model Manager UI."""
        for widget in self.model_list_frame.winfo_children():
            widget.destroy()

        for category_name, models in self.model_manager.MODELS_INFO.items():
            ctk.CTkLabel(self.model_list_frame, text=category_name, font=ctk.CTkFont(size=14, weight="bold")).pack(fill="x", padx=5, pady=(10, 5))
            
            for display_name, info in models.items():
                model_frame = ctk.CTkFrame(self.model_list_frame)
                model_frame.pack(fill="x", padx=5, pady=3)
                model_frame.grid_columnconfigure(1, weight=1)

                is_manageable = "file_path" in info or "check_path" in info
                
                left_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
                left_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                ctk.CTkLabel(left_frame, text=display_name, font=ctk.CTkFont(weight="bold")).pack(anchor="w")
                ctk.CTkLabel(left_frame, text=info['description'], text_color="gray60").pack(anchor="w")

                right_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
                right_frame.grid(row=0, column=1, padx=10, pady=5, sticky="e")

                if is_manageable:
                    is_downloaded = self.model_manager.check_model_exists(display_name)
                    status_text = f"Downloaded ({info.get('size', 'N/A')})" if is_downloaded else "Not Downloaded"
                    status_color = "#2ECC71" if is_downloaded else "#E67E22"
                    
                    if is_downloaded:
                        action_button = ctk.CTkButton(right_frame, text="Delete", fg_color="#D32F2F", hover_color="#B71C1C", width=80,
                                                      command=lambda n=display_name: self.model_manager.delete_model(n))
                    else:
                        action_button = ctk.CTkButton(right_frame, text="Download", width=80,
                                                      command=lambda n=display_name: self.model_manager.trigger_download(n))
                else:
                    status_text = info.get("source", "Built-in")
                    status_color = "#3498DB"
                    action_button = ctk.CTkButton(right_frame, text="Included", state="disabled", width=80)
                
                ctk.CTkLabel(right_frame, text=status_text, text_color=status_color).pack(side="left", padx=10)
                action_button.pack(side="left")

    def _rename_output_folder(self, job):
        """Renames the output folder based on settings, handling overwrites."""
        try:
            settings = job['settings']
            is_raw_mode = str(settings.get('translator', '')).lower() == 'none'
            
            suffix = "RAW" if is_raw_mode else settings.get('target_lang', 'TRNS')
            old_name_base = f"{os.path.basename(job['source_path'])}_translated"
            new_name_base = f"{os.path.basename(job['source_path'])}-{suffix}"
            
            output_dir = os.path.dirname(job['source_path'])
            full_old_path = os.path.join(output_dir, old_name_base)
            full_new_path = os.path.join(output_dir, new_name_base)

            if not os.path.exists(full_old_path):
                self.log("WARNING", f"Could not find output folder to rename: {full_old_path}")
                return

            # If the target path exists...
            if os.path.exists(full_new_path):
                # ...and overwrite is enabled, delete the existing folder.
                if settings.get('overwrite_output', False):
                    self.log("INFO", f"Overwrite enabled. Deleting existing folder: {os.path.basename(full_new_path)}")
                    shutil.rmtree(full_new_path)
                # ...and overwrite is disabled, find a new name with a counter.
                else:
                    count = 2
                    while os.path.exists(f"{full_new_path} ({count})"):
                        count += 1
                    full_new_path = f"{full_new_path} ({count})"

            os.rename(full_old_path, full_new_path)
            self.log("SUCCESS", f"Output folder renamed to: '{os.path.basename(full_new_path)}'")
        except Exception as e:
            self.log("ERROR", f"Failed to rename output folder: {e}")

    def _backup_folder(self, source_path):
        """Creates a backup of a given folder."""
        try:
            backup_path = f"{source_path}_backup_{int(time.time())}"
            self.log("INFO", f"Backing up '{os.path.basename(source_path)}' to '{os.path.basename(backup_path)}'")
            shutil.copytree(source_path, backup_path)
            self.log("SUCCESS", "Backup successful.")
        except Exception as e:
            self.log("ERROR", f"Backup failed: {e}")

    def log(self, level, message):
        """Writes a colored message to the log window, with intelligent error parsing."""
        log_colors = {"PIPELINE": "#5DADE2", "SUCCESS": "#2ECC71", "ERROR": "#E74C3C", 
                      "WARNING": "#F39C12", "INFO": "#FFFFFF", "DEBUG": "gray", "HINT": "#FFD700"}
        
        original_message = message.strip()
        msg_lower = original_message.lower()

        # --- Intelligent Error and Hint Detection ---
        if level == "RAW":
            # Default to INFO for raw output
            level = "INFO" 
            # Promote to ERROR only if it starts with a clear error keyword
            if msg_lower.startswith(('error:', 'validationerror:', 'exception:', 'traceback')):
                level = "ERROR"
            
            # Smart Hint System: Look for common user problems in the backend output
            if "out of memory" in msg_lower or "allocation failed" in msg_lower:
                hint_message = ("[HINT] [Code: VRAM-01] The process ran out of memory. This can happen with high-resolution settings. "
                                "Try lowering 'Detection Size' or 'Inpainting Size'.")
                self.after(0, self._insert_log_text, f"[{time.strftime('%H:%M:%S')}] {hint_message}\n", log_colors["HINT"])
            elif "corrupted" in msg_lower or "invalid image" in msg_lower:
                hint_message = ("[HINT] [Code: IMG-01] A corrupted or unsupported image was found. "
                                "Please check the files in the source folder.")
                self.after(0, self._insert_log_text, f"[{time.strftime('%H:%M:%S')}] {hint_message}\n", log_colors["HINT"])

        # This prevents the "Namespace(...)" line from ever being shown, as it's not useful user info.
        if msg_lower.startswith("[local] namespace"):
            return

        print(f"[{level}] {original_message}")
        self.after(0, self._insert_log_text, f"[{time.strftime('%H:%M:%S')}] [{level}] {original_message}\n", log_colors.get(level, "gray"))

    def _insert_log_text(self, message, color):
        try:
            self.log_textbox.configure(state="normal")
            tag_name = f"log_{time.time()}"
            self.log_textbox.tag_config(tag_name, foreground=color)
            self.log_textbox.insert("end", message, (tag_name,))
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")
        except Exception: pass

    def _clear_log(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def _update_progress(self, args):
        percent, text = args
        if percent is not None: self.progress_bar.set(percent)
        if text is not None: self.progress_label.configure(text=text)

    def _save_settings_and_destroy(self):
        """Saves settings and properly closes the application."""
        try:
            settings_to_save = {
                "window_geometry": self.geometry(),
                "default_job_settings": self.default_job_settings
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=4)
            self.log("INFO", "Application settings saved.")
        except Exception as e:
            self.log("ERROR", f"Failed to save settings on exit: {e}")
        
        self.destroy()

# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    app = TranslatorStudioApp()
    app.mainloop()