# Manga Translation Studio

This document describes the **Manga Translation Studio**, a comprehensive graphical user interface (GUI) built with PySide6 for the powerful `zyddnys/manga-image-translator` backend.

This GUI provides a user-friendly studio environment to manage, configure, and run manga/comic translation tasks without needing to use the command line.

---

### **Table of Contents**
1.  [Core Features](#-core-features)
2.  [Requirements](#-requirements)
3.  [How to Install](#-how-to-install)
4.  [Project Structure](#-project-structure)
5.  [Quick Start Guide](#-quick-start-guide)
6.  [Key Concepts Explained](#-key-concepts-explained)
    *   [The Checkpoint System](#the-checkpoint-system)
    *   [Tasks vs. Configuration](#tasks-vs-configuration)
    *   [Advanced Dictionaries and GPT Configs](#advanced-dictionaries-and-gpt-configs)

---

## ✨ Core Features

*   **Intuitive Job Management:** Add translation jobs via a file dialog or simple drag-and-drop. Reorder, duplicate, or remove jobs with ease.
*   **Checkpoint System:** Save specific settings directly to a job, making it independent of the global panel settings. Queue up multiple jobs with different configurations.
*   **Dedicated Task System:** Go beyond translation with pre-configured tasks for "RAW Output" (text removal), "Image Upscaling", or "Image Colorization".
*   **Smart VRAM Management:** Choose between **High VRAM** (fast), **Low VRAM** (safe for less powerful GPUs), or **Automatic** modes to prevent CUDA "Out of Memory" errors.
*   **Safe & Smart Processing:**
    *   **Resume on Relaunch:** Automatically skips already processed files in the output folder.
    *   **Conflict-Free Outputs:** By default, creates new numbered folders (e.g., `Manga-ENG (1)`) to prevent overwriting previous results.
*   **Full Backend Control:** Access and configure nearly every feature of the backend, from advanced detector settings to text rendering options.
*   **Live Log & History:** Monitor progress in real-time and review completed jobs in the History panel. Re-queue past jobs with a single click.
*   **Secure API & File Management:** Includes helpers for managing API keys (`.env`), translation dictionaries (`--pre/--post-dict`), and custom GPT configurations.

## 📋 Requirements

1.  **Backend Project:** This GUI requires the original `manga-image-translator` project. Please follow the setup instructions from the [official repository](https://github.com/zyddnys/manga-image-translator) first. This will install the core dependencies like `torch`.

2.  **Python Libraries for the GUI:** This user interface requires two additional libraries. Ensure they are installed in your activated virtual environment:
    ```bash
    pip install PySide6 Pillow
    ```
    *   **PySide6:** The core framework for the user interface.
    *   **Pillow:** Used for image processing tasks, such as reading image dimensions for the smart colorization feature.

## 🚀 How to Install

1.  Place the `MangaStudioMain.py` file and the entire `MangaStudio_Data` folder into the **root directory** of your cloned `manga-image-translator` project.
2.  Make sure your Python virtual environment (`venv`) is activated.
3.  Run the application from the root directory:
    ```bash
    python MangaStudioMain.py
    ```

## 📁 Project Structure

The GUI's core files are self-contained within the `MangaStudio_Data` directory to keep the main project folder clean.

```
manga-image-translator/ (Project Root)
|
|-- MangaStudioMain.py       # <-- Main executable to run the UI
|-- MangaStudioMainRun.py    # <-- Run
|
|-- MangaStudio_Data/
|   |-- app/                 # Core application logic and UI window
|   |-- dicts/               # Folder for pre/post-translation dictionaries
|   |-- fonts/               # Place custom .ttf/.otf fonts here
|   |-- gpt_configs/         # Folder for custom GPT/AI configurations
|   |-- profiles/            # Saved user setting presets
|   |-- temp/                # For temporary files (ignored by Git)
|   |-- themes/              # UI theme files
|   |-- tasks.json           # Configuration for the "Tasks" tab
|   |-- ui_map.json          # Maps backend settings to UI widgets
|   |-- README.md            # This file
|
|-- ... (all other original backend folders and files)
```

## 📖 Quick Start Guide

1.  **Add a Job:**
    *   Click `➕ Add Job` to select a folder containing your images.
    *   Or, simply **drag and drop** the folder onto the "Queue" panel.

2.  **Configure Your Job:**
    *   Select the job in the "Queue".
    *   Go to the **`Configuration ⚙️`** tab on the right.
    *   Set your desired `Translator`, `Target Language`, `Output Format`, and other settings.

3.  **Create a Checkpoint:**
    *   This is the most important step. **Right-click** the configured job in the queue.
    *   Select **`✅ Save Settings to Job (Checkpoint)`**.
    *   The job's icon will turn green (🟢), indicating it's ready and its settings are locked in.

4.  **Start Processing:**
    *   Click the **`▶️ START PIPELINE`** button.
    *   You can monitor the detailed progress in the **`Live Log 📊`** tab.

## 💡 Key Concepts Explained

#### The Checkpoint System

A "Checkpoint" is created when you use `Save Settings to Job`. This action "locks" all the settings from the right-hand panel onto that specific job. This is powerful because you can:
*   Configure a job for Japanese-to-English translation.
*   Configure a second job for German-to-Spanish translation with different quality settings.
*   Run them back-to-back in the same pipeline without the settings interfering with each other.

A job **must** have a checkpoint (🟢) to be processed.

#### Tasks vs. Configuration

*   The **`Configuration`** tab is for the main goal: translating manga. It gives you full control over every detail.
*   The **`Tasks 🛠️`** tab is for specific, one-off jobs where you don't need a full translation.
    *   **RAW Output:** Just removes the text from the bubbles.
    *   **Image Upscaling:** Just increases the resolution of the images.
    *   **Image Colorization:** Just colorizes black-and-white images.
    
To use a task, select a job, go to the `Tasks` tab, configure the few settings available, and click the **`Assign...`** button. This will automatically create a checkpoint for that task.

#### Advanced Dictionaries and GPT Configs

Under `Configuration ⚙️` > `General & Translator (Advanced)`, you can find powerful customization options:

*   **Pre/Post-Translation Dictionaries:** These allow you to provide your own `.txt` files to automatically fix common OCR errors or standardize translation terms for consistency. Example files are provided in the `MangaStudio_Data/dicts` folder.
*   **GPT Config File:** This allows you to provide a `.yaml` file to deeply customize the behavior of AI translators (like GPT-4o or Gemini). You can change their "persona," style, and more. An example file is provided in `MangaStudio_Data/gpt_configs`.