# ===============================================================
# Manga Translation Studio - Main Entry Point
#
# Author: User & Gemini Collaboration
#
# Description: This file is the main entry point for the
#              application. It configures the system path
#              and launches the main UI window.
# ===============================================================

import os
import sys
from PySide6.QtWidgets import QApplication, QMessageBox

# --- Path Configuration ---
# This is crucial for the modular structure to work correctly.
# It ensures that Python can find the 'app' module inside the 'MangaStudio_Data' directory.

# Get the absolute path of the directory where this script is located (the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the directory containing our application source code
APP_SOURCE_DIR = os.path.join(BASE_DIR, "MangaStudio_Data")

# Add the source directory to the Python system path.
# This allows us to use `from app.ui.main_window import ...`
sys.path.insert(0, APP_SOURCE_DIR)


# --- Application Launch ---
try:
    # Now that the path is configured, we can import the main application class.
    # We will modify main_window.py to contain a PySide class with the same name.
    from app.ui.main_window import TranslatorStudioApp
except ImportError as e:
    # A QApplication instance is needed to show a QMessageBox.
    # We create a dummy app here just for the error message.
    error_app = QApplication(sys.argv)
    QMessageBox.critical(
        None,
        "Fatal Import Error",
        "Could not import the main application class. "
        "Please check that the following structure is correct:\n\n"
        "MangaStudio_Data -> app -> ui -> main_window.py\n\n"
        f"Error: {e}"
    )
    sys.exit(1)


if __name__ == "__main__":
    try:
        # 1. Create the PySide Application instance
        app = QApplication(sys.argv)

        # 2. Create an instance of our main window
        #    (The TranslatorStudioApp class will be a PySide window now)
        main_window = TranslatorStudioApp()

        # 3. Show the window
        main_window.show()

        # 4. Start the application's event loop
        sys.exit(app.exec())

    except Exception as e:
        import traceback
        
        error_title = "Critical Application Error"
        error_message = (
            "The application encountered a critical error and had to shut down.\n\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Details: {e}\n\n"
            "Please check the console output for the full traceback."
        )
        
        print(f"---! {error_title.upper()} !---")
        traceback.print_exc()
        print("---------------------------------")
        
        # We still need a QApplication to show the error message.
        # Create one if it doesn't exist yet.
        error_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, error_title, error_message)
        sys.exit(1)