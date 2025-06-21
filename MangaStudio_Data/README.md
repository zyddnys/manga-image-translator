# Manga Translation Studio - GUI Module

## 1. About This Project

This GUI is the result of a "vibe coding" collaboration:

-   **Ideas, Direction & Testing:** [Kostraw](https://github.com/Kostraw)
-   **Code Implementation:** Gemini AI

This project provides a user-friendly graphical interface (GUI) for the incredible command-line tool, **`manga-image-translator`**, created by **zyddnys**.

**Original Project Link:** [https://github.com/zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)

[A screenshot of the application interface would go here.]

---

## 2. Motivation ("Why This GUI?")

I love the power and flexibility of the original `manga-image-translator` tool. However, as someone who prefers a visual workflow, I wanted to create an interface to harness its full potential without using the command line.

This project started as a personal tool to make my own life easier. As it grew and became more powerful, I decided to polish it and share it with the community, hoping it might be useful to others.

---

## 3. Installation & Usage

1.  **Prerequisites:** Ensure you have a working Python environment and the `manga-image-translator` tool itself installed and functional.
2.  **Install GUI Libraries:** Open a terminal and run the following command to install the necessary libraries for the interface:
    ```bash
    pip install customtkinter Pillow tkinterdnd2 CTkToolTip
    ```
3.  **Run the Application:** Navigate to the directory containing the script and run:
    ```bash
    python MangaStudioV6.py
    ```
4.  **Data Folder:** On first launch, the application will create a `MangaStudio_Data` folder next to the script. All user settings, profiles, and temporary files will be stored here to keep the main directory clean.

---

## 4. Key Features

-   **Full Control:** Access and manage all settings provided by the `manga-image-translator` tool through an intuitive tabbed interface.
-   **Job Queue:** Drag and drop multiple folders to process them sequentially.
-   **Preset Manager:** Save, load, and manage your favorite setting configurations for different tasks.
-   **Special Tasks Workshop:** A dedicated section to perform standalone tasks like **RAW Output (Text Cleaning)**, **Upscaling**, or **Colorization** with their own independent, fine-tuned settings.
-   **Model Manager:** View, download, and delete required AI models from within the application.
-   **Advanced Debug Mode:** An optional mode for power users to save all intermediate processing steps for detailed troubleshooting.
-   **Centralized File Structure:** Keeps the root directory clean by storing all GUI-related data in the `MangaStudio_Data` folder.

---

## 5. Current Status & Disclaimer

This GUI has been developed and tested to be stable for its main intended workflows. However, as a complex project built on top of another tool, there may be undiscovered bugs or edge cases in specific, untested scenarios. It is provided "as-is".

---

## 6. A Note on Support & Future Development

This GUI was a passion project. **I am not the official maintainer of this GUI module.** My goal was to build it, share it, and then step back.

Therefore, please **do not contact me directly** for bug reports, feature requests, or support.

-   **For Bugs or Feature Ideas:** Please use the **"Issues"** tab on the main `zyddnys/manga-image-translator` GitHub repository. This allows the original author and the entire community to track and manage contributions centrally.
-   **For Help:** Please consult the community or wait for the original project owner to address the issue.

Feel free to modify, improve, or take over the development of this GUI.

---

## 7. Future Development Ideas

If someone wishes to continue developing this GUI, here are some potential next steps:
-   **API Key Management:** A dedicated section to securely store and use API keys for services like DeepL, OpenAI, etc. A placeholder frame for this already exists in the "Extra Settings" tab.
-   **"Ultra Quality" Colorize Mode:** Implementing a two-pass system (AI upscale + standard resize) for the colorize task to potentially achieve higher quality results. A developer note for this exists in the code.

---

## 8. License

This GUI module, as a derivative work, falls under the same license as the original `manga-image-translator` project.

**License:** [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

---
*This GUI was developed in collaboration with an AI assistant.*