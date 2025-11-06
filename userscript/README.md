# Manga Image Translator Userscript

A powerful browser userscript that automatically detects manga/comic images on web pages, extracts Japanese text using OCR, translates it to English (or other languages), and overlays the translations on the images.

## Features

- **Automatic Image Detection**: Scans manga reading sites for comic images
- **OCR Text Extraction**: Uses Tesseract.js to extract Japanese text from images
- **Real-time Translation**: Translates extracted text using LibreTranslate (free) or Google Translate API
- **Smart Text Overlay**: Overlays translated text on images with proper positioning
- **Vertical Text Support**: Detects and properly displays vertical text (common in manga)
- **Toggle On/Off**: Easily enable or disable translation with a single click
- **Multiple Translation APIs**: Choose between LibreTranslate (free, no API key) or Google Translate
- **Customizable Settings**: Configure API endpoints, target language, and more
- **Works with Popular Manga Sites**: Pre-configured for MangaDex, Mangakakalot, Manganato, and more

## Supported Sites

The userscript is pre-configured to work with:

- MangaDex (mangadex.org)
- Mangakakalot (mangakakalot.com)
- Manganato (manganato.com)
- MangaPark (mangapark.net)
- Comick (comick.io)
- Bato.to (bato.to)
- MangaReader (mangareader.to)
- Webtoons (webtoons.com)
- Dynasty Scans (dynasty-scans.com)
- MangaSee123 (mangasee123.com)

**Note**: The userscript should work on any site, but you may need to add additional `@match` patterns for sites not listed above.

## Installation

### Prerequisites

You need a userscript manager browser extension:

- **[Tampermonkey](https://www.tampermonkey.net/)** (Recommended - Chrome, Firefox, Safari, Edge, Opera)
- **[Greasemonkey](https://www.greasespot.net/)** (Firefox)
- **[Violentmonkey](https://violentmonkey.github.io/)** (Chrome, Firefox, Edge, Opera)

### Install Steps

1. Install one of the userscript managers above
2. Click on the userscript manager icon in your browser toolbar
3. Select "Create a new script" or "Add new script"
4. Copy the contents of `manga-translator.user.js` from this directory
5. Paste the code into the editor
6. Save the script (usually Ctrl+S or Cmd+S)
7. Navigate to a supported manga reading site
8. The script should automatically activate

### Alternative Installation

If the userscript is hosted online (e.g., on Greasyfork):

1. Click the installation link
2. Your userscript manager will prompt you to install
3. Click "Install" or "Confirm"

## Usage

### Basic Usage

1. **Navigate to a manga reading site** (e.g., MangaDex)
2. **Open a manga chapter**
3. **Wait for images to load** - The script will automatically start processing
4. **Translated text will appear** as overlays on the manga images

### Toggle Translation

- Click the **"Manga Translator"** button (bottom-right corner) to enable/disable translation
- Green indicator = Active
- Orange indicator = Processing
- Red indicator = Disabled

### Access Settings

- **Right-click** the "Manga Translator" button to open settings
- Configure:
  - **Translation API**: Choose between LibreTranslate (free) or Google Translate
  - **API URL/Key**: Set API endpoint or authentication key
  - **Target Language**: Select your preferred translation language
  - **Debug Mode**: Enable detailed console logging

### Settings Configuration

#### Using LibreTranslate (Default - Free)

LibreTranslate is a free, open-source translation API that requires no API key:

1. Right-click the translator button
2. Select "LibreTranslate (Free)" from the API dropdown
3. (Optional) Change the LibreTranslate URL if using a custom instance
4. Click "Save Settings"

**Default URL**: `https://libretranslate.com`

**Custom Instances**: You can host your own LibreTranslate instance or use community instances:
- https://libretranslate.com (Official, rate-limited)
- https://translate.argosopentech.com (Alternative)

#### Using Google Translate API

Google Translate requires an API key:

1. Get a Google Cloud API key:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable the "Cloud Translation API"
   - Create credentials (API Key)
   - Copy the API key

2. Configure in userscript:
   - Right-click the translator button
   - Select "Google Translate" from the API dropdown
   - Paste your API key in the "Google API Key" field
   - Click "Save Settings"

**Note**: Google Translate API is a paid service after the free tier.

## How It Works

1. **Image Detection**: Script scans the page for images larger than 200x200 pixels
2. **OCR Processing**: Tesseract.js performs optical character recognition on detected manga images
3. **Text Extraction**: Japanese text is extracted from the image with bounding box coordinates
4. **Translation**: Extracted text is sent to the configured translation API (LibreTranslate or Google)
5. **Overlay Creation**: Translated text is positioned over the original text with proper styling
6. **Dynamic Loading**: Script monitors the page for new images (supports infinite scroll)

## Configuration Options

### In-Script Configuration

You can modify these constants at the top of the script:

```javascript
const CONFIG = {
    enabled: true,                  // Enable on page load
    sourceLang: 'jpn',             // Source language (Japanese)
    targetLang: 'en',              // Target language (English)
    translationAPI: 'libretranslate', // API to use
    minImageWidth: 200,            // Minimum image width to process
    minImageHeight: 200,           // Minimum image height to process
    debugMode: false               // Enable debug logging
};
```

### Supported Target Languages

- **English** (en)
- **Spanish** (es)
- **French** (fr)
- **German** (de)
- **Chinese** (zh)
- **Korean** (ko)

More languages may be available depending on your translation API.

## Troubleshooting

### No translations appearing

1. Check if the script is enabled (green status indicator)
2. Open browser console (F12) and look for errors
3. Verify images are large enough (minimum 200x200 pixels)
4. Enable debug mode in settings to see detailed logs

### Translation API errors

1. Check your internet connection
2. Verify API endpoint URL is correct
3. If using Google Translate, verify your API key is valid
4. Try switching to LibreTranslate (no API key required)

### OCR not detecting text

1. Tesseract.js requires clear, readable text in images
2. Very low-resolution images may not work well
3. Heavily stylized fonts may reduce accuracy
4. Enable debug mode to see OCR results

### Performance issues

1. Processing many images at once may slow down the page
2. Script processes images one at a time to avoid overload
3. Close other tabs or disable on pages with many images
4. Consider increasing `minImageWidth` and `minImageHeight` to process fewer images

## Advanced Usage

### Adding Custom Sites

To add support for additional manga sites:

1. Open the userscript in your userscript manager
2. Find the `@match` section at the top
3. Add a new line with the site pattern:
   ```javascript
   // @match        *://yoursite.com/*
   ```
4. Save the script

### Using Custom LibreTranslate Instance

If you want to use your own LibreTranslate instance:

1. Deploy LibreTranslate (see [LibreTranslate GitHub](https://github.com/LibreTranslate/LibreTranslate))
2. Right-click the translator button
3. Enter your instance URL in the "LibreTranslate URL" field
4. Save settings

### Styling Overlays

You can customize the appearance of translation overlays by modifying the CSS in the script:

```javascript
.manga-translator-overlay {
    background: rgba(255, 255, 255, 0.95); // Background color
    border: 2px solid #667eea;              // Border
    padding: 8px;                           // Padding
    font-size: 14px;                        // Font size
    // ... more styles
}
```

## Performance Considerations

- **OCR Processing**: Tesseract.js runs in a Web Worker, so it won't block the UI
- **Rate Limiting**: LibreTranslate has rate limits on the public instance
- **Image Processing**: Large images may take longer to process
- **Memory Usage**: Processing many images increases memory usage

## Privacy & Data

- **LibreTranslate**: Text is sent to the LibreTranslate server for translation
- **Google Translate**: Text is sent to Google's servers (requires API key)
- **Tesseract.js**: OCR processing happens locally in your browser
- **No Data Storage**: The script doesn't store or collect any data

## Limitations

- **OCR Accuracy**: Tesseract.js accuracy varies based on image quality and font style
- **Translation Quality**: Depends on the translation API used
- **Language Support**: Currently optimized for Japanese to English
- **Complex Layouts**: Very complex manga layouts may have positioning issues
- **Text Direction**: Vertical text detection is basic and may not work perfectly

## Contributing

Contributions are welcome! Areas for improvement:

- Better text positioning algorithms
- Support for more languages
- Improved OCR accuracy
- Additional translation API integrations
- Better handling of complex manga layouts
- Caching translated text to avoid re-translation

## License

This userscript is part of the manga-image-translator project. See the main project LICENSE file.

## Credits

- **Tesseract.js**: OCR engine (https://tesseract.projectnaptha.com/)
- **LibreTranslate**: Free translation API (https://libretranslate.com/)
- **manga-image-translator**: Main project (https://github.com/zyddnys/manga-image-translator)

## Support

For issues, questions, or feature requests:

1. Check the Troubleshooting section above
2. Enable debug mode and check browser console
3. Open an issue on the main project GitHub
4. Join the Discord server: https://discord.gg/Ak8APNy4vb

## Changelog

### Version 1.0.0 (2025-11-06)

- Initial release
- Tesseract.js integration for OCR
- LibreTranslate and Google Translate API support
- Automatic image detection and processing
- Text overlay with positioning
- Toggle button and settings panel
- Support for major manga reading sites
- Vertical text detection
- Progress indicator
- Debug mode
