# Usage Examples

## Adding Custom Manga Sites

### Example 1: Adding a New Site

Let's say you want to add support for "manga-example.com":

1. Open the userscript in your userscript manager
2. Find this section near the top:

```javascript
// @match        *://mangadex.org/*
// @match        *://*.mangadex.org/*
// ... more sites ...
```

3. Add your site:

```javascript
// @match        *://manga-example.com/*
// @match        *://*.manga-example.com/*
```

4. Save the script (Ctrl+S or Cmd+S)
5. Refresh the manga site

### Example 2: Wildcard Matching

To match ALL sites (use with caution):

```javascript
// @match        *://*/*
```

This will run on every website, which may cause performance issues.

## Customizing Translation Overlays

### Example 1: Change Overlay Background Color

Find the CSS section and modify:

```javascript
.manga-translator-overlay {
    background: rgba(255, 255, 255, 0.95);  // Original (white)
    // Change to:
    background: rgba(255, 255, 200, 0.95);  // Light yellow
    // Or:
    background: rgba(0, 0, 0, 0.8);         // Dark (for light text)
}
```

### Example 2: Change Font Style

```javascript
.manga-translator-overlay {
    font-family: Arial, sans-serif;  // Original
    // Change to:
    font-family: 'Comic Sans MS', cursive;  // Comic style
    // Or:
    font-family: 'Courier New', monospace;  // Monospace
}
```

### Example 3: Change Text Color

```javascript
.manga-translator-overlay {
    color: #333;  // Original (dark gray)
    // Change to:
    color: #000;  // Black
    // Or (with dark background):
    color: #fff;  // White
}
```

### Example 4: Change Border Style

```javascript
.manga-translator-overlay {
    border: 2px solid #667eea;     // Original (purple)
    // Change to:
    border: 2px solid #ff0000;     // Red
    // Or:
    border: 3px dashed #00ff00;    // Green dashed
    // Or:
    border: none;                  // No border
}
```

## Advanced Configuration

### Example 1: Process Smaller Images

Default minimum size is 200x200 pixels. To process smaller images:

Find this line:
```javascript
minImageWidth: 200,
minImageHeight: 200,
```

Change to:
```javascript
minImageWidth: 100,  // Process images 100px wide or larger
minImageHeight: 100,
```

**Warning**: Processing many small images may slow down the page.

### Example 2: Self-Hosted LibreTranslate

If you're running LibreTranslate locally or on your server:

1. Start LibreTranslate:
```bash
# Using Docker
docker run -ti --rm -p 5000:5000 libretranslate/libretranslate

# Or using pip
pip install libretranslate
libretranslate
```

2. In userscript settings:
   - Right-click translator button
   - Change "LibreTranslate URL" to: `http://localhost:5000`
   - Click "Save Settings"

### Example 3: Google Translate Configuration

1. Get API Key from [Google Cloud Console](https://console.cloud.google.com/)

2. Enable required API:
   - Go to "APIs & Services" → "Library"
   - Search for "Cloud Translation API"
   - Click "Enable"

3. Create API Key:
   - Go to "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the key

4. Configure userscript:
   - Right-click translator button
   - Select "Google Translate" from dropdown
   - Paste API key
   - Save settings

## Real-World Scenarios

### Scenario 1: Reading on MangaDex

1. Go to https://mangadex.org
2. Search for your favorite manga
3. Open a chapter
4. Wait for images to load
5. Click the purple "Manga Translator" button if not already active
6. Translations will appear on the images automatically

**Tip**: Enable translation before opening the chapter for immediate processing.

### Scenario 2: Handling Vertical Text (Traditional Manga)

The script automatically detects vertical text based on dimensions:

- If text height > width × 1.5, it's considered vertical
- Vertical text uses CSS writing-mode for proper display
- No manual configuration needed!

### Scenario 3: Multiple Languages

To translate to languages other than English:

1. Right-click translator button
2. Select target language from "Target Language" dropdown:
   - Spanish (es)
   - French (fr)
   - German (de)
   - Chinese (zh)
   - Korean (ko)
3. Save settings
4. Refresh the page

### Scenario 4: Debugging Issues

If translations aren't appearing:

1. Right-click translator button
2. Check "Debug Mode"
3. Save settings
4. Refresh the page
5. Open browser console (F12)
6. Look for messages starting with `[Manga Translator]`

Example output:
```
[Manga Translator] Initializing...
[Manga Translator] Scanning for manga images...
[Manga Translator] Found 10 images to process
[Manga Translator] Processing image: https://...
[Manga Translator] OCR complete. Text blocks found: 5
[Manga Translator] Original text: こんにちは
[Manga Translator] Translated text: Hello
```

## Performance Optimization

### Example 1: Limit Processing on Pages with Many Images

If a page has too many images:

1. Increase minimum image size:
```javascript
minImageWidth: 400,   // Only process larger images
minImageHeight: 400,
```

2. Or manually enable/disable translation:
   - Click translator button to turn off on image-heavy pages
   - Click again to re-enable when viewing manga

### Example 2: Faster Translation

For faster translations:

**Option A**: Self-host LibreTranslate (no network latency)
```bash
docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
```

**Option B**: Use Google Translate API (generally faster, but costs money)
- Configure Google API key in settings

## Troubleshooting Examples

### Problem: "Script not detecting images"

**Solution**: Check if images match minimum size requirements

```javascript
// Enable debug mode and check console
// Look for: "No text detected in image"
// If images are too small, reduce minimum size:
minImageWidth: 150,
minImageHeight: 150,
```

### Problem: "OCR not recognizing Japanese text"

**Possible causes**:
1. Image quality too low (OCR needs clear text)
2. Font too stylized (handwritten-style fonts may fail)
3. Text too small in image

**Solutions**:
- Try a different manga source with higher quality images
- Some manga with artistic fonts won't work well with OCR
- Wait for Tesseract.js to fully load (check console)

### Problem: "Translation API timeout"

**Error**: Request to translation API times out

**Solutions**:

1. Check internet connection
2. Try alternative API:
   ```javascript
   // Switch from LibreTranslate to Google, or vice versa
   ```
3. Use self-hosted LibreTranslate:
   ```bash
   docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
   ```

### Problem: "Rate limited by LibreTranslate"

**Error**: 429 Too Many Requests

**Solutions**:

1. **Wait a few minutes** (public instance has rate limits)

2. **Self-host LibreTranslate**:
   ```bash
   docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
   ```

3. **Use Google Translate API** (configure API key in settings)

4. **Use alternative LibreTranslate instance**:
   - Try: `https://translate.argosopentech.com`

## Creating Custom Presets

### Preset 1: Dark Mode Overlays

```javascript
.manga-translator-overlay {
    background: rgba(0, 0, 0, 0.85);
    color: #ffffff;
    border: 2px solid #ffffff;
}
```

### Preset 2: Comic Book Style

```javascript
.manga-translator-overlay {
    background: rgba(255, 255, 255, 0.95);
    color: #000000;
    border: 3px solid #000000;
    font-family: 'Comic Sans MS', cursive;
    font-weight: bold;
    text-transform: uppercase;
}
```

### Preset 3: Minimal Style

```javascript
.manga-translator-overlay {
    background: rgba(255, 255, 255, 0.7);
    color: #333333;
    border: none;
    box-shadow: none;
    font-family: Arial, sans-serif;
}
```

## Integration Examples

### Example: Using with Browser Extensions

The userscript works alongside other manga reader extensions:

1. **Dark Reader**: Translator overlays adapt to dark mode
2. **Image Max URL**: Works with full-resolution images
3. **uBlock Origin**: Make sure to whitelist translation API domains

### Example: Keyboard Shortcuts

You can add keyboard shortcuts by modifying the script:

```javascript
// Add this in the init() function
document.addEventListener('keydown', (e) => {
    // Press 'T' to toggle translator
    if (e.key === 't' || e.key === 'T') {
        toggleTranslator();
    }
    // Press 'S' to open settings
    if (e.key === 's' || e.key === 'S') {
        toggleSettings();
    }
});
```

## API Examples

### Example 1: LibreTranslate API Request

Manual API request format:

```bash
curl -X POST "https://libretranslate.com/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "こんにちは",
    "source": "ja",
    "target": "en",
    "format": "text"
  }'
```

Response:
```json
{
  "translatedText": "Hello"
}
```

### Example 2: Google Translate API Request

```bash
curl -X POST \
  "https://translation.googleapis.com/language/translate/v2?key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "こんにちは",
    "source": "ja",
    "target": "en"
  }'
```

Response:
```json
{
  "data": {
    "translations": [
      {
        "translatedText": "Hello"
      }
    ]
  }
}
```

---

For more examples and support, visit the [main documentation](./README.md) or join our [Discord](https://discord.gg/Ak8APNy4vb).
