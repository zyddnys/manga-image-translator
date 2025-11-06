# Quick Installation Guide

## Step 1: Install a Userscript Manager

Choose one of these browser extensions:

### Chrome / Edge / Brave
- [Tampermonkey](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo) (Recommended)
- [Violentmonkey](https://chrome.google.com/webstore/detail/violentmonkey/jinjaccalgkegednnccohejagnlnfdag)

### Firefox
- [Tampermonkey](https://addons.mozilla.org/en-US/firefox/addon/tampermonkey/) (Recommended)
- [Greasemonkey](https://addons.mozilla.org/en-US/firefox/addon/greasemonkey/)
- [Violentmonkey](https://addons.mozilla.org/en-US/firefox/addon/violentmonkey/)

### Safari
- [Tampermonkey](https://apps.apple.com/us/app/tampermonkey/id1482490089)

## Step 2: Install the Userscript

### Method 1: Direct Installation (Recommended)

1. Click on your userscript manager icon in the browser toolbar
2. Select "Create a new script" or "Dashboard" → "+" button
3. Delete any default code in the editor
4. Open `manga-translator.user.js` from this directory
5. Copy all the code
6. Paste it into the userscript editor
7. Press Ctrl+S (Windows/Linux) or Cmd+S (Mac) to save
8. You should see "Script installed successfully" or similar message

### Method 2: File Installation

1. Open your userscript manager dashboard
2. Look for "Import" or "Install from file" option
3. Select the `manga-translator.user.js` file
4. Confirm installation

## Step 3: Test the Script

1. Navigate to a manga reading site (e.g., [MangaDex](https://mangadex.org))
2. Open any manga chapter
3. Wait for the page to load
4. Look for a **purple button** labeled "Manga Translator" in the bottom-right corner
5. Click the button to enable/disable translation
6. Right-click the button to open settings

## Step 4: Configure (Optional)

The script works out of the box with **LibreTranslate** (free, no setup required).

### For Better Performance: Use Your Own API

#### Option 1: Host LibreTranslate Locally (Free)
```bash
# Using Docker
docker run -ti --rm -p 5000:5000 libretranslate/libretranslate

# Then in userscript settings, set URL to: http://localhost:5000
```

#### Option 2: Use Google Translate API (Paid)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "Cloud Translation API"
4. Create an API key under "Credentials"
5. Right-click the translator button → Enter API key → Save

## Troubleshooting

### "Script not working"
- Refresh the page after installation
- Check if the script is enabled in your userscript manager
- Look at browser console (F12) for errors

### "No button appears"
- Make sure you're on a supported manga site
- Check if the script is running (userscript manager icon should show "1" or the script name)
- Try manually adding the site to `@match` rules

### "No translations appearing"
- Wait for OCR processing (first load takes longer)
- Make sure images are loaded completely
- Check browser console for API errors
- Try switching between LibreTranslate and Google Translate

## Supported Sites

Works on:
- MangaDex, Mangakakalot, Manganato, MangaPark
- Comick, Bato.to, MangaReader, MangaSee123
- Webtoons, Dynasty Scans
- And more...

## Next Steps

- Read the full [README.md](./README.md) for detailed documentation
- Customize settings by right-clicking the translator button
- Report issues or request features on GitHub

## Quick Tips

1. **First load is slow**: Tesseract.js needs to download (~5MB) on first use
2. **Rate limits**: Public LibreTranslate may rate-limit. Consider self-hosting or Google API
3. **Better accuracy**: High-resolution images work better with OCR
4. **Vertical text**: Script automatically detects and formats vertical manga text
5. **Privacy**: OCR happens locally, but translation requires internet connection

---

**Need help?** Join the [Discord server](https://discord.gg/Ak8APNy4vb) or check the [full documentation](./README.md).
