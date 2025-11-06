// ==UserScript==
// @name         Manga Image Translator
// @namespace    http://tampermonkey.net/
// @version      1.0.0
// @description  Detect manga/comic images, extract text using OCR, translate Japanese to English, and overlay translations
// @author       manga-image-translator
// @match        *://mangadex.org/*
// @match        *://*.mangadex.org/*
// @match        *://mangakakalot.com/*
// @match        *://*.mangakakalot.com/*
// @match        *://manganato.com/*
// @match        *://*.manganato.com/*
// @match        *://mangapark.net/*
// @match        *://*.mangapark.net/*
// @match        *://comick.io/*
// @match        *://*.comick.io/*
// @match        *://bato.to/*
// @match        *://*.bato.to/*
// @match        *://mangareader.to/*
// @match        *://*.mangareader.to/*
// @match        *://webtoons.com/*
// @match        *://*.webtoons.com/*
// @match        *://dynasty-scans.com/*
// @match        *://*.dynasty-scans.com/*
// @match        *://mangasee123.com/*
// @match        *://*.mangasee123.com/*
// @icon         data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0yMSAySDNjLTEuMSAwLTIgLjktMiAydjE2YzAgMS4xLjkgMiAyIDJoMThjMS4xIDAgMi0uOSAyLTJWNGMwLTEuMS0uOS0yLTItMnptMCAxOEgzVjRoMTh2MTZ6TTcgN2g1djJIN3ptMCA0aDEwdi0xSDd6bTAgM2gxMHYtMUg3em0wIDNoNXYtMUg3eiIvPjwvc3ZnPg==
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// @grant        GM_getValue
// @grant        GM_setValue
// @require      https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js
// @run-at       document-idle
// @connect      libretranslate.com
// @connect      translation.googleapis.com
// @connect      *
// ==/UserScript==

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        enabled: GM_getValue('translatorEnabled', true),
        sourceLang: 'jpn', // Japanese for Tesseract
        targetLang: 'en',
        translationAPI: GM_getValue('translationAPI', 'libretranslate'), // 'libretranslate' or 'google'
        libretranslateURL: GM_getValue('libretranslateURL', 'https://libretranslate.com'),
        googleAPIKey: GM_getValue('googleAPIKey', ''), // User needs to set this
        minImageWidth: 200,
        minImageHeight: 200,
        processedClass: 'manga-translator-processed',
        overlayClass: 'manga-translator-overlay',
        debugMode: GM_getValue('debugMode', false)
    };

    // State
    let tesseractWorker = null;
    let isProcessing = false;
    let processQueue = [];

    // Add CSS styles
    GM_addStyle(`
        .manga-translator-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 99999;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .manga-translator-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }

        .manga-translator-toggle.disabled {
            background: linear-gradient(135deg, #757575 0%, #616161 100%);
        }

        .manga-translator-status {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4caf50;
            display: inline-block;
        }

        .manga-translator-status.processing {
            background: #ff9800;
            animation: pulse 1.5s ease-in-out infinite;
        }

        .manga-translator-status.disabled {
            background: #f44336;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .manga-translator-overlay {
            position: absolute;
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid #667eea;
            border-radius: 4px;
            padding: 8px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            line-height: 1.4;
            color: #333;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            max-width: 300px;
            word-wrap: break-word;
            pointer-events: none;
        }

        .manga-translator-overlay.vertical {
            writing-mode: vertical-rl;
            text-orientation: upright;
        }

        .manga-translator-settings {
            position: fixed;
            bottom: 80px;
            right: 20px;
            z-index: 99998;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            display: none;
            min-width: 300px;
        }

        .manga-translator-settings.show {
            display: block;
        }

        .manga-translator-settings h3 {
            margin: 0 0 15px 0;
            font-size: 16px;
            color: #333;
        }

        .manga-translator-settings label {
            display: block;
            margin-bottom: 10px;
            font-size: 13px;
            color: #666;
        }

        .manga-translator-settings input,
        .manga-translator-settings select {
            width: 100%;
            padding: 8px;
            margin-top: 4px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }

        .manga-translator-settings button {
            width: 100%;
            padding: 10px;
            margin-top: 15px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.3s;
        }

        .manga-translator-settings button:hover {
            background: #5568d3;
        }

        .manga-translator-progress {
            position: fixed;
            bottom: 80px;
            right: 20px;
            z-index: 99997;
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            display: none;
            min-width: 250px;
        }

        .manga-translator-progress.show {
            display: block;
        }

        .manga-translator-progress-text {
            font-size: 13px;
            color: #666;
            margin-bottom: 8px;
        }

        .manga-translator-progress-bar {
            width: 100%;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
        }

        .manga-translator-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            width: 0%;
        }

        img.${CONFIG.processedClass} {
            cursor: crosshair;
        }
    `);

    // Initialize Tesseract worker
    async function initTesseract() {
        if (tesseractWorker) return tesseractWorker;

        console.log('[Manga Translator] Initializing Tesseract...');
        tesseractWorker = await Tesseract.createWorker('jpn', 1, {
            logger: m => {
                if (CONFIG.debugMode) console.log('[Tesseract]', m);
            }
        });

        console.log('[Manga Translator] Tesseract initialized');
        return tesseractWorker;
    }

    // Translate text using configured API
    async function translateText(text, sourceLang = 'ja', targetLang = 'en') {
        if (!text || text.trim().length === 0) return '';

        try {
            if (CONFIG.translationAPI === 'google' && CONFIG.googleAPIKey) {
                return await translateWithGoogle(text, sourceLang, targetLang);
            } else {
                return await translateWithLibreTranslate(text, sourceLang, targetLang);
            }
        } catch (error) {
            console.error('[Manga Translator] Translation error:', error);
            return text; // Return original text if translation fails
        }
    }

    // Translate using LibreTranslate (free, no API key needed)
    function translateWithLibreTranslate(text, sourceLang, targetLang) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method: 'POST',
                url: `${CONFIG.libretranslateURL}/translate`,
                headers: {
                    'Content-Type': 'application/json'
                },
                data: JSON.stringify({
                    q: text,
                    source: sourceLang,
                    target: targetLang,
                    format: 'text'
                }),
                onload: function(response) {
                    try {
                        const data = JSON.parse(response.responseText);
                        if (data.translatedText) {
                            resolve(data.translatedText);
                        } else {
                            reject(new Error('No translation returned'));
                        }
                    } catch (error) {
                        reject(error);
                    }
                },
                onerror: function(error) {
                    reject(error);
                }
            });
        });
    }

    // Translate using Google Translate API (requires API key)
    function translateWithGoogle(text, sourceLang, targetLang) {
        return new Promise((resolve, reject) => {
            if (!CONFIG.googleAPIKey) {
                reject(new Error('Google API key not configured'));
                return;
            }

            GM_xmlhttpRequest({
                method: 'POST',
                url: `https://translation.googleapis.com/language/translate/v2?key=${CONFIG.googleAPIKey}`,
                headers: {
                    'Content-Type': 'application/json'
                },
                data: JSON.stringify({
                    q: text,
                    source: sourceLang,
                    target: targetLang,
                    format: 'text'
                }),
                onload: function(response) {
                    try {
                        const data = JSON.parse(response.responseText);
                        if (data.data && data.data.translations && data.data.translations[0]) {
                            resolve(data.data.translations[0].translatedText);
                        } else {
                            reject(new Error('No translation returned'));
                        }
                    } catch (error) {
                        reject(error);
                    }
                },
                onerror: function(error) {
                    reject(error);
                }
            });
        });
    }

    // Process image with OCR and translation
    async function processImage(img) {
        if (!CONFIG.enabled) return;
        if (img.classList.contains(CONFIG.processedClass)) return;
        if (img.width < CONFIG.minImageWidth || img.height < CONFIG.minImageHeight) return;

        img.classList.add(CONFIG.processedClass);

        try {
            updateStatus('processing');
            console.log('[Manga Translator] Processing image:', img.src);

            // Initialize Tesseract if needed
            const worker = await initTesseract();

            // Perform OCR
            updateProgress('Performing OCR...', 30);
            const { data } = await worker.recognize(img);

            console.log('[Manga Translator] OCR complete. Text blocks found:', data.blocks.length);

            if (data.blocks.length === 0) {
                console.log('[Manga Translator] No text detected in image');
                updateStatus('idle');
                return;
            }

            // Create overlay container if it doesn't exist
            let overlayContainer = img.parentElement.querySelector('.manga-translator-overlay-container');
            if (!overlayContainer) {
                overlayContainer = document.createElement('div');
                overlayContainer.className = 'manga-translator-overlay-container';
                overlayContainer.style.position = 'relative';
                overlayContainer.style.display = 'inline-block';

                // Wrap image in container
                img.parentNode.insertBefore(overlayContainer, img);
                overlayContainer.appendChild(img);
            }

            // Remove old overlays
            const oldOverlays = overlayContainer.querySelectorAll(`.${CONFIG.overlayClass}`);
            oldOverlays.forEach(overlay => overlay.remove());

            // Process each text block
            let processedBlocks = 0;
            for (const block of data.blocks) {
                if (!block.text || block.text.trim().length === 0) continue;

                const originalText = block.text.trim();
                console.log('[Manga Translator] Original text:', originalText);

                updateProgress(`Translating... (${processedBlocks + 1}/${data.blocks.length})`, 40 + (processedBlocks / data.blocks.length * 50));

                // Translate text
                const translatedText = await translateText(originalText, 'ja', CONFIG.targetLang);
                console.log('[Manga Translator] Translated text:', translatedText);

                if (translatedText && translatedText !== originalText) {
                    // Create overlay element
                    createTextOverlay(overlayContainer, img, block.bbox, translatedText);
                }

                processedBlocks++;
            }

            updateProgress('Complete!', 100);
            setTimeout(() => hideProgress(), 1000);
            updateStatus('idle');

        } catch (error) {
            console.error('[Manga Translator] Error processing image:', error);
            updateStatus('idle');
            hideProgress();
        }
    }

    // Create text overlay on image
    function createTextOverlay(container, img, bbox, text) {
        const overlay = document.createElement('div');
        overlay.className = CONFIG.overlayClass;
        overlay.textContent = text;

        // Calculate position relative to image
        const imgRect = img.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();

        const scaleX = img.naturalWidth > 0 ? img.width / img.naturalWidth : 1;
        const scaleY = img.naturalHeight > 0 ? img.height / img.naturalHeight : 1;

        const left = bbox.x0 * scaleX;
        const top = bbox.y0 * scaleY;
        const width = (bbox.x1 - bbox.x0) * scaleX;
        const height = (bbox.y1 - bbox.y0) * scaleY;

        // Position overlay
        overlay.style.left = `${left}px`;
        overlay.style.top = `${top}px`;
        overlay.style.minWidth = `${width}px`;
        overlay.style.minHeight = `${height}px`;

        // Detect if text should be vertical (common in manga)
        if (height > width * 1.5) {
            overlay.classList.add('vertical');
        }

        container.appendChild(overlay);
    }

    // Find manga images on the page
    function findMangaImages() {
        const images = document.querySelectorAll('img');
        const mangaImages = [];

        for (const img of images) {
            if (img.classList.contains(CONFIG.processedClass)) continue;

            // Check if image is large enough (likely manga panel)
            if (img.naturalWidth >= CONFIG.minImageWidth && img.naturalHeight >= CONFIG.minImageHeight) {
                mangaImages.push(img);
            } else if (!img.complete) {
                // Wait for image to load
                img.addEventListener('load', () => {
                    if (img.naturalWidth >= CONFIG.minImageWidth && img.naturalHeight >= CONFIG.minImageHeight) {
                        processQueue.push(img);
                        processNextInQueue();
                    }
                }, { once: true });
            }
        }

        return mangaImages;
    }

    // Process images in queue
    async function processNextInQueue() {
        if (isProcessing || processQueue.length === 0 || !CONFIG.enabled) return;

        isProcessing = true;
        const img = processQueue.shift();

        await processImage(img);

        isProcessing = false;

        // Process next image after a short delay
        if (processQueue.length > 0) {
            setTimeout(processNextInQueue, 500);
        }
    }

    // Update status indicator
    function updateStatus(status) {
        const statusDot = document.querySelector('.manga-translator-status');
        if (!statusDot) return;

        statusDot.className = 'manga-translator-status';
        if (status === 'processing') {
            statusDot.classList.add('processing');
        } else if (status === 'disabled') {
            statusDot.classList.add('disabled');
        }
    }

    // Update progress bar
    function updateProgress(text, percent) {
        const progressEl = document.querySelector('.manga-translator-progress');
        const progressText = document.querySelector('.manga-translator-progress-text');
        const progressFill = document.querySelector('.manga-translator-progress-fill');

        if (progressEl && progressText && progressFill) {
            progressEl.classList.add('show');
            progressText.textContent = text;
            progressFill.style.width = `${percent}%`;
        }
    }

    // Hide progress bar
    function hideProgress() {
        const progressEl = document.querySelector('.manga-translator-progress');
        if (progressEl) {
            progressEl.classList.remove('show');
        }
    }

    // Create UI controls
    function createUI() {
        // Toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = `manga-translator-toggle ${!CONFIG.enabled ? 'disabled' : ''}`;
        toggleBtn.innerHTML = `
            <span class="manga-translator-status ${!CONFIG.enabled ? 'disabled' : ''}"></span>
            <span>Manga Translator</span>
        `;
        toggleBtn.addEventListener('click', toggleTranslator);
        toggleBtn.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            toggleSettings();
        });
        document.body.appendChild(toggleBtn);

        // Settings panel
        const settingsPanel = document.createElement('div');
        settingsPanel.className = 'manga-translator-settings';
        settingsPanel.innerHTML = `
            <h3>Manga Translator Settings</h3>
            <label>
                Translation API:
                <select id="translator-api">
                    <option value="libretranslate" ${CONFIG.translationAPI === 'libretranslate' ? 'selected' : ''}>LibreTranslate (Free)</option>
                    <option value="google" ${CONFIG.translationAPI === 'google' ? 'selected' : ''}>Google Translate</option>
                </select>
            </label>
            <label id="libretranslate-url-container">
                LibreTranslate URL:
                <input type="text" id="libretranslate-url" value="${CONFIG.libretranslateURL}" placeholder="https://libretranslate.com">
            </label>
            <label id="google-api-key-container" style="display: none;">
                Google API Key:
                <input type="text" id="google-api-key" value="${CONFIG.googleAPIKey}" placeholder="Enter your API key">
            </label>
            <label>
                Target Language:
                <select id="target-lang">
                    <option value="en" ${CONFIG.targetLang === 'en' ? 'selected' : ''}>English</option>
                    <option value="es" ${CONFIG.targetLang === 'es' ? 'selected' : ''}>Spanish</option>
                    <option value="fr" ${CONFIG.targetLang === 'fr' ? 'selected' : ''}>French</option>
                    <option value="de" ${CONFIG.targetLang === 'de' ? 'selected' : ''}>German</option>
                    <option value="zh" ${CONFIG.targetLang === 'zh' ? 'selected' : ''}>Chinese</option>
                    <option value="ko" ${CONFIG.targetLang === 'ko' ? 'selected' : ''}>Korean</option>
                </select>
            </label>
            <label>
                <input type="checkbox" id="debug-mode" ${CONFIG.debugMode ? 'checked' : ''}>
                Debug Mode
            </label>
            <button id="save-settings">Save Settings</button>
        `;
        document.body.appendChild(settingsPanel);

        // Handle API selection change
        const apiSelect = document.getElementById('translator-api');
        const libretranslateContainer = document.getElementById('libretranslate-url-container');
        const googleApiKeyContainer = document.getElementById('google-api-key-container');

        apiSelect.addEventListener('change', () => {
            if (apiSelect.value === 'google') {
                libretranslateContainer.style.display = 'none';
                googleApiKeyContainer.style.display = 'block';
            } else {
                libretranslateContainer.style.display = 'block';
                googleApiKeyContainer.style.display = 'none';
            }
        });

        // Save settings
        document.getElementById('save-settings').addEventListener('click', () => {
            CONFIG.translationAPI = apiSelect.value;
            CONFIG.libretranslateURL = document.getElementById('libretranslate-url').value;
            CONFIG.googleAPIKey = document.getElementById('google-api-key').value;
            CONFIG.targetLang = document.getElementById('target-lang').value;
            CONFIG.debugMode = document.getElementById('debug-mode').checked;

            GM_setValue('translationAPI', CONFIG.translationAPI);
            GM_setValue('libretranslateURL', CONFIG.libretranslateURL);
            GM_setValue('googleAPIKey', CONFIG.googleAPIKey);
            GM_setValue('targetLang', CONFIG.targetLang);
            GM_setValue('debugMode', CONFIG.debugMode);

            alert('Settings saved! Refresh the page for changes to take effect.');
            toggleSettings();
        });

        // Progress indicator
        const progressPanel = document.createElement('div');
        progressPanel.className = 'manga-translator-progress';
        progressPanel.innerHTML = `
            <div class="manga-translator-progress-text">Processing...</div>
            <div class="manga-translator-progress-bar">
                <div class="manga-translator-progress-fill"></div>
            </div>
        `;
        document.body.appendChild(progressPanel);
    }

    // Toggle translator on/off
    function toggleTranslator() {
        CONFIG.enabled = !CONFIG.enabled;
        GM_setValue('translatorEnabled', CONFIG.enabled);

        const toggleBtn = document.querySelector('.manga-translator-toggle');
        const statusDot = document.querySelector('.manga-translator-status');

        if (CONFIG.enabled) {
            toggleBtn.classList.remove('disabled');
            statusDot.classList.remove('disabled');
            scanAndProcess();
        } else {
            toggleBtn.classList.add('disabled');
            statusDot.classList.add('disabled');

            // Remove all overlays
            const overlays = document.querySelectorAll(`.${CONFIG.overlayClass}`);
            overlays.forEach(overlay => overlay.remove());

            // Clear processed flag
            const images = document.querySelectorAll(`.${CONFIG.processedClass}`);
            images.forEach(img => img.classList.remove(CONFIG.processedClass));
        }
    }

    // Toggle settings panel
    function toggleSettings() {
        const settingsPanel = document.querySelector('.manga-translator-settings');
        settingsPanel.classList.toggle('show');
    }

    // Scan page and process images
    function scanAndProcess() {
        if (!CONFIG.enabled) return;

        console.log('[Manga Translator] Scanning for manga images...');
        const images = findMangaImages();

        if (images.length > 0) {
            console.log(`[Manga Translator] Found ${images.length} images to process`);
            processQueue.push(...images);
            processNextInQueue();
        }
    }

    // Initialize the script
    function init() {
        console.log('[Manga Translator] Initializing...');

        // Create UI
        createUI();

        // Scan for images immediately
        if (CONFIG.enabled) {
            setTimeout(scanAndProcess, 1000);
        }

        // Watch for new images (for infinite scroll sites)
        const observer = new MutationObserver((mutations) => {
            if (!CONFIG.enabled) return;

            let hasNewImages = false;
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeName === 'IMG') {
                        hasNewImages = true;
                        break;
                    } else if (node.querySelectorAll) {
                        const images = node.querySelectorAll('img');
                        if (images.length > 0) {
                            hasNewImages = true;
                            break;
                        }
                    }
                }
                if (hasNewImages) break;
            }

            if (hasNewImages) {
                setTimeout(scanAndProcess, 500);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        console.log('[Manga Translator] Initialization complete');
    }

    // Start the script when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
