// ==UserScript==
// @name         manga-image-translator (viewport only, wait after capture, fullscreen)
// @namespace    ankpixel.translate
// @version      0.3.1
// @description  Capture current viewport, then show waiting dialog while sending to backend;
// @author       ank
// @match        *://*/*
// @grant        GM_addStyle
// @grant        GM_xmlhttpRequest
// @connect    127.0.0.1
// @require      https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js
// ==/UserScript==

(function () {
  'use strict';

  const API_HOST = 'http://127.0.0.1:8000';
  const API_PATH = '/translate/image';

  const FIXED_CONFIG = {
    "render": {
      "alignment": "auto",
      "direction": "auto",
      "disable_font_border": false,
      "font_size_minimum": -1,
      "font_size_offset": 0,
      "gimp_font": "Sans-serif",
      "lowercase": false,
      "no_hyphenation": false,
      "renderer": "default",
      "rtl": true,
      "uppercase": false,
      "font_color": null,
      "line_spacing": null,
      "font_size": null
    },
    "upscale": {
      "revert_upscaling": false,
      "upscaler": "esrgan",
      "upscale_ratio": null
    },
    "translator": {
      "enable_post_translation_check": true,
      "no_text_lang_skip": false,
      "post_check_max_retry_attempts": 3,
      "post_check_repetition_threshold": 20,
      "post_check_target_lang_threshold": 0.5,
      "target_lang": "CHS",
      "translator": "custom_openai",
      "skip_lang": null,
      "gpt_config": null,
      "translator_chain": null,
      "selective_translation": null
    },
    "detector": {
      "box_threshold": 0.7,
      "det_auto_rotate": false,
      "det_gamma_correct": false,
      "det_invert": false,
      "det_rotate": false,
      "detection_size": 2048,
      "detector": "default",
      "text_threshold": 0.5,
      "unclip_ratio": 2.3
    },
    "colorizer": {
      "colorization_size": 576,
      "colorizer": "none",
      "denoise_sigma": 30
    },
    "inpainter": {
      "inpainter": "lama_large",
      "inpainting_precision": "bf16",
      "inpainting_size": 2048
    },
    "ocr": {
      "ignore_bubble": 0,
      "min_text_length": 0,
      "ocr": "48px",
      "use_mocr_merge": false
    },
    "force_simple_sort": false,
    "kernel_size": 3,
    "mask_dilation_offset": 20,
    "filter_text": null
  };

  // ---------- Styles ----------
  GM_addStyle(`
    #__tp_translate_btn__ {
      position: fixed; right: 18px; bottom: 18px;
      z-index: 2147483647;
      background: #111827; color: #fff;
      border-radius: 999px; padding: 10px 14px; font-size: 14px;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans";
      box-shadow: 0 6px 18px rgba(0,0,0,.25); cursor: pointer; user-select: none;
      border: 1px solid rgba(255,255,255,.12);
    }
    #__tp_translate_btn__:hover { transform: translateY(-1px); }
    #__tp_translate_btn__:active { transform: translateY(0); }

    /* 全屏覆盖层 */
    #__tp_full_overlay__ {
      position: fixed; inset: 0; z-index: 2147483646;
      display: none;
      background: rgba(0,0,0,0.9);
      color: #e5e7eb;
      overflow: hidden;
    }
    #__tp_overlay_toolbar__ {
      position: absolute; top: 10px; right: 10px; display: flex; gap: 8px;
      z-index: 1;
    }
    #__tp_close_btn__ {
      background: rgba(255,255,255,.08);
      color: #e5e7eb; border: 1px solid rgba(255,255,255,.12);
      padding: 6px 10px; border-radius: 8px; cursor: pointer; font-size: 14px;
    }
    #__tp_download_btn__ {
      background: rgba(255,255,255,.08);
      color: #e5e7eb; border: 1px solid rgba(255,255,255,.12);
      padding: 6px 10px; border-radius: 8px; cursor: pointer; font-size: 14px;
      text-decoration: none;
    }
    #__tp_wait_area__ {
      width: 100vw; height: 100vh;
      display: flex; align-items: center; justify-content: center; flex-direction: column;
      gap: 14px; text-align: center; padding: 20px;
    }
    #__tp_spinner__ {
      width: 28px; height: 28px; border: 3px solid rgba(255,255,255,.25);
      border-top-color: #fff; border-radius: 50%;
      animation: tp-spin .9s linear infinite;
    }
    @keyframes tp-spin { to { transform: rotate(360deg); } }
    #__tp_wait_text__ { font-size: 15px; }

    /* 全屏图像容器 */
    #__tp_img_stage__ {
      width: 100vw; height: 100vh; display: none; position: relative;
    }
    #__tp_img__ {
      position: absolute; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      max-width: 100vw; max-height: 100vh; object-fit: contain;
      image-rendering: auto; box-shadow: 0 8px 24px rgba(0,0,0,.5);
      border-radius: 2px;
    }
  `);

  // ---------- UI ----------
  function ensureUI() {
    if (!document.getElementById('__tp_translate_btn__')) {
      const btn = document.createElement('button');
      btn.id = '__tp_translate_btn__';
      btn.textContent = 'Translate';
      btn.addEventListener('click', startTranslateFlow);
      document.body.appendChild(btn);
    }

    if (!document.getElementById('__tp_full_overlay__')) {
      const overlay = document.createElement('div');
      overlay.id = '__tp_full_overlay__';
      overlay.innerHTML = `
        <div id="__tp_overlay_toolbar__">
          <a id="__tp_download_btn__" href="#" download="translated.png" style="display:none">下载</a>
          <button id="__tp_close_btn__">✕ 关闭</button>
        </div>
        <div id="__tp_wait_area__" style="display:none">
          <div id="__tp_spinner__"></div>
          <div id="__tp_wait_text__">等待翻译中…</div>
        </div>
        <div id="__tp_img_stage__">
          <img id="__tp_img__" alt="translated"/>
        </div>
      `;
      document.body.appendChild(overlay);

      overlay.querySelector('#__tp_close_btn__').addEventListener('click', hideOverlay);
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.style.display === 'block') hideOverlay();
      });
      overlay.addEventListener('click', (e) => {
        if (e.target.id === '__tp_full_overlay__') hideOverlay();
      });
    }
  }

  function showWaitingOverlay() {
    const overlay = document.getElementById('__tp_full_overlay__');
    overlay.style.display = 'block';
    overlay.querySelector('#__tp_wait_area__').style.display = 'flex';
    overlay.querySelector('#__tp_img_stage__').style.display = 'none';
    overlay.querySelector('#__tp_download_btn__').style.display = 'none';
    document.documentElement.style.overflow = 'hidden';
  }

  function showImageOverlay(blobUrl) {
    const overlay = document.getElementById('__tp_full_overlay__');
    const img = overlay.querySelector('#__tp_img__');
    const dl = overlay.querySelector('#__tp_download_btn__');
    img.src = blobUrl;
    dl.href = blobUrl;

    overlay.querySelector('#__tp_wait_area__').style.display = 'none';
    overlay.querySelector('#__tp_img_stage__').style.display = 'block';
    overlay.querySelector('#__tp_download_btn__').style.display = 'inline-block';
  }

  function hideOverlay() {
    const overlay = document.getElementById('__tp_full_overlay__');
    overlay.style.display = 'none';
    document.documentElement.style.overflow = '';
  }

  // ---------- Core ----------
  async function captureViewportDataURL() {
    // 只截当前可见区域；临时隐藏“Translate”按钮避免出现在截图里
    const btn = document.getElementById('__tp_translate_btn__');
    const prevVis = btn ? btn.style.visibility : '';
    if (btn) btn.style.visibility = 'hidden';

    try {
      const canvas = await html2canvas(document.body, {
        useCORS: true,
        allowTaint: true,
        windowWidth: window.innerWidth,
        windowHeight: window.innerHeight,
        x: window.scrollX,
        y: window.scrollY,
        width: window.innerWidth,
        height: window.innerHeight,
        scale: Math.min(window.devicePixelRatio || 1, 2)
      });
      return canvas.toDataURL('image/png');
    } finally {
      if (btn) btn.style.visibility = prevVis;
    }
  }

  function postTranslate(imageDataUrl) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: 'POST',
        url: API_HOST + API_PATH,
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({ image: imageDataUrl, config: FIXED_CONFIG }),
        responseType: 'blob',
        onload: (resp) => {
          if (resp.status >= 200 && resp.status < 300) {
            const blob = resp.response;
            if (blob && blob.size > 0) resolve(blob);
            else reject(new Error('空响应'));
          } else {
            reject(new Error(`HTTP ${resp.status}`));
          }
        },
        onerror: () => reject(new Error('网络错误或被拦截')),
        ontimeout: () => reject(new Error('请求超时')),
        timeout: 120000
      });
    });
  }

  async function startTranslateFlow() {
    try {
      // ① 先截图（此时不展示任何提示层，防止被截进图里）
      const imageDataUrl = await captureViewportDataURL();

      // ② 截完再显示等待对话框
      showWaitingOverlay();

      // ③ 发起翻译请求并等待返回
      const blob = await postTranslate(imageDataUrl);
      const blobUrl = URL.createObjectURL(blob);

      // ④ 全屏展示译后图片
      showImageOverlay(blobUrl);
    } catch (err) {
      console.error('[Translate] error:', err);
      const overlay = document.getElementById('__tp_full_overlay__');
      const wait = overlay.querySelector('#__tp_wait_area__');
      overlay.style.display = 'block';
      wait.style.display = 'flex';
      overlay.querySelector('#__tp_img_stage__').style.display = 'none';
      const text = document.getElementById('__tp_wait_text__');
      if (text) text.textContent = '翻译失败：' + (err && err.message ? err.message : '未知错误');
    }
  }

  // ---------- Init ----------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureUI, { once: true });
  } else {
    ensureUI();
  }
})();
