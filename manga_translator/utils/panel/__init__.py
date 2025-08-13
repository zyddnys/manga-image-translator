from .kumikolib import Kumiko
import tempfile, cv2, os
import hashlib
import numpy as np

# 全局分镜检测缓存 | Global panel detection cache
_panel_cache = {}
_cache_max_size = 50  # 最大缓存数量 | Maximum cache size

def _get_image_cache_key(img_rgb, rtl, method, use_gpu):
    """生成图片的缓存键 | Generate cache key for image"""
    # 使用图片的形状和部分像素数据生成快速哈希 | Use image shape and partial pixel data to generate fast hash
    h = hashlib.md5()
    h.update(f"{img_rgb.shape}_{rtl}_{method}_{use_gpu}".encode())
    # 采样部分像素数据以提高缓存命中率 | Sample partial pixel data to improve cache hit rate
    if img_rgb.size > 0:
        step = max(1, img_rgb.size // 1000)  # 采样约1000个像素点 | Sample about 1000 pixels
        sample_data = img_rgb.flat[::step]
        h.update(sample_data.tobytes())
    return h.hexdigest()[:16]

def get_panels_from_array(img_rgb, rtl=True, method='dl', use_gpu=False):
    """
    Detect panels from image array with caching | 从图像数组中检测分镜（带缓存）

    Args:
        img_rgb: RGB format image array | RGB格式的图像数组
        rtl: Whether reading right-to-left (default True) | 是否为从右到左阅读 (默认True)
        method: Detection method ('kumiko' or 'dl', default 'dl') | 检测方法 ('kumiko' 或 'dl'，默认'dl')
        use_gpu: Whether to use GPU (default False), corresponds to --use-gpu parameter
                是否使用GPU (默认False)，对应--use-gpu参数

    Returns:
        List[Tuple[int, int, int, int]]: Panel list in format (x, y, w, h) | 分镜列表，格式为 (x, y, w, h)
    """
    global _panel_cache, _cache_max_size

    # 生成缓存键 | Generate cache key
    cache_key = _get_image_cache_key(img_rgb, rtl, method, use_gpu)

    # 检查缓存 | Check cache
    if cache_key in _panel_cache:
        return _panel_cache[cache_key]

    # 执行分镜检测 | Perform panel detection
    if method == 'dl':
        try:
            result = _get_panels_from_array_dl(img_rgb, rtl, use_gpu)
        except Exception as e:
            print(f"Deep learning detection failed, falling back to Kumiko: {e}")
            # Fall back to Kumiko method | 回退到Kumiko方法
            method = 'kumiko'
            result = _get_panels_from_array_kumiko(img_rgb, rtl)
    elif method == 'kumiko':
        result = _get_panels_from_array_kumiko(img_rgb, rtl)
    else:
        raise ValueError(f"Unsupported detection method: {method}")

    # 缓存结果（LRU策略） | Cache result (LRU strategy)
    if len(_panel_cache) >= _cache_max_size:
        # 删除最旧的缓存项 | Remove oldest cache item
        oldest_key = next(iter(_panel_cache))
        del _panel_cache[oldest_key]

    _panel_cache[cache_key] = result
    return result

def _get_panels_from_array_kumiko(img_rgb, rtl=True):
    """Detect panels using Kumiko method | 使用Kumiko方法检测分镜"""
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    path = tmp.name
    tmp.close()

    cv2.imwrite(path, img_rgb)

    k = Kumiko({'rtl': rtl})
    k.parse_image(path)
    infos = k.get_infos()

    os.unlink(path)

    return infos[0]['panels']

def _get_panels_from_array_dl(img_rgb, rtl=True, use_gpu=False):
    """Detect panels using deep learning method | 使用深度学习方法检测分镜"""
    from .dl_panel_detector import get_global_dl_interface

    interface = get_global_dl_interface(use_gpu=use_gpu)
    return interface.detect_panels(img_rgb, rtl)