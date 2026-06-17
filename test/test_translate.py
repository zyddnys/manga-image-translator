import pytest
from PIL import Image
from manga_translator.config import Config, Direction, Ocr, Detector, DetectorConfig, InpainterConfig, OcrConfig, RenderConfig, Translator, Renderer, Inpainter, TranslatorConfig, UpscaleConfig, Upscaler
from manga_translator.utils import Context
from manga_translator.manga_translator import MangaTranslator
from pathlib import Path

@pytest.mark.asyncio
async def test_translate(image_path):
    if image_path is None:
        pytest.skip("pass, image_path not set")
    assert Path(image_path).exists(), f"missing fixture: {image_path}"

    image = Image.open(image_path)
    translator = MangaTranslator({
        'verbose': True,
        'font_path': 'fonts/msyh.ttc',
        'use_gpu': False,
        'kernel_size': 3
    })
    config = Config(
        render=RenderConfig(
            renderer=Renderer.default,
            fit_to_box=True,
        ),
        translator=TranslatorConfig(
            translator=Translator.gemini,
            model_name='gemini-3.5-flash',
            target_lang="CHS"
        ),
        detector=DetectorConfig(
            detector=Detector.ctd
        ),
        inpainter=InpainterConfig(
            inpainter=Inpainter.lama_large
        ),
        ocr=OcrConfig(
            ocr=Ocr.mocr,
            use_mocr_merge=True, # 合并的检测框合成一个大框（逻辑和 textline_merge 类似），再对整框跑一次 MangaOcr
            # prob=0.1 # mocr 写死了0.2置信度
        ),
        upscale=UpscaleConfig(
            upscaler=Upscaler.esrgan,
            revert_upscaling=False,
            upscale_ratio=2
        )
    )
    # 精修后的结果反而不准了
    ctx = await translator.translate(image, config)