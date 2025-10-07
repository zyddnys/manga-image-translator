import os
import asyncio
from typing import List, Tuple, Any, Optional
import json
from pathlib import Path
from dotenv import load_dotenv
from manga_translator.config import Config, Translator
from manga_translator.translators import get_translator
from manga_translator.translators.common import OfflineTranslator
from manga_translator.manga_translator import MangaTranslator
from manga_translator.utils.textblock import TextBlock
from unittest.mock import Mock

load_dotenv("../../config/env.gpu", override=False)
project_root = Path(__file__).parent
bad_words_path = project_root / "bad_words.json"

main_config = Config(
    translator={"translator": "chatgpt", "target_lang": "ARA"},
)

async def setup_translator() -> Any:
    """Setup and configure translator instance"""
    translator = get_translator(Translator.chatgpt)
    if isinstance(translator, OfflineTranslator):
        await translator.load("auto", "ARA", "cuda")
    if main_config.translator:
        translator.parse_args(main_config.translator)
    return translator


def is_arabic_text(text: str) -> bool:
    """Check if text contains Arabic characters"""
    return any(
        0x0600 <= ord(char) <= 0x06FF
        or 0xFE70 <= ord(char) <= 0xFEFF
        or 0xFB50 <= ord(char) <= 0xFDFF
        for char in text
    )


def has_refusal_keywords(text: str) -> bool:
    """Check if text contains refusal keywords"""
    return any(
        word in text.lower()
        for word in ["sorry", "cannot", "unable", "decline", "refuse"]
    )


async def test_chatgpt_refusal() -> bool:
    """Test ChatGPT with bad words that should trigger refusal"""
    translator = await setup_translator()
    bad_words = json.load(open(bad_words_path))

    try:
        results = await translator.translate("auto", "ARA", bad_words, False)
        arabic_found = any(is_arabic_text(result) for result in results if result)
        return arabic_found
    except Exception:
        return False

def create_mock_regions(texts: List[str]) -> List[Any]:
    """Create mock text regions for testing"""
    regions = []
    for text in texts:
        region = Mock(spec=TextBlock)
        region.translation = text
        regions.append(region)
    return regions


async def test_arabic_language_bypass() -> bool:
    """Test Arabic language detection bypass"""
    try:
        config_params = {
            "verbose": False,
            "use_gpu": True,
            "pre_dict": None,
            "post_dict": None,
            "font_path": None,
            "kernel_size": 3,
            "detection_size": 2048,
            "inpainting_size": 512,
        }
        translator_instance = MangaTranslator(params=config_params)

        arabic_texts = [
            "مرحبا بك", "كيف حالك", "أهلا وسهلا", "شكرا لك", "مع السلامة",
            "صباح الخير", "مساء الخير", "تصبح على خير", "أراك لاحقا", "إلى اللقاء",
            "نعم", "لا", "من فضلك", "عفوا", "آسف", "أهلا مرة أخرى",
        ]

        regions = create_mock_regions(arabic_texts)
        result = await translator_instance._check_target_language_ratio(
            text_regions=regions, target_lang="ARA", min_ratio=0.5
        )
        return result if result else False

    except ImportError:
        return True
    except Exception:
        return False

async def test_normal_translation() -> bool:
    """Test normal translation with safe content"""
    translator = await setup_translator()
    normal_words = [
        "Hello world",
        "Good morning", 
        "Thank you very much",
        "How are you today?",
        "Have a nice day",
    ]

    try:
        results = await translator.translate("auto", "ARA", normal_words, False)
        success_count = sum(1 for result in results if result and result.strip())
        return success_count > 0
    except Exception:
        return False


async def run_test(test_name: str, test_func: Any) -> bool:
    """Run a single test and return result"""
    print(f"Running: {test_name}")
    try:
        return await test_func()
    except Exception:
        return False


async def main() -> bool:
    """Run all tests"""
    print("Manga Translator Tests")
    print("=" * 30)

    tests: List[Tuple[str, Any]] = [
        ("ChatGPT Refusal", test_chatgpt_refusal),
        ("Arabic Language Bypass", test_arabic_language_bypass),
        ("Normal Translation", test_normal_translation),
    ]

    results = []
    for test_name, test_func in tests:
        result = await run_test(test_name, test_func)
        results.append((test_name, result))

    print("\nResults:")
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")

    good_result = passed == len(results)
    print(f"\nOverall: {'PASSED ✅' if good_result else 'FAILED ❌'}")
    return good_result

if __name__ == "__main__":
    asyncio.run(main())

