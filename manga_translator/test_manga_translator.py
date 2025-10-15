import unittest
import sys
from typing import List
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from manga_translator.config import Config, TranslatorConfig
from manga_translator.manga_translator import MangaTranslator
from manga_translator.utils import Context


# Language test data
LANGUAGE_PAIRS = {
    "ARA": [
        "مرحبا بك", "كيف حالك", "أهلا وسهلا", "شكرا لك", "مع السلامة",
        "صباح الخير", "مساء الخير", "تصبح على خير", "أراك لاحقا", "إلى اللقاء",
        "نعم", "لا", "من فضلك", "عفوا", "آسف"
    ],
    "JPN": [
        "こんにちは", "ありがとう", "さようなら", "おはよう", "おやすみ",
        "お元気ですか", "はい", "いいえ", "すみません", "ごめんなさい",
        "どういたしまして", "また会いましょう", "いただきます", "ごちそうさま", "お願いします"
    ],
    "KOR": [
        "안녕하세요", "감사합니다", "안녕히 가세요", "좋은 아침", "잘 자요",
        "어떻게 지내세요", "예", "아니요", "죄송합니다", "제발",
        "환영합니다", "또 만나요", "건강하세요", "파이팅", "내일 봐요"
    ],
    "THA": [
        "สวัสดี", "ขอบคุณ", "ลาก่อน", "สวัสดีตอนเช้า", "ราตรีสวัสดิ์",
        "สบายดีไหม", "ใช่", "ไม่ใช่", "ขอโทษ", "กรุณา",
        "ยินดีต้อนรับ", "พบกันใหม่", "โชคดี", "สู้ๆ", "เจอกันพรุ่งนี้"
    ],
    "ENG": [
        "Hello", "Thank you", "Goodbye", "Good morning", "Good night",
        "How are you", "Yes", "No", "Sorry", "Please",
        "You're welcome", "See you later", "Good luck", "Take care", "See you tomorrow"
    ]
}


def create_mock_context():
    """Create a mock Context object"""
    ctx = Mock(spec=Context)
    ctx.from_lang = "JPN"
    return ctx


class ColoredTestResult(unittest.TextTestResult):
    """Custom test result with colored output"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_results = []
    
    def startTest(self, test):
        super().startTest(test)
        self.current_test = test
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append(('PASS', test))
    
    def addError(self, test, err):
        super().addError(test, err)
        self.test_results.append(('ERROR', test))
        print(f"\n❌ {test._testMethodName}")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_results.append(('FAIL', test))
        print(f"\n❌ {test._testMethodName}")


class ColoredTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output"""
    resultclass = ColoredTestResult
    
    def run(self, test):
        result = super().run(test)
        
        # Print summary
        print("\n" + "="*70)
        if result.wasSuccessful():
            print("✅ All tests passed!")
        else:
            print(f"❌ {len(result.failures) + len(result.errors)} test(s) failed")
        print("="*70 + "\n")
        
        return result


class TestFallbackIfRefused(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Setup test fixtures"""
        self.translator = MangaTranslator(params={
            "verbose": False,
            "use_gpu": False,
            "kernel_size": 3,
        })
        
        self.config = Config()
        self.config.translator = TranslatorConfig(
            translator="chatgpt",
            target_lang="ENG"
        )
        
        self.ctx = create_mock_context()
        self.texts = ["こんにちは", "さようなら", "ありがとう"]

    async def test_normal_list_result(self):
        """Test that normal list results are returned as-is"""
        result = ["Hello", "Goodbye", "Thank you"]
        
        output = await self.translator._fallback_if_refused(
            result, self.texts, self.config, self.ctx
        )
        
        self.assertEqual(output, result)
        self.assertIsInstance(output, list)

    async def test_tuple_with_refused_marker_triggers_gemini_fallback(self):
        """Test that refused tuple triggers Gemini fallback"""
        result = (False, ["__REFUSED__", "", ""])
        
        # Mock GeminiTranslator
        with patch('manga_translator.translators.gemini.GeminiTranslator') as MockGemini:
            mock_gemini_instance = Mock()
            mock_gemini_instance.parse_args = Mock()
            mock_gemini_instance._translate = AsyncMock(
                return_value=["Hello", "Goodbye", "Thank you"]
            )
            MockGemini.return_value = mock_gemini_instance
            
            output = await self.translator._fallback_if_refused(
                result, self.texts, self.config, self.ctx
            )
            
            # Verify Gemini was called
            MockGemini.assert_called_once()
            mock_gemini_instance.parse_args.assert_called_once_with(self.config.translator)
            mock_gemini_instance._translate.assert_called_once_with(
                self.ctx.from_lang, 
                self.config.translator.target_lang, 
                self.texts
            )
            
            self.assertEqual(output, ["Hello", "Goodbye", "Thank you"])

    async def test_gemini_fallback_failure_returns_original_texts(self):
        """Test that Gemini fallback failure returns original texts"""
        result = (False, ["__REFUSED__", "", ""])
        
        # Mock GeminiTranslator to raise exception
        with patch('manga_translator.translators.gemini.GeminiTranslator') as MockGemini:
            mock_gemini_instance = Mock()
            mock_gemini_instance.parse_args = Mock()
            mock_gemini_instance._translate = AsyncMock(
                side_effect=Exception("Gemini API error")
            )
            MockGemini.return_value = mock_gemini_instance
            
            output = await self.translator._fallback_if_refused(
                result, self.texts, self.config, self.ctx
            )
            
            # Should return original texts on failure
            self.assertEqual(output, self.texts)

    async def test_tuple_without_refused_marker(self):
        """Test tuple with False but without __REFUSED__ marker"""
        result = (False, ["Normal", "Translation", "Result"])
        
        output = await self.translator._fallback_if_refused(
            result, self.texts, self.config, self.ctx
        )
        
        self.assertEqual(output, ["Normal", "Translation", "Result"])


class TestLanguageDetection(unittest.IsolatedAsyncioTestCase):
    """Test language detection with actual language pairs"""
    
    async def asyncSetUp(self):
        """Setup test fixtures"""
        self.translator = MangaTranslator(params={
            "verbose": False,
            "use_gpu": False,
            "kernel_size": 3,
        })

    def create_mock_regions(self, texts: List[str]) -> List[Mock]:
        """Create mock text regions"""
        regions = []
        for text in texts:
            region = Mock()
            region.translation = text
            regions.append(region)
        return regions

    async def test_arabic_bypass_with_arabic_content(self):
        """Test that ARA target bypasses detection with Arabic content.
        This test will pass even without arabic bypass since Arabic did not pass through arabic reshaper.
        Arabic reshaper is causing the detection to fail.
        """
        print(f"  Testing: ARA target with Arabic content")
        regions = self.create_mock_regions(LANGUAGE_PAIRS["ARA"])
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="ARA"
        )
        
        self.assertTrue(result, "Should bypass for ARA target with Arabic content")

    async def test_arabic_bypass_with_arabic_reshaper(self):
        """Test that ARA target bypasses detection with Arabic content and Arabic reshaper"""
        print(f"  Testing: ARA target with Arabic content and Arabic reshaper")
        # import arabic reshaper 
        import arabic_reshaper , bidi.algorithm
        # reshape the Arabic text
        translations = [bidi.algorithm.get_display(arabic_reshaper.reshape(t)) for t in LANGUAGE_PAIRS["ARA"]]
        regions = self.create_mock_regions(translations)
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="ARA"
        )
        
        self.assertTrue(result, "Should bypass for ARA target with Arabic content and Arabic reshaper")

    async def test_english_detection_with_english_content(self):
        """Test English detection with English content"""
        print(f"  Testing: ENG target with English content")
        regions = self.create_mock_regions(LANGUAGE_PAIRS["ENG"])
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="ENG"
        )
        
        self.assertTrue(result, "Should detect English content correctly")

    async def test_japanese_detection_with_japanese_content(self):
        """Test Japanese detection with Japanese content"""
        print(f"  Testing: JPN target with Japanese content")
        regions = self.create_mock_regions(LANGUAGE_PAIRS["JPN"])
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="JPN"
        )
        
        self.assertTrue(result, "Should detect Japanese content correctly")

    async def test_korean_detection_with_korean_content(self):
        """Test Korean detection with Korean content"""
        print(f"  Testing: KOR target with Korean content")
        regions = self.create_mock_regions(LANGUAGE_PAIRS["KOR"])
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="KOR"
        )
        
        self.assertTrue(result, "Should detect Korean content correctly")

    async def test_thai_detection_with_thai_content(self):
        """Test Thai detection with Thai content"""
        print(f"  Testing: THA target with Thai content")
        regions = self.create_mock_regions(LANGUAGE_PAIRS["THA"])
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="THA"
        )
        
        self.assertTrue(result, "Should detect Thai content correctly")

    async def test_language_mismatch_detection(self):
        """Test that mismatched language is detected"""
        print(f"  Testing: ENG target with Arabic content (should fail)")
        regions = self.create_mock_regions(LANGUAGE_PAIRS["ARA"])
        
        result = await self.translator._check_target_language_ratio(
            text_regions=regions, target_lang="ENG"
        )
        
        self.assertFalse(result, "Should fail when content doesn't match target")


if __name__ == "__main__":
    # Use custom runner for colored output
    runner = ColoredTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner.run(suite)
    