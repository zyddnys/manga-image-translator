"""
Smoke test and health check for Modal deployment.

This script tests critical endpoints of the deployed Manga Image Translator
to ensure everything is working correctly after deployment.

Usage:
    # Test deployed Modal endpoint
    python deploy/smoke_test.py --url https://your-app.modal.run

    # Test local server
    python deploy/smoke_test.py --url http://localhost:5003

    # Run all tests with verbose output
    python deploy/smoke_test.py --url https://your-app.modal.run --verbose
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")


def print_error(msg: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")


def create_test_image() -> Image.Image:
    """
    Create a simple test image with text.

    Returns:
        PIL Image object
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create a simple white image with black text
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()

    # Draw some text
    text = "テスト\nTest Image\n测试"
    draw.text((50, 50), text, fill='black', font=font)

    return img


def image_to_base64(img: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        img: PIL Image object

    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        b64_string: Base64 encoded image string

    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data))


def save_result_image(result_img: Image.Image, original_path: Optional[str] = None, output_dir: Optional[str] = None) -> Path:
    """
    Save result image with -result suffix.

    Args:
        result_img: PIL Image object to save
        original_path: Original image file path (optional)
        output_dir: Output directory (optional, defaults to current directory)

    Returns:
        Path to saved result image
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if original_path:
        # Use original filename with -result suffix
        original_path = Path(original_path)
        stem = original_path.stem
        ext = ".png"
        result_filename = f"{stem}-result{ext}"
        result_path = output_dir / result_filename
    else:
        # Use default filename with timestamp
        timestamp = int(time.time())
        result_path = output_dir / f"test-result-{timestamp}.png"

    result_img.save(result_path)
    return result_path


class SmokeTest:
    """Smoke test suite for Manga Image Translator."""

    def __init__(self, base_url: str, verbose: bool = False, test_images: Optional[List[str]] = None):
        """
        Initialize smoke test.

        Args:
            base_url: Base URL of the deployed service
            verbose: Enable verbose output
            test_images: Paths to test image files (optional)
        """
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.test_images = test_images or []
        self.results = []

    def run_test(self, name: str, func):
        """
        Run a single test and record result.

        Args:
            name: Test name
            func: Test function to run

        Returns:
            bool: True if test passed
        """
        print(f"\n{Colors.BOLD}Running: {name}{Colors.RESET}")
        start_time = time.time()

        try:
            func()
            elapsed = time.time() - start_time
            print_success(f"PASSED ({elapsed:.2f}s)")
            self.results.append((name, True, elapsed, None))
            return True

        except Exception as e:
            elapsed = time.time() - start_time
            print_error(f"FAILED ({elapsed:.2f}s)")
            print_error(f"Error: {str(e)}")
            self.results.append((name, False, elapsed, str(e)))
            return False

    def test_health_check(self):
        """Test the /health endpoint."""
        url = f"{self.base_url}/health"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if self.verbose:
            print_info(f"Response: {json.dumps(data, indent=2)}")

        assert data["status"] == "healthy", f"Unhealthy status: {data}"
        print_info(f"GPU available: {data.get('gpu_available', 'unknown')}")

    def test_queue_size(self):
        """Test the /queue-size endpoint."""
        url = f"{self.base_url}/queue-size"
        response = requests.post(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if self.verbose:
            print_info(f"Response: {json.dumps(data, indent=2)}")

        print_info(f"Queue size: {data.get('size', 'unknown')}")

    def test_translate_image(self):
        """Test the /translate/image endpoint and save result image."""
        url = f"{self.base_url}/translate/image"

        # Load the first test image for single-image endpoint tests.
        if self.test_images:
            primary_image = self.test_images[0]
            print_info(f"Using test image: {primary_image}")
            img = Image.open(primary_image)
            original_path = Path(primary_image)
        else:
            print_error("No test image provided")
            return 

        img_base64 = image_to_base64(img)
        img_data_url = f"data:image/png;base64,{img_base64}"

        payload = {
            "image": img_data_url,
            "config": {
                "translator": {
                    "translator": "youdao",  # Use 'none' to skip actual translation
                    "target_lang": "CHS",
                },
                "detector": {
                    "detector": "ctd",
                },
                "ocr": {
                    "ocr": "48px",
                },
                "inpainter": {
                    "inpainter": "default",
                },
                "render": {
                    "direction": "auto",
                }
            }
        }

        # Get rendered image using /translate/image endpoint
        print_info("Fetching rendered image from /translate/image endpoint...")
        response_img = requests.post(
            url,
            json=payload,
            timeout=120,
        )
        response_img.raise_for_status()

        # Save result image
        try:
            result_img = Image.open(io.BytesIO(response_img.content))
            result_path = save_result_image(result_img, original_path=str(original_path), output_dir=str(original_path.parent))
            print_info(f"Saved result image to: {result_path}")
        except Exception as e:
            print_warning(f"Failed to save result image: {e}")

        print_info("Translation completed successfully")

    def test_translate_form_image(self):
        """Test the /translate/with-form/image endpoint with multipart form."""
        url = f"{self.base_url}/translate/with-form/image"

        # Use fixed test asset for more realistic OCR coverage.
        if self.test_images:
            primary_image = self.test_images[0]
            print_info(f"Using test image: {primary_image}")
            img = Image.open(primary_image)
            original_path = Path(primary_image)
        else:
            print_error("No test image provided")
            return

        # Convert image to bytes for upload
        image_buffer = io.BytesIO()
        img.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        config = {
            "translator": {
                "translator": "youdao",
                "target_lang": "CHS",
            },
            "detector": {
                "detector": "ctd",
            },
            "ocr": {
                "ocr": "48px",
            },
            "inpainter": {
                "inpainter": "default",
            },
            "render": {
                "direction": "auto",
            }
        }

        response = requests.post(
            url,
            files={"image": (original_path.name, image_buffer, "image/png")},
            data={"config": json.dumps(config)},
            timeout=120,  # Translation can take time
        )
        response.raise_for_status()

        # Response is image bytes, similar to server/main.py image_form endpoint (line 116-125)
        # Save result image with -result suffix in the same directory as original
        try:
            result_img = Image.open(io.BytesIO(response.content))
            result_path = save_result_image(result_img, original_path=str(original_path), output_dir=str(original_path.parent))
            print_info(f"Saved result image to: {result_path}")
        except Exception as e:
            print_warning(f"Failed to save result image: {e}")

        print_info("Form translation completed successfully")

    def test_translate_batch_json(self):
        """Test the /translate/batch/json endpoint."""
        url = f"{self.base_url}/translate/batch/json"

        if not self.test_images:
            print_error("No test image provided")
            return
        print_info(f"Using {len(self.test_images)} test image(s) for batch translation")
        image_data_urls = []
        for image_path in self.test_images:
            img = Image.open(image_path)
            img_base64 = image_to_base64(img)
            image_data_urls.append(f"data:image/png;base64,{img_base64}")

        payload = {
            "images": image_data_urls,
            "config": {
                "translator": {
                    # Keep smoke test fast and stable by avoiding network translators.
                    "translator": "youdao",
                    "target_lang": "CHS",
                },
                "detector": {
                    "detector": "ctd",
                },
                "ocr": {
                    "ocr": "48px",
                },
                "inpainter": {
                    "inpainter": "default",
                },
                "render": {
                    "direction": "auto",
                },
            },
            "batch_size": max(1, len(image_data_urls)),
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        if self.verbose:
            print_info(f"Response: {json.dumps(data, indent=2)}")

        assert isinstance(data, list), f"Expected list response, got: {type(data)}"
        assert len(data) == len(payload["images"]), (
            f"Expected {len(payload['images'])} results, got {len(data)}"
        )

        for idx, item in enumerate(data):
            assert isinstance(item, dict), f"Result {idx} is not object: {type(item)}"
            assert "translations" in item, f"Result {idx} missing 'translations'"
            assert isinstance(item["translations"], list), (
                f"Result {idx} has invalid translations type: {type(item['translations'])}"
            )
            if idx < len(self.test_images):
                original_path = Path(self.test_images[idx])
                debug_folder = item.get("debug_folder")
                if debug_folder:
                    try:
                        result_url = f"{self.base_url}/result/{debug_folder}/final.png"
                        result_response = requests.get(result_url, timeout=60)
                        result_response.raise_for_status()
                        result_img = Image.open(io.BytesIO(result_response.content))
                        result_path = save_result_image(
                            result_img,
                            original_path=str(original_path),
                            output_dir=str(original_path.parent),
                        )
                        print_info(f"Saved batch result image {idx + 1} to: {result_path}")
                    except Exception as e:
                        print_warning(f"Failed to save batch result image {idx + 1}: {e}")
                else:
                    print_warning(f"Batch result {idx + 1} missing 'debug_folder', skip image save")

        print_info(f"Batch translation completed with {len(data)} result(s)")

    def test_results_list(self):
        """Test the /results/list endpoint."""
        url = f"{self.base_url}/results/list"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if self.verbose:
            print_info(f"Response: {json.dumps(data, indent=2)}")

        result_count = len(data.get("results", []))
        print_info(f"Found {result_count} results in storage")

    def run_all(self):
        """Run all smoke tests."""
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}Manga Image Translator - Smoke Test{Colors.RESET}")
        print(f"{Colors.BOLD}Testing: {self.base_url}{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        # Core tests
        self.run_test("Health Check", self.test_health_check)
        self.run_test("Queue Size", self.test_queue_size)
        self.run_test("Results List", self.test_results_list)

        # Translation tests
        self.run_test("Translate Image", self.test_translate_image)
        self.run_test("Translate Form Image", self.test_translate_form_image)
        self.run_test("Translate Batch JSON", self.test_translate_batch_json)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        passed = sum(1 for _, success, _, _ in self.results if success)
        failed = len(self.results) - passed
        total_time = sum(elapsed for _, _, elapsed, _ in self.results)

        for name, success, elapsed, error in self.results:
            status = f"{Colors.GREEN}PASS{Colors.RESET}" if success else f"{Colors.RED}FAIL{Colors.RESET}"
            print(f"  {status} - {name} ({elapsed:.2f}s)")
            if error and self.verbose:
                print(f"       Error: {error}")

        print()
        print(f"Total: {len(self.results)} tests")
        print(f"Passed: {Colors.GREEN}{passed}{Colors.RESET}")
        print(f"Failed: {Colors.RED}{failed}{Colors.RESET}")
        print(f"Total time: {total_time:.2f}s")

        if failed == 0:
            print()
            print_success("All tests passed! 🎉")
            return 0
        else:
            print()
            print_error(f"{failed} test(s) failed")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for Manga Image Translator deployment"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Base URL of the deployed service (e.g., https://your-app.modal.run)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["health", "queue", "translate_image", "translate_form_image", "translate_batch_json", "results"],
        help="Run only a specific test"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        action="append",
        help="Path to test image file (repeat to pass multiple images)"
    )

    args = parser.parse_args()

    # Ensure PIL is available
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print_error("Pillow is required for smoke tests")
        print_info("Install with: pip install Pillow")
        return 1

    # Validate test image if provided
    if args.image:
        for image_path in args.image:
            if not Path(image_path).exists():
                print_error(f"Test image not found: {image_path}")
                return 1

    tester = SmokeTest(args.url, args.verbose, test_images=args.image)

    if args.test:
        # Run specific test
        test_map = {
            "health": tester.test_health_check,
            "queue": tester.test_queue_size,
            "translate_image": tester.test_translate_image,
            "translate_form_image": tester.test_translate_form_image,
            "translate_batch_json": tester.test_translate_batch_json,
            "results": tester.test_results_list,
        }
        tester.run_test(args.test.title(), test_map[args.test])
        return 0 if tester.results[0][1] else 1
    else:
        # Run all tests
        return tester.run_all()


if __name__ == "__main__":
    sys.exit(main())
