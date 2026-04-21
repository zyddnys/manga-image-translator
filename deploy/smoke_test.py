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
from typing import Dict, Optional

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


class SmokeTest:
    """Smoke test suite for Manga Image Translator."""

    def __init__(self, base_url: str, verbose: bool = False):
        """
        Initialize smoke test.

        Args:
            base_url: Base URL of the deployed service
            verbose: Enable verbose output
        """
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
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

    def test_translate_json(self):
        """Test the /translate/json endpoint."""
        url = f"{self.base_url}/translate/json"

        # Create test image
        img = create_test_image()
        img_base64 = image_to_base64(img)

        payload = {
            "image": img_base64,
            "config": {
                "translator": "none",  # Use 'none' to skip actual translation
                "detector": "default",
                "ocr": "48px",
                "inpainter": "none",
                "direction": "auto",
                "target_lang": "ENG",
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=120,  # Translation can take time
        )
        response.raise_for_status()

        data = response.json()
        if self.verbose:
            print_info(f"Response keys: {list(data.keys())}")

        # Check for expected response structure
        assert "translation_mask" in data or "image_base64" in data, \
            "Response missing expected fields"

        print_info("Translation completed successfully")

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

    def test_streaming_endpoint(self):
        """Test a streaming endpoint (basic check)."""
        url = f"{self.base_url}/translate/json/stream"

        img = create_test_image()
        img_base64 = image_to_base64(img)

        payload = {
            "image": img_base64,
            "config": {
                "translator": "none",
                "detector": "default",
                "ocr": "48px",
                "inpainter": "none",
                "target_lang": "ENG",
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        # Check that we're getting streaming responses
        chunk_count = 0
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                chunk_count += 1
                if chunk_count >= 3:  # Just verify we get multiple chunks
                    break

        print_info(f"Received {chunk_count} chunks in stream")
        assert chunk_count > 0, "No streaming data received"

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
        self.run_test("JSON Translation", self.test_translate_json)
        self.run_test("Streaming Endpoint", self.test_streaming_endpoint)

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
        choices=["health", "queue", "translate", "stream", "results"],
        help="Run only a specific test"
    )

    args = parser.parse_args()

    # Ensure PIL is available
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print_error("Pillow is required for smoke tests")
        print_info("Install with: pip install Pillow")
        return 1

    tester = SmokeTest(args.url, args.verbose)

    if args.test:
        # Run specific test
        test_map = {
            "health": tester.test_health_check,
            "queue": tester.test_queue_size,
            "translate": tester.test_translate_json,
            "stream": tester.test_streaming_endpoint,
            "results": tester.test_results_list,
        }
        tester.run_test(args.test.title(), test_map[args.test])
        return 0 if tester.results[0][1] else 1
    else:
        # Run all tests
        return tester.run_all()


if __name__ == "__main__":
    sys.exit(main())
