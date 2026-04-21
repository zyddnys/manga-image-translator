"""
Standalone script to prepare models for Modal deployment.

This script can be used to:
1. Test model downloads locally before deploying
2. Pre-populate the Modal volume with models
3. Verify model integrity

Usage:
    # Test locally
    python deploy/prepare_models.py --test-local

    # Run via Modal to populate volume
    modal run deploy.modal_app::download_models
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_local_download():
    """
    Test model downloads locally using the docker_prepare.py script.
    """
    import subprocess

    print("Testing local model download...")
    print("This will download models to ./models/ directory")

    script_path = Path(__file__).parent.parent / "docker_prepare.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--continue-on-error"],
            check=True,
            cwd=str(script_path.parent),
        )

        print("✅ Local model download test completed successfully!")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during model download: {e}")
        return 1


def verify_models(model_dir: Path):
    """
    Verify that required models are present in the model directory.

    Args:
        model_dir: Path to the models directory

    Returns:
        bool: True if all required models are present
    """
    print(f"\nVerifying models in: {model_dir}")

    required_dirs = [
        "detection",
        "ocr",
        "inpainting",
    ]

    all_present = True

    for dir_name in required_dirs:
        dir_path = model_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("**/*")))
            print(f"  ✅ {dir_name}: {file_count} files")
        else:
            print(f"  ❌ {dir_name}: NOT FOUND")
            all_present = False

    # Check for transformers cache
    transformers_cache = model_dir / "transformers"
    if transformers_cache.exists():
        print(f"  ✅ transformers cache: present")
    else:
        print(f"  ⚠️  transformers cache: not present (will download on first use)")

    return all_present


def estimate_model_size(model_dir: Path):
    """
    Estimate the total size of downloaded models.

    Args:
        model_dir: Path to the models directory

    Returns:
        int: Total size in bytes
    """
    import os

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    return total_size


def format_size(size_bytes):
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def list_models(model_dir: Path):
    """
    List all downloaded models with their sizes.

    Args:
        model_dir: Path to the models directory
    """
    print(f"\nModel directory contents: {model_dir}")
    print("=" * 80)

    if not model_dir.exists():
        print("Model directory does not exist!")
        return

    for category_dir in sorted(model_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        print(f"\n📁 {category_dir.name}/")

        # Count files and calculate size
        files = list(category_dir.rglob("*"))
        file_count = len([f for f in files if f.is_file()])
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        print(f"   Files: {file_count}")
        print(f"   Size: {format_size(total_size)}")

        # List top-level items
        for item in sorted(category_dir.iterdir())[:10]:  # Limit to first 10
            if item.is_file():
                size = format_size(item.stat().st_size)
                print(f"   - {item.name} ({size})")
            elif item.is_dir():
                sub_files = list(item.rglob("*"))
                sub_count = len([f for f in sub_files if f.is_file()])
                print(f"   - {item.name}/ ({sub_count} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare models for Manga Image Translator"
    )
    parser.add_argument(
        "--test-local",
        action="store_true",
        help="Test model download locally"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify models are present"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List downloaded models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Path to models directory (default: ./models)"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()

    if args.test_local:
        return test_local_download()

    if args.verify:
        success = verify_models(model_dir)
        if success:
            print("\n✅ All required models are present!")
            return 0
        else:
            print("\n❌ Some models are missing. Run with --test-local to download.")
            return 1

    if args.list:
        list_models(model_dir)
        total_size = estimate_model_size(model_dir)
        print("\n" + "=" * 80)
        print(f"Total model size: {format_size(total_size)}")
        return 0

    # Default: show usage
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
