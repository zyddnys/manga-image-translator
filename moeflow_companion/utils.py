from pathlib import Path
from PIL import Image

import tempfile


def create_unique_dir(purpose: str | None = None) -> Path:
    """Create a unique directory name in the system's temporary directory"""
    # Use tempfile to create a temporary directory
    _r = tempfile.TemporaryDirectory(
        prefix=f"moeflow_companion.{purpose or 'unknown'}."
    )
    result = Path(_r.name)
    # NOTE
    # result.mkdir(parents=True, exist_ok=True)
    # assert result.is_dir()
    return result


def read_image_dim(img_path: Path | str) -> tuple[int, int]:
    """Read the size of an image file."""
    with Image.open(img_path) as img:
        return img.size
