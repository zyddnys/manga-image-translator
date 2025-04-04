from pathlib import Path

import tempfile


def create_unique_dir(purpose: str | None = None) -> Path:
    """Create a unique directory in the system's temporary directory."""
    # Use tempfile to create a temporary directory
    _r = tempfile.TemporaryDirectory(
        prefix=f"moeflow_companion.{purpose or 'unknown'}."
    )
    result = Path(_r.name)
    result.mkdir(parents=True, exist_ok=True)
    assert result.is_dir()
    return result
