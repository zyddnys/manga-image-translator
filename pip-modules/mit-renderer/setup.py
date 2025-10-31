import os
import re

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

FILES = {
    "__init__.py": "manga_translator/rendering/__init__.py",
    "text_render_pillow_eng.py": "manga_translator/rendering/text_render_pillow_eng.py",
    "ballon_extractor.py": "manga_translator/rendering/ballon_extractor.py",
    "text_render_eng.py": "manga_translator/rendering/text_render_eng.py",
    "text_render.py": "manga_translator/rendering/text_render.py",
    "utils/textblock.py": "manga_translator/utils/textblock.py",
    "utils/generic2.py": "manga_translator/utils/generic2.py",
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "mit_renderer")


def fix_imports(content):
    """
    Replace relative imports like 'from ..utils import' with 'from .utils import'
    """
    content = re.sub(r'from \.\.utils import', 'from .utils import', content)
    return content


def build_files():
    print(f"🔧 Copying {len(FILES)} files into {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for target_name, rel_path in FILES.items():
        src_path = os.path.join(REPO_ROOT, rel_path)
        dst_path = os.path.join(OUTPUT_DIR, target_name)
        print(f"Copying {src_path} -> {dst_path}")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = fix_imports(content)

        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ All files copied into {OUTPUT_DIR}")


class build_py(_build_py):
    def run(self):
        build_files()
        super().run()


setup(
    packages=["mit_renderer"],
    cmdclass={"build_py": build_py})
