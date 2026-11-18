import builtins
from types import SimpleNamespace

import pytest

from manga_translator.mode import local as local_mode
from manga_translator.mode.local import MangaTranslatorLocal


def _windows_locale_open(file, mode='r', *args, **kwargs):
    # Simulate open() on a Windows machine whose ANSI code page is cp1252:
    # text-mode calls without an explicit encoding use the locale encoding.
    if 'b' not in mode and 'encoding' not in kwargs and len(args) < 2:
        kwargs['encoding'] = 'cp1252'
    return builtins.open(file, mode, *args, **kwargs)


@pytest.mark.asyncio
async def test_translate_txt_file_uses_utf8(tmp_path, monkeypatch):
    queries = ['僕はアイネと共に一度、宿の方に戻った', '目标意识']
    translations = ['Łukasz went back to the inn ✨', 'Sense of purpose']
    src = tmp_path / 'chapter.txt'
    src.write_text('\n'.join(queries), encoding='utf-8')
    dest = tmp_path / 'out' / 'chapter.txt'
    dest.parent.mkdir()

    received = []

    async def fake_dispatch(chain, texts, *args, **kwargs):
        received.append(list(texts))
        return translations

    monkeypatch.setattr(local_mode, 'dispatch_translation', fake_dispatch)
    monkeypatch.setattr(local_mode, 'open', _windows_locale_open, raising=False)

    translator = MangaTranslatorLocal.__new__(MangaTranslatorLocal)
    translator.use_mtpe = False
    translator._gpu_limited_memory = False
    translator.device = 'cpu'
    config = SimpleNamespace(translator=SimpleNamespace(translator_gen=None))

    assert await translator._translate_file(str(src), str(dest), config, None) is True
    assert received == [queries]
    assert dest.read_text(encoding='utf-8') == '\n'.join(translations)
