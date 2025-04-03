import manga_translator.translators as translators


async def translate_text(
    texts: list[str],
    *,
    translator_key: str,
    target_lang: str,
) -> list[str]:
    translator = translators.get_translator(translator_key)
    if isinstance(translator, translators.OfflineTranslator):
        await translator.download()
        await translator.load("auto", target_lang, device="cpu")
    result = await translator.translate(
        from_lang="auto",
        to_lang=target_lang,
        queries=texts,
    )
    return result
