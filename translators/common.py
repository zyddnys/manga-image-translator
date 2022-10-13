
class CommonTranslator:

    def _get_language_code(self, key):
        return key

    async def translate(self, from_lang, to_lang, queries):
        try:
            _from_lang = self._get_language_code(from_lang) if from_lang != 'auto' else 'auto'
            if not _from_lang:
                raise KeyError(from_lang)
            _to_lang = self._get_language_code(to_lang)
            if not _to_lang:
                raise KeyError(to_lang)
        except KeyError as e:
            print(f'Could not parse language key: "{e}"')
            raise e
        return await self._translate(_from_lang, _to_lang, queries)

    async def _translate(self, from_lang, to_lang, queries):
        raise NotImplementedError()
