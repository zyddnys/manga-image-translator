
import deepl
from .keys import DEEPL_AUTH_KEY

class Translator(object):
	def __init__(self):
		self.translator = deepl.Translator(DEEPL_AUTH_KEY)

	async def translate(self, from_lang, to_lang, query_text) :
		return self.translator.translate_text(query_text, target_lang = to_lang).text.split('\n')


