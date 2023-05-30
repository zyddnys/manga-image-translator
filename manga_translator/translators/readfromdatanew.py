

import hashlib
import urllib.parse
import random
import sqlite3

from .common import CommonTranslator, InvalidServerResponse, MissingAPIKeyException



class ReadTextTranslator(CommonTranslator):
	_LANGUAGE_CODE_MAP = {
		'CHS': 'zh',
		'CHT': 'cht',
		'JPN': 'ja',
		'ENG': 'en',
		'KOR': 'kor',
		'VIN': 'vie',
		'CSY': 'cs',
		'NLD': 'nl',
		'FRA': 'fra',
		'DEU': 'de',
		'HUN': 'hu',
		'ITA': 'it',
		'PLK': 'pl',
		'PTB': 'pt',
		'ROM': 'rom',
		'RUS': 'ru',
		'ESP': 'spa',
	}

	def __init__(self) -> None:
		super().__init__()

	async def _translate(self, from_lang, to_lang, queries):
		# Create a connection and open the database
		conn = sqlite3.connect('manga_page.db')

		# Create a cursor object
		cursor = conn.cursor()
		# Read the page field value of the first piece of data in the page_count table
		cursor.execute('SELECT page FROM page_count LIMIT 1')
		result = cursor.fetchone()
		page = result[0]
		
		# Query the translation data of the current page number
		cursor.execute("SELECT trans FROM manga_page ORDER BY id ")
		#result = c.fetchone()
		result = cursor.fetchall()
		# Split the trans field value into a translations array by \r\n
		# translations = result[page - 1][0].split('\r\n')
		translations = result[page - 1][0].split('\n')

		result_list = []

		# Use list comprehension and strip() to remove leading and trailing spaces and tabs from strings
		translations = [s.strip() for s in translations]

		result_list.extend(translations)

		# If the translation length is not enough, add placeholders
		if len(result_list) < len(queries):
			result_list += [''] * (len(queries) - len(result_list))
		# If the length is too long, splicing at the end (actual effect to be tested)
		if len(result_list) > len(queries):
			new_end = ''.join(result_list[len(queries):])
			result_list = result_list[:len(queries)]
			last_string = result_list[-1] + new_end
			result_list[-1] = last_string

		# create form
		#cursor.execute('''CREATE TABLE manga_page(id INTEGER PRIMARY KEY AUTOINCREMENT, words TEXT, trans TEXT)''')

		# Update page field value
		new_page = page + 1
		cursor.execute('UPDATE page_count SET page=? WHERE rowid=1', (new_page,))

		# commit changes
		conn.commit()

		# close connection
		conn.close()
		return result_list

