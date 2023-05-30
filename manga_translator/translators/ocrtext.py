

import hashlib
import urllib.parse
import random
import sqlite3

from .common import CommonTranslator, InvalidServerResponse, MissingAPIKeyException


class OCRTextTranslator(CommonTranslator):
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

# create cursor object
cursor = conn. cursor()
# Read the page field value of the first piece of data in the page_count table
cursor.execute('SELECT page FROM page_count LIMIT 1')
result = cursor. fetchone()
page = result[0]

atext = ""
atext = "@Page "+str(page)+", "+str(len(queries))+" sentences in total.\r\n" # test to add special symbols
# atext = "Page "+str(page)+", a total of "+str(len(queries))+" sentences.\r\n\r\n" # Test to add special symbols

result_list = [] # label by number

for i, text in enumerate(queries):
# Replace text with wrong words
for key, value in dict_kv.items():
text = text.replace(key, value)
atext += f"{i + 1}.{text}\r\n"
# result_list.append(str(i+1)) # mark order
atext = atext + "@Page "+str(page)+" End"

# filter[]
atext = atext.replace("[", "")
atext = atext.replace("]", "")

print("Current text overview:")
print(atext)
print("import database")




# Update the page field value
new_page = page + 1
cursor.execute('UPDATE page_count SET page=? WHERE rowid=1', (new_page,))


# create table
#cursor.execute('''CREATE TABLE manga_page(id INTEGER PRIMARY KEY AUTOINCREMENT, words TEXT, trans TEXT)''')

# insert data
#data = 'Hello, World!'
cursor.execute("INSERT INTO manga_page (words) VALUES (?)", (atext,))

# commit changes
conn.commit()

# close the connection
		conn.close()
		return result_list

