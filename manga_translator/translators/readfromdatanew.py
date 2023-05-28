

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
		# 创建连接并打开数据库
		conn = sqlite3.connect('manga_page.db')

		# 创建游标对象
		cursor = conn.cursor()
		# 读取page_count表中第一条数据的page字段值
		cursor.execute('SELECT page FROM page_count LIMIT 1')
		result = cursor.fetchone()
		page = result[0]
		
		# 查询当前页码翻译数据
		cursor.execute("SELECT trans FROM manga_page ORDER BY id ")
		#result = c.fetchone()
		result = cursor.fetchall()
		# 将trans字段值通过\r\n分割成translations数组
		# translations = result[page - 1][0].split('\r\n')
		translations = result[page - 1][0].split('\n')

		result_list = []

		# 使用列表解析和strip()去除字符串的首尾的空格和制表符
		translations = [s.strip() for s in translations]

		result_list.extend(translations)

		# 如果翻译长度不够，补充占位
		if len(result_list) < len(queries):
			result_list += [''] * (len(queries) - len(result_list))
		# 如果长度过长，拼接在末尾（实际效果待测试）
		if len(result_list) > len(queries):
			new_end = ''.join(result_list[len(queries):])
			result_list = result_list[:len(queries)]
			last_string = result_list[-1] + new_end
			result_list[-1] = last_string

		# 创建表格
		#cursor.execute('''CREATE TABLE manga_page(id INTEGER PRIMARY KEY AUTOINCREMENT, words TEXT, trans TEXT)''')

		# 更新page字段值
		new_page = page + 1
		cursor.execute('UPDATE page_count SET page=? WHERE rowid=1', (new_page,))

		# 提交更改
		conn.commit()

		# 关闭连接
		conn.close()
		return result_list

