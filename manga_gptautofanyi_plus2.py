# -*- coding: utf-8 -*-
import sqlite3
import openai
import os
import re
import json
import jsonschema
import json5

# key
openai.api_key = ""

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# frequency_penalty_default = 0.35 # Used for CG with less air bubbles 0.35
# frequency_penalty_default = 0.1 # For comics with many bubbles
frequency_penalty_default = 0 # For comics with many bubbles

frequency_penalty_current = frequency_penalty_default

temp_origin_words_json = {}
temp_trans_words_json = {}



# 测试每个值都给出具体指示
prompt = '''
此JSON表示漫画的数页,每个“@Page”对象是一页,每个“@IndependentDialogue”的值是对话框内的文本,
根据[]内的要求翻译，尽可能按字面意思准确翻译，人物的中文名称请保持原状，尽可能少使用人称代词避免可能的代词歧义。
在漫画中，角色间的对话可能包括俚语、口语表达和面部表情等不太规范的内容，请尽可能保留这些元素的含义。
要根据动漫常用的表达习惯翻译,不添加注释。
只回复翻译后的JSON,不要其他回答,确保JSON没有语法错误。
'''



print(prompt)

# hensigui 20230512 Determine whether the translated json is in the same format as the original json
def compare_key_value_pairs(json_obj1, json_obj2):
    if len(json_obj1) != len(json_obj2):
        return False

    for key, value1 in json_obj1.items():
        if key not in json_obj2:
            return False

        value2 = json_obj2[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_key_value_pairs(value1, value2):
                return False
        elif not (isinstance(value1, dict) or isinstance(value2, dict)):
            continue
        else:
            return False

    return True

# Add json check and repair json
def validate_and_fix_json(json_data, schema):
    try:
        jsonschema.validate(json_data, schema)
        print("JSON data is valid")
    except jsonschema.exceptions.ValidationError as e:
        print("Invalid JSON data：", e)
        print("trying to fix...")
        json_data = jsonschema.validators.ref_resolver.RefResolver('', json_data).resolve(json_data)
        print("Repaired JSON data：", json.dumps(json_data, indent=2))


# Defines the schema for JSON data
schema = {
    "type": "object",
    "patternProperties": {
        "^@Page [0-9]+$": {
            "type": "object",
            "patternProperties": {
                "^[0-9]+$": {"type": "string"}
            }
        }
    }
}


# 补}
# def check_string_end(s):
#     # Use regex to find strings ending with "}" and allow spaces, tabs or newlines between two "}"s
#     pattern = r'\}\s*\}\s*$'
#
#     # Whether the search string matches the pattern
#     match = re.search(pattern, s)
#
#     if match:
#         return True
#     else:
#         return False

# Check format supplement after adding summary}
def check_string_end(s):
    # Use regex to find strings ending with "}" and allow spaces, tabs or newlines between two "}"s
    pattern = r'\}\s*$'

    # Whether the search string matches the pattern
    match = re.search(pattern, s)

    if match:
        return True
    else:
        return False



def your_chatgpt_api_call(conn, input_text):
    print("current translated text：")
    print(input_text)

    # First filter out double quotes, which are easy to cause JSON errors
    input_text = input_text.replace("\"", " ")

    print("Convert to JSON format：")
    pages = re.findall(r'@Page (\d+)，(\d+) sentences in total\.(.*?)@Page \1 End', input_text, re.DOTALL)

    # get summary
    cur = conn.cursor()
    cur.execute("select summary from manga_summary WHERE rowid=1", )
    # conn.commit()
    row = cur.fetchone()
    summary = row[0]
    if summary:
        summary = "前文摘要："+summary


    json_obj = {}

    # for page in pages:
    #     page_key = f"@Page {page[0]}"
    #     json_obj[page_key] = {}
    #     sentences = re.findall(r'(\d+)\.(.*?)\n', page[2], re.DOTALL)
    #
    #     for sentence in sentences:
    #         json_obj[page_key][sentence[0]] = sentence[1].strip()
    #         # 强制分段保证数据格式正确
    #         # json_obj[page_key][sentence[0]] = sentence[1].strip()+"\n"

    # 让键更详细
    for page in pages:
        page_key = f"@Page {page[0]}"
        json_obj[page_key] = {}
        sentences = re.findall(r'(\d+)\.(.*?)\n', page[2], re.DOTALL)

        for sentence in sentences:
            # json_obj[page_key]["@IndependentDialogue "+str(sentence[0])] = "[No need for complete sentences and translate separately to Chinese without merging]#START# " + sentence[1].strip().replace(" ", "") + " #END#[stop translate]"
            json_obj[page_key]["@IndependentDialogue "+str(sentence[0])] = "[No need for complete sentences and translate separately to Chinese without merging] " + sentence[1].strip().replace(" ", "") + " [stop translate]"

    # 塞入需要提取摘要
    json_obj["@Summary"] = "[Please summarize the new translated content into a short sentence and put it in this line]"
    # 缓存原文json用作对比
    global temp_origin_words_json
    temp_origin_words_json = json_obj

    print(json.dumps(json_obj, ensure_ascii=False, indent=2))

    print(f"塞入的Summary:{summary}")

    resText = ""

    while True:
        try:
            # openai.api_requestor.timeout = 300
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                # model='gpt-4',
                messages=[
                    {"role": "system", "content": "你是一个大师级的漫画编剧。"},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": summary},
                    {"role": "user", "content": json.dumps(json_obj, ensure_ascii=False, indent=2)}
                ],
                temperature=0,
                top_p=0.9,
                frequency_penalty=frequency_penalty_current,
                timeout=2400,
                request_timeout=600,
            )
            # 返回信息
            resText = response.choices[0].message.content
            break
        except Exception as ec:
            print(ec)
            print("请求遇到问题，重试！")

    return resText

# json修复
def remove_invalid_comma(json_string):
    # 移除对象中的多余逗号
    json_string = re.sub(r',\s*}', '}', json_string)
    # 移除数组中的多余逗号
    # json_string = re.sub(r',\s*]', ']', json_string)
    return json_string
def get_translation(conn, input_text):
    # Call your ChatGPT 3.5 API here and get the response
    response = your_chatgpt_api_call(conn, input_text)

    response = re.sub('\[.*?\]', '', response)
    response = re.sub('\<.*?\>', '', response)



    # 尝试修补}
    if not check_string_end(response):
        print("尝试修补第一个括号")
        response = response + "}"

    # if not check_string_end(response):
    #     print("尝试修补第二个括号")
    #     response = response + "}"

    print("得到翻译文本：")
    print(response)




    # Replace this line with the actual method to extract translations from your ChatGPT 3.5 response
    # translations = response.text.strip().split('\n')
    # translations = response.strip().split('\n')
    # 老代码
    # response = remove_invalid_comma(response)
    # json_obj = json.loads(response)
    json_obj = json5.loads(response)
    json_obj = json.loads(json.dumps(json_obj))

    global temp_trans_words_json
    temp_trans_words_json = json_obj


    # json_obj = json.loads(json.dumps(response))

    translations = []

    # for page, content in json_obj.items():
    #     page_str = "\n".join(content.values())
    #     translations.append(page_str)
    # 增加空字符串检测，遇到空抛出异常缩减规模重试
    # for page, content in json_obj.items():
    #     # 检查所有值是否都不为空字符串
    #     # if all(value != '' for value in content.values()):
    #     # 要把空格也排除
    #     if all(value and not value.isspace() for value in content.values()):
    #         page_str = "\n".join(content.values())
    #         translations.append(page_str)
    #     else:
    #         raise ValueError(f"在页码 {page} 中，存在空字符串。")

    # 排除summary拼接
    for page, content in json_obj.items():
        # Exclude the "summary" field
        if page == "@Summary":
            continue

        # Check that all values are not empty strings or spaces
        if all(value and not value.isspace() for value in content.values()):
            page_str = "\n".join(content.values())
            translations.append(page_str)
        else:
            raise ValueError(f"Page {page} contains empty or whitespace-only strings.")

    # 获取摘要更新
    # summary = json_obj.get("@Summary", "")
    summary = json_obj["@Summary"]
    print("当前摘要信息：")
    print(summary)
    conn.execute("UPDATE manga_summary SET summary=? WHERE rowid=1", (summary, ))
    conn.commit()

    # pattern = r'@Msg.*?@Msgend'
    # translations = re.findall(pattern, response.strip(), flags=re.DOTALL)

    return translations


def update_translations(conn, translations, starting_row):
    for index, translation in enumerate(translations):
        conn.execute("UPDATE manga_page SET trans=? WHERE rowid=?", (translation, starting_row + index))
    conn.commit()

# 查错函数
def update_translations2(conn, translations, row_ids):
    for index, translation in enumerate(translations):
        conn.execute("UPDATE manga_page SET trans=? WHERE rowid=?", (translation, row_ids[index]))
    conn.commit()

def process_translations(translations):
    need_re = []
    for index, translation in enumerate(translations):
        new_translation = translation.replace(' ', '')
        if new_translation.count('"') != 2:
            need_re.append(index + 1)
        translations[index] = new_translation
    return need_re

def split_need_re(need_re):
    return [need_re[i:i + 10] for i in range(0, len(need_re), 10)]


conn = sqlite3.connect('manga_page.db')
cur = conn.cursor()

cur.execute('UPDATE page_count SET page=1 WHERE rowid=1')
conn.commit()

# 测试变量化操作
batch_size_default = 5
batch_size = batch_size_default

while True:
    cur.execute("SELECT rowid, words FROM manga_page WHERE trans IS NULL LIMIT 1")
    row = cur.fetchone()
    if row is None:
        break

    starting_row = row[0]

    print("当前位置："+str(starting_row))

    cur.execute("SELECT words FROM manga_page WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
                # (starting_row, starting_row + 19))
                # (starting_row, starting_row + 4))
                (starting_row, starting_row + batch_size - 1))

    # words = '\n'.join([word[0] for word in cur.fetchall()])

    # 执行数据库查询并获取结果集合
    result_set = cur.fetchall()

    # 创建一个列表，包含每个单词的第一个字符
    word_array = [word[0] for word in result_set]
    # print(len(word_array))
    # 将列表转换为字符串，每个元素占一行
    words = '\n'.join(word_array)

    # 在其他地方使用word_array
    # ...

    try:
        translations = get_translation(conn, words)

        if not compare_key_value_pairs(temp_origin_words_json, temp_trans_words_json):
            print("两个 JSON 对象的格式不一致，重试")
            batch_size = batch_size - 2
            print("修改batch_size为" + str(batch_size) + "继续尝试")
            if batch_size < 0:
                print("输出卡死，强制中断")
                break
            continue

        # 用于结尾
        if len(word_array) < batch_size:
            print("当前未翻译页数少于batch_size自动调整为:"+str(len(word_array)))
            update_translations(conn, translations, starting_row)
            break


        # if len(translations) == 5:
        if len(translations) == batch_size:
            update_translations(conn, translations, starting_row)
            # starting_row += 5
            starting_row += batch_size
            batch_size = batch_size_default
        else:
            batch_size = batch_size - 2
            print("修改batch_size为" + str(batch_size) + "继续尝试")
            if batch_size < 0:
                print("输出卡死，强制中断")
                break
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        batch_size = batch_size - 2
        print("出现解析错误，修改batch_size为"+str(batch_size)+"继续尝试")
        if batch_size < 0:
            print("输出卡死，强制中断")
            break





conn.close()
