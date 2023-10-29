import os
from dotenv import load_dotenv
load_dotenv()

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID')
if BAIDU_APP_ID is None:
    raise ValueError("BAIDU_APP_ID is not set in the environment.")

BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY')

# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY')
if YOUDAO_APP_KEY is None:
    raise ValueError("YOUDAO_APP_KEY is not set in the environment.")

YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY')

# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY')
if DEEPL_AUTH_KEY is None:
    raise ValueError("DEEPL_AUTH_KEY is not set in the environment.")

# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY')  # TODO: Replace with --proxy

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')  # 使用api-for-open-llm例子 http://127.0.0.1:8000/v1

CAIYUN_TOKEN = os.getenv('CAIYUN_TOKEN')
if CAIYUN_TOKEN is None:
    raise ValueError("CAIYUN_TOKEN is not set in the environment.")
