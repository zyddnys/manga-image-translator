import os
from dotenv import load_dotenv
load_dotenv()

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '20200814000543070') #你的appid
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', 'GZWWEwgVh2a0OL7itzA6') #你的密钥
# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY', '5077fb725e38c9d3') # 应用ID
YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY', 'BHyiUU3c3ITyBNNaNzsaxnMnpamuePNo') # 应用秘钥
# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY', 'fe2175e3-d834-930c-229e-fc7e9839ac49') #YOUR_AUTH_KEY
# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY') # TODO: Replace with --proxy

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1') #使用api-for-open-llm例子 http://127.0.0.1:8000/v1

CAIYUN_TOKEN = os.getenv('CAIYUN_TOKEN', '') # 彩云小译API访问令牌