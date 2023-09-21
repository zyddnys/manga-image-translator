import os

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '20200814000543070') #你的appid
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', 'GZWWEwgVh2a0OL7itzA6') #你的密钥
# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY', '5077fb725e38c9d3') # 应用ID
YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY', 'BHyiUU3c3ITyBNNaNzsaxnMnpamuePNo') # 应用秘钥
# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY', 'ff0adcf6-f119-6fcf-d084-d51a2df9aa47') #YOUR_AUTH_KEY
# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY') # TODO: Replace with --proxy
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'http://api.openai.com/v1')

CAIYUN_TOKEN = ''
