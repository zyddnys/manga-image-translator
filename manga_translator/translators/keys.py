import os

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '') #你的appid
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', '') #你的密钥
# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY', '') # 应用ID
YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY', '') # 应用秘钥
# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY', '') #YOUR_AUTH_KEY
# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY') # TODO: Replace with --proxy
# caiyun
CAIYUN_TOKEN = os.getenv('CAIYUN_TOKEN', '') # 彩云小译API访问令牌
