import os
from dotenv import load_dotenv

current_directory = os.path.dirname(os.path.abspath(__file__))
load_dotenv(
    os.path.join(current_directory, "../../../../config/env.gpu"), override=False, verbose=True
)

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '') #你的appid
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', '') #你的密钥
# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY', '') # 应用ID
YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY', '') # 应用秘钥
# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY', '') #YOUR_AUTH_KEY
# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY') # TODO: Replace with --proxy

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1') #使用api-for-open-llm例子 http://127.0.0.1:8000/v1

# sakura
SAKURA_API_BASE = os.getenv('SAKURA_API_BASE', 'http://127.0.0.1:8080/v1') #SAKURA API地址
SAKURA_VERSION = os.getenv('SAKURA_VERSION', '0.9') #SAKURA API版本，可选值：0.9、0.10，选择0.10则会加载术语表。
SAKURA_DICT_PATH = os.getenv('SAKURA_DICT_PATH', './sakura_dict.txt') #SAKURA 术语表路径


CAIYUN_TOKEN = os.getenv('CAIYUN_TOKEN', '') # 彩云小译API访问令牌

# deepseek
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_BASE  = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')

# ollama, with OpenAI API compatibility
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY', 'ollama') # Unsed for ollama, but maybe useful for other LLM tools.
OLLAMA_API_BASE = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/v1') # Use OLLAMA_HOST env to change binding IP and Port.
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', '') # e.g "qwen2.5:7b". Make sure to pull and run it before use.
