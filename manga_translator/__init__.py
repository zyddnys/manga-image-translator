import colorama
from dotenv import load_dotenv

colorama.init(autoreset=True)
load_dotenv()

from .manga_translator import *
