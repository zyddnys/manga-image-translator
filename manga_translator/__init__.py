import colorama
import logging
# from rich.logging import RichHandler

colorama.init(autoreset=True)

class Formatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.ERROR:
            self._style._fmt = f'{colorama.Fore.RED}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno == logging.WARN:
            self._style._fmt = f'{colorama.Fore.YELLOW}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        else:
            self._style._fmt = '[%(name)s] %(message)s'
        return super().formatMessage(record)

# logging.basicConfig(format='[%(name)s] %(message)s', level=logging.INFO, handlers=[RichHandler()])
logging.basicConfig(level=logging.INFO)

for h in logging.root.handlers:
    h.setFormatter(Formatter())

# Limit asyncio logger
logging.getLogger('asyncio').setLevel(logging.WARNING)

from .manga_translator import *
