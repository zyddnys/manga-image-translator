import logging
import colorama

from .general import replace_prefix

ROOT_TAG = 'manga-translator'

class Formatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            self._style._fmt = f'{colorama.Fore.RED}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno >= logging.WARN:
            self._style._fmt = f'{colorama.Fore.YELLOW}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        else:
            self._style._fmt = '[%(name)s] %(message)s'
        return super().formatMessage(record)

class Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Try to filter out logs from imported modules
        if not record.name.startswith(ROOT_TAG):
            return False
        # Shorten the name
        record.name = replace_prefix(record.name, ROOT_TAG + '.', '')
        return super().filter(record)

logging.basicConfig(level=logging.INFO)
root = logging.getLogger(ROOT_TAG)

for h in logging.root.handlers:
    h.setFormatter(Formatter())
    h.addFilter(Filter())

def set_log_level(level):
    root.setLevel(level)

def get_logger(name: str):
    return root.getChild(name)
