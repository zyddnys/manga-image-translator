import transformers
import os

_SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))


class deepseekTokenCounter():
    def __init__(self):
        chat_tokenizer_dir = os.path.join(_SCRIPT_DIR, 'deepseek')
        # Initialize the tokenizer:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained( 
                        chat_tokenizer_dir, trust_remote_code=True
                    )
        
    def count_tokens(self, text: str) -> int:
        """
            Count tokens using the deepseek tokenizer
            Tokenizer downloaded from:
            https://api-docs.deepseek.com/quick_start/token_usage
        """

        return len(self.tokenizer.encode(text))
