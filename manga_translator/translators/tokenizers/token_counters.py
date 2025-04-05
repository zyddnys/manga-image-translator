import os

_SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))


class deepseekTokenCounter():
    """
    A class to count tokens using the DeepSeek tokenizer.
    This class initializes with the DeepSeek tokenizer (loaded from local subdirectory).
    
    Tokenizer downloaded from:
    https://api-docs.deepseek.com/quick_start/token_usage
    
    Methods:
        count_tokens(text:str) -> int Count the number of tokens within string `text`
    """
    import transformers
    
    def __init__(self):
        chat_tokenizer_dir = os.path.join(_SCRIPT_DIR, 'deepseek')
        # Initialize the tokenizer:
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained( 
                            chat_tokenizer_dir, trust_remote_code=True
                        )
        
    def count_tokens(self, text: str) -> int:
        """
            Count tokens using the deepseek tokenizer
        """

        return len(self.tokenizer.encode(text))


class ChatGPTTokenCounter():
    """
    A class for counting tokens in text using the tiktoken library, specifically designed for OpenAI models.
        Attributes:
            tokenizer: The tiktoken encoder object used for tokenizing text.
        Methods:
            _get_encoder_for_model(CHATGPT_MODEL: str) -> str: Get the token encoder used by  for a given model
            count_tokens(text: str) -> int: Count the number of tokens within string `text`
    """

    import tiktoken

    def __init__(self, CHATGPT_MODEL: str):
        self.tokenizer = self._get_encoder_for_model(CHATGPT_MODEL)

    def _get_encoder_for_model(self, CHATGPT_MODEL: str) -> str:
        """
            Get the appropriate tiktoken encoder for a given OpenAI model.
            Args:
                CHATGPT_MODEL (str): The name of the OpenAI model.
            Returns:
                Encoding: The tiktoken encoder object. Defaults to "cl100k_base" if the model is unknown.
        """

        try:
            # Use tiktoken's built-in mapping for OpenAI models
            encoding = self.tiktoken.encoding_for_model(CHATGPT_MODEL)
            return encoding
        except KeyError:
            # Fallback for unknown OpenAI models
            return self.tiktoken.get_encoding("cl100k_base")


    def count_tokens(self, text: str) -> int:
        """
            Count tokens using the specified encoding.
        """
        return len(self.tokenizer.encode(text))
