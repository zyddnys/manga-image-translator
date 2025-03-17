import asyncio
import json
import re
from abc import abstractmethod
import time

from .config_gpt import ConfigGPT, TextValue, TranslationList
from .common import CommonTranslator, VALID_LANGUAGES
from typing import List, Dict



class CommonGPTTranslator(ConfigGPT, CommonTranslator):
    """
    A base class for GPT-based translators, providing common functionality
    such as prompt assembly and response parsing.
    
    Inherits from both `ConfigGPT` and `CommonTranslator`
    

    Attributes:
        _LANGUAGE_CODE_MAP (dict): A dictionary mapping language codes to
            language names.  Assumes that GPT translators support all languages
        _MAX_TOKENS_IN (int): The maximum number of input tokens allowed
            per query. Defaults to half of `_MAX_TOKENS` if not specified.

    Abstract Methods
    ----------------
        `count_tokens`
            Parent classes must provide a way to count the tokens, to allow for batch-chunking.
            See: `tokenizers/token_counters.py` for example implementations.
            
            See the `Notes` section of the abstract definition for fall-back \
                solutions when obtaining the true token count is not feasible.
    """
    
    _LANGUAGE_CODE_MAP=VALID_LANGUAGES # Assume that GPT translators support all languages

    def __init__(self, config_key: str):
        """
        Initializes the CommonGPT translator with configurations and token limits.
        Args:
            config_key (str): The configuration key to use for parsing the `config_gpt` file.
        """

        ConfigGPT.__init__(self, config_key=config_key)
        CommonTranslator.__init__(self)
        
        # `_MAX_TOKENS` indicates the maximum output tokens.
        #   Unless specified otherwise: 
        #       Limit input tokens per query to 1/2 max output
        try:
            self._MAX_TOKENS_IN
        except:
            self._MAX_TOKENS_IN = self._MAX_TOKENS//2

    def parse_args(self, args: CommonTranslator):
        self.config = args.chatgpt_config

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text string.

        This method should be implemented using the appropriate tokenizer for the
        GPT model being used to accurately measure the number of tokens
        that will be sent to the API.

        return len(text) will be safe in most scenarios

        Args
        ----
            text (str): The input text string.

        Returns
        -------
            int: The estimated number of tokens in the text.
 
        Notes
        -----
        If unable to implement fully to get a true token count:
        
        - In most insances, simply counting char-length will be a sufficiently \
            safe over-estimation: 
        ```
        def count_tokens(text: str):
            return len(text)
        ```
        
        - If you wish to check for an upper-bound limit: A ratio of \
            `1 token` / `UTF-8 byte` is generally safe for most modern tokenizers
        ```
        def count_tokens(text: str):
            return len(text.encode('utf-8'))
        ```
        
        """
        
        pass
        
        
    def withinTokenLimit(self, text: str) -> bool:
        """
        Simple helper function to check if `text` has a token count
            less-than/equal-to `_MAX_TOKENS_IN`.
             
        First checks assuming worst-case-scenario of 1 token per utf-8 byte,
            short-circuiting if string length is less-than/equal-to `_MAX_TOKENS_IN`
        
        Falls through to using the token counter class to count the actual tokens.


        Args:
            text (str): The text to check.

        Returns:
            bool: 
                True if `text` token length is less-than/equal-to `_MAX_TOKENS_IN`
            
                False if `text` token length is greater-than `_MAX_TOKENS_IN`
        """
        if len(text.encode('utf-8')) <= self._MAX_TOKENS_IN:
            return True
        
        return self.count_tokens(text) <= self._MAX_TOKENS_IN


    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        self.to_lang=to_lang
        self.from_lang=from_lang
        return True

    def fallback_fewShot(self) -> str:
        """
        Generates a few-shot example string for the GPT model.
            
        If the translator does not natively support input / output examples, this 
        formats the examples as a string, to attached to the prompt, formatted as:

            <EXAMPLE>
            INPUT: {input_text}
            
            OUTPUT: {output_text}
            </EXAMPLE>
        
        Returns:
            str: A string containing the few-shot example or `None` If no sample is available
        """
        fewshot=None

        lang_chat_samples = self.get_sample(self.to_lang)

        # 如果需要先给出示例对话
        # Add chat samples if available
        if lang_chat_samples:
            fewshot="<EXAMPLE>\n"
            fewshot+=f"  INPUT:{lang_chat_samples[0]}\n"
            fewshot+=f"  \n"
            fewshot+=f"  OUTPUT:{lang_chat_samples[1]}\n"
            fewshot+="</EXAMPLE>\n"

        return fewshot

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        """
        原脚本中用来把多个 query 组装到一个 Prompt。
        同时可以做长度控制，如果过长就切分成多个 prompt。
        
        Original script's method to assemble multiple queries into prompts.
        Handles length control by splitting long queries into multiple prompts.
        """
        batch = []          # List [ <queries> ]
        chunk_queries = []  # List [ List [ <queries> ] ] 
        current_length = 0

        def _list2prompt(queryList=List[str]):
            prompt = ""
            if self.include_template:
                prompt = self.prompt_template.format(to_lang=to_lang)
            
            # 加上分行内容
            # Add line breaks
            for id_num, query in enumerate(queryList, start=1):
                prompt += f"\n<|{id_num}|>{query.strip()}"

            return prompt            

        # Test if batching is necessary
        #   Chunking is likely only necessary in edge-cases 
        #       (small token limit or huge amounts of text)
        #   
        #   Checking if it is required should reduce workload and minimize
        #       repeated `count_token` queries (which is not always be done locally)
        prompt=_list2prompt(queries)
        if self.withinTokenLimit(prompt):
            yield prompt, len(queries)
        else:
            # Buffer for ID tag prepended to each query. 
            # Assume 1 token per char (worst case scenario)
            # 
            # - Use `len(queries)` to get max digit count
            #   (i.e. 0-9 => 1, 10-99 => 2, 100-999 => 3, etc.)
            IDTagBuffer=len(f"\n<|{len(queries)}|>")
            
            for q in queries:
                qTokens=self.count_tokens(q) + IDTagBuffer

                if batch and ( (current_length + qTokens) > self._MAX_TOKENS_IN):
                    # 输出当前 batch
                    # Output current batch
                    chunk_queries.append(batch)
                    batch = []
                    current_length = 0
                
                batch.append(q)
                current_length += qTokens
            if batch:
                chunk_queries.append(batch)

            # 逐个批次生成 prompt
            # Generate prompts batch by batch
            for this_batch in chunk_queries:
                prompt = _list2prompt(this_batch)
                
                yield prompt.lstrip(), len(this_batch)
    
    def _assemble_request(self, to_lang: str, prompt: str) -> Dict:
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]

        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        messages.append({'role': 'user', 'content': prompt})

        # Arguments for the API call:
        kwargs = {
            "model": self.MODEL,
            "messages": messages,
            "max_tokens": self._MAX_TOKENS // 2,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self._TIMEOUT
        }

        return kwargs
    

    def _parse_response(self, response: str, queries: List):
        # Split response into translations  
        new_translations = re.split(r'<\|\d+\|>', response)  
        if not new_translations[0].strip():  
            new_translations = new_translations[1:]  

        if len(queries) == 1 and len(new_translations) == 1 and not re.match(r'^\s*<\|\d+\|>', response):  
            raise Warning('Single query response does not contain prefix.')  
        
        return new_translations

    async def _ratelimit_sleep(self):
        """
        在请求前先做一次简单的节流 (如果 _MAX_REQUESTS_PER_MINUTE > 0)。
        Simple rate limiting before requests (if _MAX_REQUESTS_PER_MINUTE > 0).
        """
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            delay = 60.0 / self._MAX_REQUESTS_PER_MINUTE
            elapsed = now - self._last_request_ts
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self._last_request_ts = time.time()



class _CommonGPTTranslator_JSON:
    import pprint
    from os import get_terminal_size
    
    """Internal helper class for JSON mode logic"""

    def __init__(self, translator: CommonGPTTranslator):
        self.translator = translator



    def ppJSON(self, jsonText: str) -> str:
        """ 
        Helper function to PrettyPrint format a JSON string
        
        Args:
            jsonText (str): The JSON string to format.
        Returns:
            str: A pretty-printed string representation of the JSON object.
        """
      

        # By default: pformat sets line width to 80 chars. 
        # Get terminal width to override (with buffer of 10 chars)
        WIDTH=(self.get_terminal_size().columns - 10)
        
        return self.pprint.pformat( object=self.json.loads(jsonText), 
                                    width=WIDTH
                                )
    

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        """
        原脚本中用来把多个 query 组装到一个 Prompt。
        同时可以做长度控制，如果过长就切分成多个 prompt。

        Original script's method to assemble multiple queries into prompts.
        Handles length control by splitting long queries into multiple prompts.
        """
        batch = []          # List [ <queries> ]
        chunk_queries = []  # List [ List [ <queries> ] ] 
        input_ID = 0
        
        # Test if batching is necessary
        #   Chunking is likely only necessary in edge-cases 
        #       (small token limit or huge amounts of text)
        #
        #   Checking if it is required should reduce workload and minimize
        #       repeated `count_token` queries (which is not always be done locally)
        testFull=self._list2json(queries)
        if self.translator.withinTokenLimit(testFull.model_dump_json()):
            yield testFull.model_dump_json(), len(testFull.TextList)
        else:
            for input_text in queries:
                # temp list, to check if it exceeds token limit:
                temp_list = batch + [TextValue(ID=input_ID, text=input_text)]
                temp_json = TranslationList(TextList=temp_list).model_dump_json()

                if self.translator.withinTokenLimit(temp_json):
                    # Commit value to current batch
                    batch = temp_list
                    input_ID += 1
                else:
                    # If there are values in the batch, add batch to chunk list
                    if batch:
                        chunk_queries.append(TranslationList(TextList=batch))

                    # Start new chunk with current item (even if it exceeds limit)
                    batch = [TextValue(ID=0, text=input_text)]
                    
                    # Reset ID counter for new chunk
                    input_ID = 0
        
            if batch:
                chunk_queries.append(TranslationList(TextList=batch))

            # 逐个批次生成 JSON
            # Generate JSON batch by batch
            for this_batch in chunk_queries:
                yield this_batch.model_dump_json(), len(this_batch.TextList)

    def _assemble_request(self, to_lang: str, prompt: str, response_format=True) -> Dict:
        messages = [{'role': 'system', 'content': self.translator.chat_system_template.format(to_lang=to_lang)}]
        
        jSample=self.translator.get_json_sample(to_lang)
        if jSample:
            messages.append({'role': 'user', 'content': jSample[0].model_dump_json()})
            messages.append({'role': 'assistant', 'content': jSample[1].model_dump_json()})
        else:
            # If no appropriate `json_sample` is available, but a `chat_sample` is found: 
            #   Convert and use the `chat_sample`
            chatSample=self.translator.chat_sample.get(to_lang)
            if chatSample:
                asJSON = [
                    self.text2json(self.translator.chat_sample[0]).model_dump_json(),
                    self.text2json(self.translator.chat_sample[1]).model_dump_json()
                ]

                messages.append({'role': 'user', 'content': asJSON[0]})
                messages.append({'role': 'assistant', 'content': asJSON[1]})


        messages.append({'role': 'user', 'content': prompt})

        # Arguments for the API call:
        kwargs = {
            "model": self.translator.MODEL,
            "messages": messages,
            "max_tokens": self.translator._MAX_TOKENS,
            "temperature": self.translator.temperature,
            "top_p": self.translator.top_p,
            "timeout": self.translator._TIMEOUT,
            "response_format": TranslationList
        }

        # Fallback to providing schema info via System prompt if `response_format` is disabled
        if not response_format:
            # Remove the response_format key from API call
            kwargs.pop("response_format")
            
            # Append JSON schema specification to the System message:
            SYS_FMT="\nRespond only with JSON matching this JSON schema:\n" 
            SYS_FMT+=str(TranslationList.model_json_schema())
            kwargs["messages"][0]["content"] += SYS_FMT

        return kwargs

    def _parse_response(self, response: json, queries: List[str]) -> List[str]:
        """
        Parses a JSON response from the API and maps translations to their respective positions.

        Args:
            response (json): The JSON response from the API.
            queries (List[str]): The original input values

        Returns:
            List[str]: A list of translations in the same order as the input queries.
                       If a translation is missing, the original query is preserved.
        """
        translations = queries.copy()  # Initialize with the original queries
        expected_count = len(translations)

        try:
            # Parse the JSON response
            response_data = json.loads(response)
            # Validate the JSON structure
            if not isinstance(response_data, dict) or "TextList" not in response_data:
                raise ValueError("Invalid JSON structure: Missing 'TextList' key")

            translated_items = response_data["TextList"]

            # Validate that 'TextList' is a list
            if not isinstance(translated_items, list):
                raise ValueError("Invalid JSON structure: 'TextList' must be a list")

            rangeOffset = min([val['ID'] for val in translated_items])
            expected_max = (expected_count - 1) + rangeOffset
            
            # Process each translated item
            for item in translated_items:
                # Validate item structure
                if not isinstance(item, dict) or "ID" not in item or "text" not in item:
                    raise ValueError("Invalid translation item: Missing 'ID' or 'text'")

                id_num = item["ID"]
                translation = item["text"].strip()

                # Check if the ID is within the expected range
                if (id_num < 0) or (id_num > expected_max):
                    raise ValueError(f"ID {id_num} out of range (expected 0 to {expected_max})")

                # Update the translation at the correct position
                translations[id_num - rangeOffset] = translation

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e

        return translations
 
    def text2json(text: str) -> TranslationList:
        """
        Convert text samples to TranslationList format.
        Assists with backwards compatiblity for `<|ID|>`-based samples.
        
        Args:
            input_data: Text samples, keyed by `<|ID|>` tags
            
        Returns:
            Text samples stored as a TranslationList
        """

        segment_pattern = re.compile(r'<\|(\d+)\|>(.*?)(?=<\|(\d+)\|>|$)', re.DOTALL)
        segments = segment_pattern.findall(text)

        jsonified=TranslationList(
                            TextList=[
                                TextValue(
                                    ID=int(seg[0]),
                                    text=seg[1].strip()
                                ) for seg in segments
                            ]
                        )

        return jsonified

    def _list2json(self, vals: List[str]) -> TranslationList:
        """
        Convert list text values to TranslationList format.
        
        Args:
            input_data: List of text samples
            
        Returns:
            Text samples stored as a TranslationList
        """

        jsonified=TranslationList(
                            TextList=[
                                TextValue(
                                    ID=id_num,
                                    text=line.strip()
                                ) for id_num, line 
                                    in enumerate(vals)
                            ]
                        )

        return jsonified
