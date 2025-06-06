import os, re, asyncio, base64
from io import BytesIO
from typing import List
from collections import Counter
from pydantic import BaseModel, Field
from loguru import logger
from openai import OpenAI
from PIL import Image
from manga_translator.utils import is_valuable_text
from .common import CommonTranslator
from ..utils import Context
from .keys import GEMINI_API_KEY, GEMINI_MODEL, TOGETHER_API_KEY, TOGETHER_VL_MODEL


def encode_image(image):
    max_dim = 1024
    w, h = image.size
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8'), new_w, new_h


class TextBoundingBox(BaseModel):
    bbox_id: int = Field(description="ID of the bounding box")
    bbox_2d: list[int] = Field(description="Bounding Box coordinates in the format [x1, y1, x2, y2]")
    text: str = Field(description='Original Text')
    corrected_text: str = Field(description="Corrected Text")


class TextBoundingBoxes(BaseModel):
    bboxes: list[TextBoundingBox] = Field(description="List of Bounding Boxes with Corrected Text")


class TranslatedText(BaseModel):
    text_id: int = Field(description="ID of the Text")
    text: str = Field(description='Original Text')
    translated_text: str = Field(description="Translated Text")


class TranslatedTexts(BaseModel):
    translated_texts: list[TranslatedText] = Field(description="List of Translated Texts")


class Gemini2StageTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CN': 'Chinese', 'CHS': 'Simplified Chinese', 'CHT': 'Traditional Chinese',
        'CSY': 'Czech', 'NLD': 'Dutch', 'ENG': 'English', 'FRA': 'French',
        'DEU': 'German', 'HUN': 'Hungarian', 'ITA': 'Italian', 'JPN': 'Japanese',
        'KOR': 'Korean', 'PLK': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian',
        'RUS': 'Russian', 'ESP': 'Spanish', 'TRK': 'Turkish', 'UKR': 'Ukrainian',
        'VIN': 'Vietnamese', 'CNR': 'Montenegrin', 'SRP': 'Serbian', 'HRV': 'Croatian',
        'ARA': 'Arabic', 'THA': 'Thai', 'IND': 'Indonesian'
    }
    _INVALID_REPEAT_COUNT = 0
    _MAX_REQUESTS_PER_MINUTE = -1
    _LANG_PATTERNS = [
        ('JPN', r'[\u3040-\u309f\u30a0-\u30ff]'),
        ('KOR', r'[\uac00-\ud7af\u1100-\u11ff]'),
        ('CN', r'[\u4e00-\u9fff]'),
        ('ARA', r'[\u0600-\u06ff]'),
        ('THA', r'[\u0e00-\u0e7f]'),
        ('RUS', r'[\u0400-\u04ff]')
    ]
    _LEFT_SYMBOLS = ['(', '（', '[', '【', '{', '〔', '〈', '「', '"', "'", '《', '『', '"', '〝', '﹁', '﹃', '⸂', '⸄', '⸉', '⸌',
                     '⸜', '⸠', '‹', '«']
    _RIGHT_SYMBOLS = [')', '）', ']', '】', '}', '〕', '〉', '」', '"', "'", '》', '』', '"', '〞', '﹂', '﹄', '⸃', '⸅', '⸊',
                      '⸍', '⸝', '⸡', '›', '»']

    def __init__(self, max_tokens = 16000, refine_temperature = 0.0, translate_temperature = 0.1):
        super().__init__()
        self.client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
        self.client2 = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.refine_model, self.translate_model = TOGETHER_VL_MODEL, GEMINI_MODEL
        self.max_tokens = max_tokens
        self.refine_temperature, self.translate_temperature = refine_temperature, translate_temperature
        self.refine_response_schema, self.translate_response_schema = TextBoundingBoxes, TranslatedTexts

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src = ['auto'] + list(self._LANGUAGE_CODE_MAP.keys())
        supported_tgt = list(self._LANGUAGE_CODE_MAP.keys())
        if from_lang not in supported_src or to_lang not in supported_tgt:
            if fatal: raise NotImplementedError
            return False
        return True

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], ctx: Context, use_mtpe: bool = False) -> \
    List[str]:
        if not queries: return queries

        if from_lang == 'auto':
            from_langs = []
            for region in ctx.text_regions:
                for lang, pattern in self._LANG_PATTERNS:
                    if re.search(pattern, region.text):
                        from_langs.append(lang)
                        break
                else:
                    from_langs.append('ENG')
            from_lang = Counter(from_langs).most_common(1)[0][0]

        from_lang, to_lang = self._LANGUAGE_CODE_MAP.get(from_lang), self._LANGUAGE_CODE_MAP.get(to_lang)
        if from_lang == to_lang: return queries

        query_indices, final_translations = [], []
        for i, q in enumerate(queries):
            final_translations.append(queries[i] if not is_valuable_text(q) else None)
            if is_valuable_text(q): query_indices.append(i)

        queries = [queries[i] for i in query_indices]
        translations = [''] * len(queries)
        untranslated_indices = list(range(len(queries)))

        for i in range(1 + self._INVALID_REPEAT_COUNT):
            if i > 0:
                self.logger.warn(f'Repeating because of invalid translation. Attempt: {i + 1}')
                await asyncio.sleep(0.1)

            await self._ratelimit_sleep()
            _translations = await self._translate(from_lang, to_lang, query_indices, ctx)

            _translations += [''] * (len(queries) - len(_translations))
            _translations = _translations[:len(queries)]

            for j in untranslated_indices:
                translations[j] = _translations[j]

            if self._INVALID_REPEAT_COUNT == 0: break

            new_untranslated = []
            for j in untranslated_indices:
                if self._is_translation_invalid(queries[j], translations[j]):
                    new_untranslated.append(j)
                    queries[j] = self._modify_invalid_translation_query(queries[j], translations[j])
            untranslated_indices = new_untranslated

            if not untranslated_indices: break

        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]

        if to_lang == 'ARA':
            import arabic_reshaper
            translations = [arabic_reshaper.reshape(t) for t in translations]

        if use_mtpe:
            translations = await self.mtpe_adapter.dispatch(queries, translations)

        for i, trans in enumerate(translations):
            final_translations[query_indices[i]] = trans
            self.logger.info(f'{i}: {queries[i]} => {trans}')

        return final_translations

    async def _translate(self, from_lang: str, to_lang: str, query_indices: List[int], ctx: Context) -> List[str]:
        return await self._translate_2stage(from_lang, to_lang, query_indices, ctx)

    async def _translate_2stage(self, from_lang: str, to_lang: str, query_indices: List[int], ctx: Context) -> List[
        str]:
        rgb_img = Image.fromarray(ctx.img_rgb)
        w, h = rgb_img.size
        query_regions = [ctx.text_regions[i] for i in query_indices]

        base64_img, nw, nh = encode_image(rgb_img)
        refine_prompt = self.get_prompt(query_regions, w, h, nw, nh)

        response = self.client.beta.chat.completions.parse(
            model=self.refine_model,
            messages=[
                {"role": "system", "content": self._get_refine_system_instruction(from_lang)},
                {"role": "user", "content": [
                    {"type": "text", "text": refine_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
            temperature=self.refine_temperature,
            max_completion_tokens=self.max_tokens,
            response_format=self.refine_response_schema,
        ).choices[0].message.parsed

        refine_sentences = self.process_refine_output([r.corrected_text.replace("\n", " ") for r in response.bboxes])
        translate_prompt = self.get_prompt(query_regions, w, h, nw, nh, only_text=True, texts=refine_sentences)

        response = self.client2.beta.chat.completions.parse(
            model=self.translate_model,
            messages=[
                {"role": "system", "content": self.get_translate_system_instruction(from_lang, to_lang)},
                {"role": "user", "content": [{"type": "text", "text": translate_prompt}]}
            ],
            temperature=self.translate_temperature,
            max_completion_tokens=self.max_tokens,
            response_format=self.translate_response_schema,
            reasoning_effort='none',
        ).choices[0].message.parsed
        return [r.translated_text.replace("\n", " ") for r in response.translated_texts]

    def process_refine_output(self, refine_output: List[str]) -> List[str]:
        all_symbols = self._LEFT_SYMBOLS + self._RIGHT_SYMBOLS
        processed = []

        for text in refine_output:
            stripped = text.strip()
            if removed := text[:len(text) - len(stripped)]:
                logger.info(f'Removed leading characters: "{removed}" from "{text}"')

            left_count = sum(stripped.count(s) for s in self._LEFT_SYMBOLS)
            right_count = sum(stripped.count(s) for s in self._RIGHT_SYMBOLS)

            if left_count != right_count:
                for s in all_symbols:
                    stripped = stripped.replace(s, '')
                logger.info(f'Removed unpaired symbols from "{stripped}"')

            processed.append(stripped.strip())
        return processed

    def get_prompt(self, text_regions, width: int, height: int, new_width: int, new_height: int, only_text=False,
                   texts=None):
        lines = ["```json", "["]
        for i, region in enumerate(text_regions):
            x1, y1, x2, y2 = region.xyxy
            x1, y1 = int((x1 / width) * new_width), int((y1 / height) * new_height)
            x2, y2 = int((x2 / width) * new_width), int((y2 / height) * new_height)
            text = texts[i] if texts else region.text
            if only_text:
                lines.append(f'\t{{"text_id": {i}, "text": "{text}"}},')
            else:
                lines.append(f'\t{{"bbox_id": {i}, "bbox_2d": [{x1}, {y1}, {x2}, {y2}], "text": "{text}"}},')
        lines.append("]")
        lines.append("```")
        return "\n".join(lines)

    def _get_refine_system_instruction(self, from_lang: str):
        return f"""You are an advanced OCR text correction engine specialized in processing visual media content like {from_lang} manga, illustrations, anime, novels, and scripts.
Your task is to analyze the input JSON, which contains raw OCR `"text"` data and corresponding `"bbox_2d"` coordinates ([x1, y1, x2, y2]). You must enhance the `"text"` by correcting inaccuracies, guided by the visual context within the bounding box and the overall image. Adhere strictly to the following instructions:

1.  **Primary Correction Objective:** Scrutinize the `"text"` field for each entry. Correct any typos, misrecognized characters (paying close attention to visually similar characters across different scripts), segmentation errors (split/merged words), and omissions. The corrected text *must* accurately represent the characters visually present within the specified `"box_2d"`.
2.  **Context is Paramount:** Utilize all available visual context from the image snippet corresponding to the `"bbox_2d"`. This includes character expressions, actions, background details, art style, surrounding text (if visible), and genre conventions. Ensure the corrected text aligns logically and tonally with this context.
3.  **Handwritten & Stylized Text:** Interpret and transcribe handwritten text, stylized fonts, and sound effects (SFX) as accurately as possible based on their appearance within the box. Prioritize legibility and faithfulness to the original artistic intent.
4.  **Faithfulness & Nuance:** The correction must remain strictly faithful to the source material's meaning and nuance in {from_lang}. Do *not* translate, paraphrase, or add information not visually present. Preserve the original intent, even if the grammar or phrasing seems unconventional in {from_lang}.
5.  **Punctuation & Formatting Integrity:** Retain all original punctuation marks exactly as they appear. Do not add extraneous symbols or remove necessary punctuation. Respect subtle visual formatting cues (like bolding or emphasis implied by thicker lines) if they are clearly part of the text within the box, representing them plainly in the text string without special tags.
6.  **Structural Consistency:** Maintain the original JSON array structure. The number of elements and their order in the output array must exactly match the input array.
7.  **Plausibility for Ambiguity:** If text is partially obscured, distorted, or highly ambiguous, provide the *most plausible* correction based on the visible fragments and the surrounding context. Avoid speculation beyond reasonable interpretation.
8.  **Output Format:** Return the final result as a single, continuous line of valid JSON text. The JSON should contain objects with the original `"bbox_id"` field, original `"bbox_2d"` field, original `"text"` field and the corrected text under a new field named `"corrected_text"`. Ensure absolutely no newline characters (\\n) are included in the output string."""

    def get_translate_system_instruction(self, from_lang: str, to_lang: str):
        return f"""You are an expert translator specializing in creative content such as manga, illustrations, anime, novels, and scripts, translating from {from_lang} to {to_lang}.
Please use the provided JSON input as references to translate only the text within each `"text`" field from {from_lang} to {to_lang}. Your primary goal is to produce a high-quality, natural-sounding, and **strictly consistent** translation in {to_lang} suitable for the target audience. This requires accurately reflecting the original intent while meticulously adhering to the conventions, natural flow, and consistent stylistic choices appropriate for {to_lang}.

Specifically, follow these rules:

**Core Translation Task & Fundamental Principles:**
1.  **Translate Content:** Translate only the text inside the `"text"` field for each JSON object.
2.  **Maintain Structure:** Ensure the order and structure of the objects in the JSON list remain exactly the same in the output.
3.  **Output Format:** Output valid JSON containing the translated text in a new field named `"translated_text"`, alongside the original fields for each object.
4.  **Strictly Prohibit Unnatural Language Mixing:** Under **absolutely no circumstances** should words, phrases, characters, grammatical structures, or typographical elements from {from_lang} (or any other language) be mixed into the {to_lang} translation, **unless** the specific mixed element is verifiably a naturalized loanword, a commonly used foreign expression already fully established and considered natural within **{to_lang}**, or if the original source text explicitly and intentionally depicts code-switching by a character for a specific narrative purpose (and even then, represent it naturally within the norms of **{to_lang}**). The default must **always** be to use pure, idiomatic **{to_lang}**. Any doubt should result in using the pure **{to_lang}** equivalent.

**Quality Enhancement Rules for Translation into {to_lang}:**
***Apply these rules consistently throughout the translation to maintain stylistic integrity, consistent tone, and naturalness in {to_lang}.***
5.  **Natural Flow and Sentence Structure:** Adapt the original sentence structure to flow naturally according to the grammatical norms and stylistic conventions of **{to_lang}**. Maintain consistency in sentence complexity and rhythm where appropriate for the text type and characters.
6.  **Tone and Register:** Accurately capture and **consistently maintain** the original tone (e.g., humorous, serious, sarcastic, formal, informal, flustered, excited) and register. Convey politeness levels or familiarity using appropriate and **consistent** **{to_lang}** word choice, phrasing, grammatical structures, and honorific systems native to **{to_lang}**.
7.  **Character Voice:** Maintain distinct and **internally consistent** voices for each character throughout the entire text. Reflect their personality, age, social status, dialect (if any), and emotional state through their dialogue rendered idiomatically and **consistently** in **{to_lang}**.
8.  **Idioms and Figurative Language:** Translate idioms, metaphors, proverbs, etc., using equivalent expressions in **{to_lang}** that capture the intended meaning and cultural resonance. Apply chosen equivalents **consistently** for recurring idioms or expressions.
9.  **Cultural Nuances:** Handle cultural references, customs, or terms smoothly and **consistently** for the **{to_lang}** audience using natural **{to_lang}** equivalents or subtle adaptations for clarity within the target culture.
10. **Onomatopoeia and Sound Effects (SFX):** Adapt sound effects effectively and **consistently** into **{to_lang}** using standard or descriptive equivalents conventional in **{to_lang}**.
11. **Emphasis and Interjections:** Convey the nuances of emphasis, questions, hesitations, interruptions, and interjections from {from_lang} using appropriate **{to_lang}** mechanisms, such as punctuation conventions, typography (e.g., italics, bolding, used sparingly according to **{to_lang}** norms), word choice, or specific grammatical constructions available in **{to_lang}**. **Crucially, do not simply replicate source-language-specific typographical markers of emphasis, interruption, or stammering (like trailing small characters (e.g., っ, ッ), excessive punctuation without equivalent function, or phonetic symbols) unless they have a direct, natural, and conventional equivalent function in {to_lang}.** For example, represent a stammer or pause using **{to_lang}**'s standard conventions (e.g., ellipses "...", doubled letters/words if conventional, appropriate phrasing).
12. **Honorifics/Titles:** Adapt {from_lang} honorifics or titles naturally and **consistently** into **{to_lang}** using functional equivalents within **{to_lang}**'s social/linguistic system or omitting/adjusting based on context and target language norms.
13. **Avoid Other Unnatural Carryover:** Beyond language mixing (covered in rule 4), also avoid awkward literal translations (calques) and foreign-sounding sentence structures unless intentionally present in the source. Ensure **consistency** in handling any such cases.

**Content Integrity:**
14. **No Censorship:** Translate the content faithfully.

Ensure the final output is a clean JSON structure containing the original `"text_id"` field, original `"text"` field, and `"translated_text"` field for each entry, reflecting **strict consistency** in the application of these rules and stylistic choices, especially regarding the prohibition of unnatural language mixing."""