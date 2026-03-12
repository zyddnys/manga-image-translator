# Guide to Improve Mongolian Translation Accuracy with GPT

This guide provides comprehensive strategies to improve the accuracy of Japanese-to-Mongolian translations when using GPT-based translators.

## Table of Contents
1. [System Prompt Optimization](#system-prompt-optimization)
2. [Temperature and Parameter Settings](#temperature-and-parameter-settings)
3. [Using Glossaries and Dictionaries](#using-glossaries-and-dictionaries)
4. [Pre-Dict and Post-Dict Usage](#pre-dict-and-post-dict-usage)
5. [Chat Samples and Examples](#chat-samples-and-examples)
6. [Language-Specific Considerations](#language-specific-considerations)
7. [Testing and Iteration](#testing-and-iteration)

---

## System Prompt Optimization

### Recommended System Prompt for Mongolian

```yaml
chat_system_template: >
  You are an expert translator specializing in Japanese-to-Mongolian manga translation.
  
  ## Critical Accuracy Requirements:
  - Use proper Cyrillic Mongolian script (Монгол кирилл)
  - Follow Mongolian vowel harmony rules strictly
  - Maintain natural Mongolian word order (SOV structure)
  - Use correct case endings based on grammatical context
  - Preserve Japanese honorifics in romanized form: "-san", "-chan", "-kun"
  
  ## Translation Process:
  1. Understand the Japanese meaning accurately
  2. Convert to natural Mongolian while preserving:
     - Emotional tone and intensity
     - Character voice and personality
     - Cultural context (adapt appropriately)
  3. Verify: Does it sound natural to a native Mongolian speaker?
  
  ## Common Accuracy Issues to Avoid:
  - Incorrect vowel harmony (e.g., mixing front/back vowels)
  - Wrong case endings (nominative, accusative, dative, etc.)
  - Literal word-for-word translations (sounds unnatural)
  - Missing or incorrect particles
  - Over-formalizing casual speech
  
  Translate to {to_lang} with maximum accuracy:
```

### Key Points in System Prompt:
- **Explicitly mention vowel harmony** - This is critical for Mongolian
- **Emphasize naturalness** - Not literal translation
- **Specify script** - Cyrillic Mongolian
- **Mention common errors** - Helps GPT avoid them

---

## Temperature and Parameter Settings

### Optimal Settings for Accuracy

```yaml
# For maximum accuracy (recommended)
temperature: 0.2  # Lower = more consistent, less creative
top_p: 0.9        # Focus on most likely tokens

# For balanced accuracy and naturalness
temperature: 0.3
top_p: 0.95

# Model-specific recommendations:
chatgpt:
  gpt-4o:          # Best for Mongolian (better language understanding)
    temperature: 0.2
  gpt-4:
    temperature: 0.25
  gpt-4o-mini:
    temperature: 0.3
  gpt-3.5-turbo:
    temperature: 0.2  # Lower for less capable models
```

### Why Lower Temperature?
- **Consistency**: Same input produces similar output
- **Accuracy**: Less creative "hallucinations"
- **Reliability**: Better adherence to grammar rules
- **Vowel Harmony**: More likely to follow linguistic rules correctly

---

## Using Glossaries and Dictionaries

### Creating a Mongolian Glossary

Create a glossary file (e.g., `mongolian_glossary.txt`) with format:

```
# Character names and proper nouns
田中	Танака
佐藤	Сато
東京	Токио

# Manga-specific terms
魔法	ид шид
剣士	байлдагч
冒険	аялал

# Consistent translations
大丈夫	зүгээр
ありがとう	баярлалаа
```

### Setting Up Glossary

1. **Create the glossary file** in your preferred format:
   - MIT format: `source\ttarget`
   - Sakura format: `source->target`
   - Galtransl format: `source    target` (4 spaces)

2. **Set environment variable**:
   ```bash
   export OPENAI_GLOSSARY_PATH=/path/to/mongolian_glossary.txt
   ```

3. **Benefits**:
   - Consistent character name translations
   - Proper noun accuracy
   - Manga-specific terminology
   - Cultural term handling

### Glossary Best Practices:
- **Character names**: Keep consistent across entire series
- **Proper nouns**: Cities, places, organizations
- **Technical terms**: Magic, technology, etc.
- **Cultural concepts**: Adapt or explain appropriately

---

## Pre-Dict and Post-Dict Usage

### Pre-Dict (Before Translation)

Use `--pre-dict` to fix OCR errors or prepare text:

**Example `pre_dict.txt`**:
```
# Fix common OCR errors
0	О
1	I
l	I

# Mark text to skip translation
(.*sound effect.*)	🔇$1🔇
```

### Post-Dict (After Translation)

Use `--post-dict` to fix common translation errors:

**Example `post_dict.txt`**:
```
# Fix common mistranslations
зүгээр байна	зүгээр
баярлалаа	баярлалаа
# Fix vowel harmony errors
хэрэгтэй байна	хэрэгтэй
# Natural phrasing improvements
юу хийж байна	юу хийж байна вэ
```

### Usage:
```bash
python -m manga_translator local -i input_folder -o output_folder \
  --translator chatgpt -l JPN:MON \
  --pre-dict pre_dict.txt \
  --post-dict post_dict.txt \
  --gpt-config gpt_config.yaml
```

---

## Chat Samples and Examples

### Importance of Diverse Examples

Provide multiple examples covering:
1. **Casual dialogue** - Informal speech patterns
2. **Formal dialogue** - Honorifics and polite speech
3. **Emotional expressions** - Exclamations, questions
4. **Action scenes** - Commands, urgency
5. **Thought bubbles** - Internal monologue

### Example Configuration:

```yaml
chat_sample:
  Mongolian:
    # Casual/emotional
    - <|1|>恥ずかしい… 目立ちたくない…
    - <|1|>Ичихээс... Би харагдахыг хүсэхгүй байна...
    
    # Formal with honorifics
    - <|1|>田中さん、おはようございます
    - <|1|>Танака-сан, өглөөний мэнд
    
    # Action/command
    - <|1|>危ない！早く逃げろ！
    - <|1|>Аюултай! Хурдан зугт!
```

---

## Language-Specific Considerations

### 1. Vowel Harmony

Mongolian has strict vowel harmony rules:
- **Front vowels**: э, ө, ү
- **Back vowels**: а, о, у
- **Neutral vowels**: и, е

**Example**:
- ✅ Correct: "хэрэгтэй" (all front vowels)
- ❌ Wrong: "хэрэгтой" (mixing front and back)

### 2. Case Endings

Mongolian uses case endings extensively:
- **Nominative**: (no ending) - subject
- **Accusative**: -г, -ийг - direct object
- **Dative**: -д, -т - indirect object
- **Genitive**: -ын, -ийн - possession

**Example**:
- "Би ном уншиж байна" (I am reading a book)
- "Би номд хандаж байна" (I am looking at the book)

### 3. Formality Levels

- **Formal "та"**: Use with strangers, superiors, formal situations
- **Informal "чи"**: Use with friends, family, casual situations

**Example**:
- Formal: "Та юу хийж байна вэ?" (What are you doing?)
- Informal: "Чи юу хийж байна?" (What are you doing?)

### 4. Word Order

Mongolian is SOV (Subject-Object-Verb), similar to Japanese:
- Japanese: "私は本を読む" (I book read)
- Mongolian: "Би ном уншдаг" (I book read)

### 5. Honorifics

**Always preserve Japanese honorifics in romanized form**:
- ✅ "Танака-сан" (correct)
- ❌ "Ноён Танака" (wrong - don't translate honorifics)

---

## Testing and Iteration

### Step 1: Baseline Test
1. Translate a sample page with default settings
2. Identify common error patterns:
   - Vowel harmony mistakes
   - Wrong case endings
   - Unnatural phrasing
   - Honorific handling

### Step 2: Adjust Parameters
1. Lower temperature if too creative
2. Add glossary entries for consistent errors
3. Update chat samples with better examples

### Step 3: Create Dictionaries
1. Build pre-dict for OCR fixes
2. Build post-dict for common mistranslations
3. Test and refine

### Step 4: Iterate
1. Review translations with native speakers
2. Collect feedback on accuracy issues
3. Update prompts, samples, and dictionaries
4. Repeat

### Testing Checklist:
- [ ] Vowel harmony is correct
- [ ] Case endings are appropriate
- [ ] Formality level matches context
- [ ] Honorifics preserved correctly
- [ ] Natural Mongolian phrasing
- [ ] Character names consistent
- [ ] Cultural context appropriate

---

## Advanced Tips

### 1. Use GPT-4 or GPT-4o
- Better language understanding
- More accurate grammar
- Better cultural adaptation

### 2. Context Size
- Use `--context-size` to provide page context
- Helps with character name consistency
- Improves dialogue flow

### 3. Batch Processing
- Process related pages together
- Maintains consistency across pages
- Better character voice preservation

### 4. Manual Review
- Always review first few translations
- Identify patterns in errors
- Update configuration accordingly

### 5. Native Speaker Review
- Get feedback from Mongolian speakers
- Identify unnatural phrasing
- Refine based on feedback

---

## Example Complete Configuration

```yaml
# gpt_config_mongolian.yaml
temperature: 0.2
top_p: 0.9

chat_system_template: >
  You are an expert Japanese-to-Mongolian manga translator.
  Use proper Cyrillic Mongolian with correct vowel harmony.
  Preserve Japanese honorifics in romanized form.
  Translate naturally, not literally.
  Translate to {to_lang}:

chat_sample:
  Mongolian:
    - <|1|>おはよう
    - <|1|>Өглөөний мэнд
    # Add more diverse examples...

chatgpt:
  gpt-4o:
    temperature: 0.2
```

---

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Vowel harmony errors | Lower temperature, add explicit instructions |
| Wrong case endings | Add examples with different cases |
| Unnatural phrasing | Add more diverse chat samples |
| Inconsistent names | Use glossary for character names |
| Over-formal speech | Add casual dialogue examples |
| Missing honorifics | Explicitly mention in system prompt |

---

## Resources

- [Mongolian Grammar Guide](https://en.wikipedia.org/wiki/Mongolian_grammar)
- [Vowel Harmony Rules](https://en.wikipedia.org/wiki/Mongolian_language#Vowel_harmony)
- Project glossary files: `dict/` directory
- Example configs: `examples/gpt_config-example.yaml`

---

## Conclusion

Improving Mongolian translation accuracy requires:
1. **Optimized prompts** with language-specific instructions
2. **Lower temperature** for consistency
3. **Comprehensive glossaries** for terminology
4. **Diverse examples** covering different contexts
5. **Iterative refinement** based on feedback

Start with the recommended settings and adjust based on your specific needs and feedback from native speakers.




