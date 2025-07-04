# ===============================================================
# Application Constants
#
# Description: This file holds all semi-static data for the
#              application, such as language lists, model groups,
#              and other constants. This makes it easy to update
#              this data without searching through all the code.
# ===============================================================

# A dictionary mapping user-friendly language names to their API codes.
LANGUAGES = {
    "Auto-Detect": "auto",
    "English": "ENG",
    "Turkish": "TRK",
    "Japanese": "JPN",
    "Korean": "KOR",
    "Simplified Chinese": "CHS",
    "Traditional Chinese": "CHT",
    "Spanish": "ESP",
    "French": "FRA",
    "German": "DEU",
    "Russian": "RUS",
    "Portuguese (Brazilian)": "PTB",
    "Italian": "ITA",
    "Polish": "PLK",
    "Dutch": "NLD",
    "Czech": "CSY",
    "Hungarian": "HUN",
    "Romanian": "ROM",
    "Ukrainian": "UKR",
    "Vietnamese": "VIN",
    "Arabic": "ARA",
    "Serbian": "SRP",
    "Croatian": "HRV",
    "Thai": "THA",
    "Indonesian": "IND",
    "Filipino (Tagalog)": "FIL"
}

# A dictionary to group translators in the dropdown menu for better readability.
TRANSLATOR_GROUPS = {
    "--- OFFLINE MODELS (No API Key) ---": [
        "sugoi", "m2m100", "m2m100_big", "nllb", "nllb_big", "mbart50",
        "jparacrawl", "jparacrawl_big", "qwen2", "qwen2_big", "offline"
    ],
    "--- API-BASED (Requires Setup) ---": [
        "deepl", "gemini", "deepseek", "groq", "youdao", "baidu",
        "caiyun", "sakura", "papago", "openai", "custom_openai"
    ],
    "--- OTHER ACTIONS ---": [
        "original",
        "none"
    ]
}

# A dictionary that maps translators to their supported language pairs (source, target).
# This provides the data for both filtering the target language dropdown and
# displaying informative tooltips to the user.
#
# Format:
# 'translator_name': {
#     'source_language_code': ['list_of', 'supported', 'target_codes'],
#     '__any__': '__all__' // A special key indicating that this model can translate
#                          // from any supported language to any other.
# }
TRANSLATOR_CAPABILITIES = {
    # --- API-BASED (Generally versatile) ---
    # For major APIs, assuming they can handle any pair we have in our language list.
    "deepl": {'__any__': '__all__'},
    "gemini": {'__any__': '__all__'},
    "deepseek": {'__any__': '__all__'},
    "groq": {'__any__': '__all__'},
    "youdao": {'__any__': '__all__'},
    "baidu": {'__any__': '__all__'},
    "caiyun": {'__any__': '__all__'},
    "openai": {'__any__': '__all__'},
    "custom_openai": {'__any__': '__all__'},

    # --- SPECIALIZED APIs (Limited pairs) ---
    "papago": {  # Based on Naver Papago's known strengths
        'KOR': ['ENG', 'JPN', 'CHS', 'CHT', 'FRA', 'DEU', 'RUS', 'ESP', 'ITA', 'VIE', 'THA', 'IND'],
        'JPN': ['ENG', 'KOR', 'CHS', 'CHT'],
        'CHS': ['ENG', 'KOR', 'JPN'],
        'CHT': ['ENG', 'KOR', 'JPN'],
        'ENG': ['KOR', 'JPN', 'CHS', 'CHT', 'FRA', 'DEU', 'ESP', 'ITA'],
        'FRA': ['ENG', 'KOR'],
        'ESP': ['ENG', 'KOR'],
        'ITA': ['ENG', 'KOR'],
        'DEU': ['ENG', 'KOR']
    },
    "sakura": {  # Specialized for CJK languages as per docs
        'JPN': ['CHS', 'CHT'],
        'CHS': ['JPN'],
        'CHT': ['JPN']
    },

    # --- SPECIALIZED OFFLINE MODELS (Often one-way) ---
    "sugoi": {  # Primarily Japanese to English
        'JPN': ['ENG']
    },
    "jparacrawl": {  # Japanese to English
        'JPN': ['ENG']
    },
    "jparacrawl_big": {  # Japanese to English
        'JPN': ['ENG']
    },

    # --- MULTILINGUAL OFFLINE MODELS (Assumed to be versatile) ---
    "nllb": {'__any__': '__all__'},
    "nllb_big": {'__any__': '__all__'},
    "m2m100": {'__any__': '__all__'},
    "m2m100_big": {'__any__': '__all__'},
    "mbart50": {'__any__': '__all__'},
    "qwen2": {'__any__': '__all__'},
    "qwen2_big": {'__any__': '__all__'},
    "offline": {'__any__': '__all__'},

    # --- OTHER ACTIONS (No translation capabilities) ---
    "original": {},
    "none": {}
}

LOG_COLORS = {
    "ERROR": "#E74C3C",
    "SUCCESS": "#2ECC71",
    "PIPELINE": "#5DADE2",
    "WARNING": "#F39C12",
    "INFO": "white",
    "DEBUG": "gray",
    "RAW": "gray"
}
