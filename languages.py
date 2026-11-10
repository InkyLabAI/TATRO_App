"""Supported transcription and display languages."""
LANGUAGES = {
    "1":  {"name": "Hebrew",              "code": "he", "whisper_code": "he", "is_rtl": True},
    "2":  {"name": "Arabic",              "code": "ar", "whisper_code": "ar", "is_rtl": True},
    "3":  {"name": "English",             "code": "en", "whisper_code": "en", "is_rtl": False},
    "4":  {"name": "French",              "code": "fr", "whisper_code": "fr", "is_rtl": False},
    "5":  {"name": "Spanish",             "code": "es", "whisper_code": "es", "is_rtl": False},
    "6":  {"name": "Chinese",             "code": "zh", "whisper_code": "zh", "is_rtl": False},
    "7":  {"name": "Turkish",             "code": "tr", "whisper_code": "tr", "is_rtl": False},
    "8":  {"name": "German",              "code": "de", "whisper_code": "de", "is_rtl": False},
    "9":  {"name": "Italian",             "code": "it", "whisper_code": "it", "is_rtl": False},
    "10": {"name": "Russian",             "code": "ru", "whisper_code": "ru", "is_rtl": False},
    "11": {"name": "Persian (Farsi)",     "code": "fa", "whisper_code": "fa", "is_rtl": True},
}


def by_name(name):
    return next(item for item in LANGUAGES.values() if item["name"] == name)
