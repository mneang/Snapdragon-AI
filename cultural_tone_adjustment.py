import json
import os

def load_tone_rules(config_path="tone_rules.json"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def adjust_tone(text, target_lang, mode="formal"):
    """
    Adjusts the tone of the translated text based on language-specific formal/business rules.
    """
    rules = load_tone_rules()
    if target_lang not in rules or mode not in rules[target_lang]:
        return text  # No specific rules, return as is.

    adjustments = rules[target_lang][mode]
    
    # Handle verb conjugation dynamically
    if "verbs" in adjustments:
        text = apply_verb_conjugation(text, target_lang, adjustments["verbs"])

    words = text.split()
    last_word = words[-1] if words else ""

    # Check if last word is a verb (including honorific forms)
    is_last_word_verb = check_if_verb(last_word, target_lang)

    # Apply suffix ONLY if the last word is NOT a verb (fixes the issue)
    if "suffix" in adjustments and adjustments["suffix"]:
        if not is_last_word_verb:
            text = text.strip() + adjustments["suffix"]
        else:
            text = text.strip() + "。"  # End properly

    return text

def check_if_verb(word, lang):
    """
    Returns True if the word is a verb, including honorific forms like お話しになります.
    """
    if lang == "ja":
        keigo_verbs = ["おっしゃいます", "なさいます", "いらっしゃいます",
                       "召し上がります", "お話しになります", "ご覧になります", "おいでになります"]
        masu_form_verbs = ["行きます", "食べます", "話します", "します", "見ます", "書きます", "来ます"]

        # Check if word matches a known keigo verb or a -ます form verb
        return word in keigo_verbs or word in masu_form_verbs

    elif lang == "ko":
        habnida_verbs = ["갑니다", "먹습니다", "말합니다", "합니다", "봅니다", "씁니다", "옵니다"]
        seyo_verbs = ["가세요", "드세요", "말씀하세요", "하세요", "보세요", "쓰세요", "오세요"]

        return word in habnida_verbs or word in seyo_verbs

    return False  # If language not recognized, assume not a verb


def verb_list(lang):
    """ Returns a list of known verbs for a given language to check endings. """
    if lang == "ja":
        return ["行く", "食べる", "話す", "する", "見る", "書く", "来る", "なる",
                "行きます", "食べます", "話します", "します", "見ます", "書きます", "来ます"]
    elif lang == "ko":
        return ["가다", "먹다", "말하다", "하다", "보다", "쓰다", "오다",
                "갑니다", "먹습니다", "말합니다", "합니다", "봅니다", "씁니다", "옵니다"]
    return []

def apply_verb_conjugation(text, lang, conjugation_type):
    """
    Converts verbs into their proper conjugation forms based on language rules.
    """
    if lang == "ja":
        if conjugation_type == "masu-form":
            return convert_to_masu_form(text)
        elif conjugation_type == "keigo":
            return convert_to_keigo(text)
    
    if lang == "ko":
        if conjugation_type == "habnida":
            return convert_to_habnida_form(text)
        elif conjugation_type == "seyo":
            return convert_to_seyo_form(text)

    return text  # No modification needed for this language

def convert_to_masu_form(text):
    masu_dict = {
        "する": "します",
        "行く": "行きます",
        "食べる": "食べます",
        "話す": "話します",
        "見る": "見ます",
        "書く": "書きます",
        "来る": "来ます"
    }
    words = text.split()
    return " ".join([masu_dict.get(word, word) for word in words])

def convert_to_keigo(text):
    keigo_dict = {
        "する": "なさいます",
        "行く": "いらっしゃいます",
        "食べる": "召し上がります",
        "話す": "お話しになります",
        "見る": "ご覧になります",
        "書く": "お書きになります",
        "来る": "おいでになります"
    }
    words = text.split()
    return " ".join([keigo_dict.get(word, word) for word in words])

def convert_to_habnida_form(text):
    habnida_dict = {
        "하다": "합니다",
        "가다": "갑니다",
        "먹다": "먹습니다",
        "말하다": "말합니다",
        "보다": "봅니다",
        "쓰다": "씁니다",
        "오다": "옵니다"
    }
    words = text.split()
    return " ".join([habnida_dict.get(word, word) for word in words])

def convert_to_seyo_form(text):
    seyo_dict = {
        "하다": "하세요",
        "가다": "가세요",
        "먹다": "드세요",
        "말하다": "말씀하세요",
        "보다": "보세요",
        "쓰다": "쓰세요",
        "오다": "오세요"
    }
    words = text.split()
    return " ".join([seyo_dict.get(word, word) for word in words])

# ✅ **TEST**
if __name__ == "__main__":
    test_text = "行く 食べる 話す"
    print("Formal (Japanese):", adjust_tone(test_text, "ja", "formal"))
    print("Business (Japanese):", adjust_tone(test_text, "ja", "business"))
