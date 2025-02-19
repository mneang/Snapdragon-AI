# fixed_phrases.py

import re

FIXED_PHRASES = {
    # Japanese
    "よろしくお願いいたします": "Looking forward to working with you.",
    "どうぞよろしく": "Nice to meet you.",
    "お世話になっております": "Thank you for your continued support.",
    "こちらこそ": "Likewise.",
    "お疲れ様です": "Great work today.",
    "ご苦労様です": "Thank you for your hard work.",
    "申し訳ありません": "I sincerely apologize.",
    # Korean
    "잘 부탁드립니다": "Looking forward to working with you.",
    "수고하셨습니다": "Great work today.",
    "고생하셨습니다": "Thank you for your hard work.",
    "감사합니다": "Thank you.",
    "죄송합니다": "I'm sorry.",
    "천천히 하세요": "Take your time.",
    "만나서 반갑습니다": "Nice to meet you."
}

def protect_fixed_phrases(text):
    """
    Replace occurrences of fixed cultural phrases with unique markers.
    Returns the protected text and a mapping of markers to fixed phrases.
    """
    marker_dict = {}
    protected_text = text
    marker_id = 1
    for key, value in FIXED_PHRASES.items():
        if key in protected_text:
            marker = f"__FIXED_{marker_id}__"
            marker_dict[marker] = value
            protected_text = protected_text.replace(key, marker)
            marker_id += 1
    return protected_text, marker_dict

def restore_fixed_phrases(text, marker_dict):
    """
    Replace markers in the text with the corresponding fixed phrase.
    Any punctuation immediately following a marker is removed.
    """
    restored_text = text
    for marker, fixed in marker_dict.items():
        # Pattern: marker followed by any number of punctuation (。 , .)
        pattern = re.escape(marker) + r"[。,.]*"
        restored_text = re.sub(pattern, fixed, restored_text)
    return restored_text

def check_fixed_phrases(text):
    """
    Directly replace fixed phrases in text.
    """
    for key, value in FIXED_PHRASES.items():
        if key in text:
            text = text.replace(key, value)
    return text

# Standalone Test
if __name__ == "__main__":
    test_text = "こんにちは、これはテストです。本日もよろしくお願いいたします。 皆様と素晴らしい一日を過ごせることを楽しみにしております。 どうぞよろしくお願いいたします。"
    protected, markers = protect_fixed_phrases(test_text)
    print("Protected:", protected)
    restored = restore_fixed_phrases(protected, markers)
    print("Restored:", restored)
