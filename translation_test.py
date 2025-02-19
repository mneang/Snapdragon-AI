from translation_engine import translate_text

# Define test cases as tuples of (input_text, model_identifier)
# The model_identifier here is our shorthand for the translation direction.
# Note: For pairs not directly supported (like en-ja, ko-ja, zh-ja), our main.py fallback logic kicks in.
# For our testing here, we simulate that by passing the appropriate model identifier.
# Our current system supports these paths (with fallbacks where needed):
# en->ja, en->zh, en->ko, ja->en, ja->zh, ja->ko, zh->en, zh->ja, zh->ko, ko->en, ko->ja, ko->zh.

test_cases = [
    # English → Japanese (fallback used)
    ("hello, i like you baby", "en-ja"),
    # English → Chinese
    ("hello, i like you baby", "en-zh"),
    # English → Korean
    ("hello, i like you baby", "en-ko"),
    
    # Japanese → English
    ("こんにちは、お元気ですか？", "ja-en"),
    # Japanese → Chinese
    ("こんにちは、お元気ですか？", "ja-zh"),
    # Japanese → Korean
    ("こんにちは、お元気ですか？", "ja-ko"),
    
    # Chinese → English
    ("你好，这是一个测试。", "zh-en"),
    # Chinese → Japanese (fallback: pivot zh→en then fallback en→ja)
    ("你好，这是一个测试。", "zh-ja"),
    # Chinese → Korean
    ("你好，这是一个测试。", "zh-ko"),
    
    # Korean → English
    ("안녕하세요, 잘 지내세요?", "ko-en"),
    # Korean → Japanese (fallback: pivot ko→en then fallback en→ja)
    ("안녕하세요, 잘 지내세요?", "ko-ja"),
    # Korean → Chinese
    ("안녕하세요, 잘 지내세요?", "ko-zh"),
]

for text, model in test_cases:
    print(f"\n🔍 Input Text: {text}\nModel: {model}")
    try:
        output = translate_text(text, model)
        print(f"✅ Output: {output}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
