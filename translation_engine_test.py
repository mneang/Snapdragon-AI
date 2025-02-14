from translation_engine import translate_text

test_cases = [
    ("こんにちは、これはテストです。", "ja-en"),            # Japanese -> English
    ("Hello, this is a test of our complete translation system designed for multinational business meetings.", "en-zh"),                     # English -> Chinese
    ("你好，这是一个测试。", "zh-en"),                         # Chinese -> English
    ("Hello, this is a test.", "ko-en-reverse"),             # Simulated English -> Korean using reverse mode
]

for text, model in test_cases:
    print("\n🔍 INPUT TEXT:", text)
    output = translate_text(text, model)
    print("✅ TRANSLATED OUTPUT:", output)
