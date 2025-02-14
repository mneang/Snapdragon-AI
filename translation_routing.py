
from translation_engine import translate_text

# Define the list of available translation model keys.
# (Ensure these keys match those used in your translation_engine.py.)
AVAILABLE_MODEL_KEYS = ["ja-en", "en-zh", "zh-en", "ko-en", "en-ko"]

def get_translation_path(source_lang, target_lang):
    """
    Determines the translation path from source_lang to target_lang.
    If a direct model exists (e.g. "ja-ko"), returns [ "ja-ko" ].
    Otherwise, if models for source-en and en-target exist,
    returns [ source-en, en-target ].
    """
    direct_key = f"{source_lang}-{target_lang}"
    if direct_key in AVAILABLE_MODEL_KEYS:
        return [direct_key]
    else:
        pivot_key1 = f"{source_lang}-en"
        pivot_key2 = f"en-{target_lang}"
        if pivot_key1 in AVAILABLE_MODEL_KEYS and pivot_key2 in AVAILABLE_MODEL_KEYS:
            return [pivot_key1, pivot_key2]
    return None

def run_translation_pipeline(text, translation_path):
    """
    Runs the translation process by applying each translation step in order.
    """
    if translation_path is None:
        print("❌ ERROR: No valid translation path found.")
        return text

    for model_key in translation_path:
        print(f"🔄 Translating with model: {model_key} ...")
        text = translate_text(text, model_key)
    return text

# Export these functions for use in main.py or elsewhere.
__all__ = ["get_translation_path", "run_translation_pipeline"]

# Quick test (when running this file directly)
if __name__ == "__main__":
    path = get_translation_path("ja", "ko")
    print(f"Translation path for ja -> ko: {path}")
    sample_text = "こんにちは、これはテストです。"
    result = run_translation_pipeline(sample_text, path)
    print("Final translation:", result)
