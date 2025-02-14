import onnxruntime as ort
import torch
import numpy as np
import librosa
import time
from faster_whisper import WhisperModel
from translation_routing import get_translation_path, run_translation_pipeline
from cultural_tone_adjustment import adjust_tone

# ================================
# 1. Load the Faster Whisper Model for Speech-to-Text
# ================================
MODEL_SIZE = "small"  # "small" for speed; "medium" for increased accuracy

try:
    model = WhisperModel(MODEL_SIZE, compute_type="int8")  # Use int8 for efficiency
    print(f"✅ Faster-Whisper model loaded ({MODEL_SIZE}).")
except Exception as e:
    print("❌ ERROR: Failed to load Faster-Whisper model.", e)
    exit()

# ================================
# 2. Audio Preprocessing Function
# ================================
def load_audio(audio_path):
    """Loads an audio file and converts it to 16kHz mono format."""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"🔊 Audio Loaded: Shape {audio.shape}, Sample Rate {sr}")
        return audio
    except Exception as e:
        raise Exception("Audio loading failed: " + str(e))

# ================================
# 3. Load and Process the Audio File
# ================================
AUDIO_FILE = "output.mp3"
try:
    audio = load_audio(AUDIO_FILE)
except Exception as e:
    print("❌ ERROR: Failed to load or preprocess audio file.\n", e)
    exit()

# ================================
# 4. Run Whisper Speech-to-Text
# ================================
try:
    print("🛠 Running Faster-Whisper Speech-to-Text...")
    start_time = time.time()
    
    segments, info = model.transcribe(audio, beam_size=3)  # Reduced beam size for speed
    transcription_text = " ".join([segment.text for segment in segments])
    
    end_time = time.time()
    print(f"✅ Whisper Transcription Output: {transcription_text}")
except Exception as e:
    print("❌ ERROR: Speech-to-Text Failed.\n", e)
    exit()

# ================================
# 5. Select Source & Target Languages
# ================================
lang_map = {
    "ja": "Japanese (日本語)", 
    "ko": "Korean (한국어)", 
    "zh": "Chinese (中文)", 
    "en": "English"
}

# Reverse mapping to allow native script & full language name input
native_lang_map = {
    "日本語": "ja", "にほんご": "ja", "일본어": "ja", "日本语": "ja", "japanese": "ja",
    "한국어": "ko", "조선말": "ko", "韓国語": "ko", "韩国语": "ko", "korean": "ko",
    "中文": "zh", "汉语": "zh", "漢語": "zh", "chinese": "zh",
    "English": "en", "영어": "en", "英語": "en", "英语": "en", "english": "en"
}

def get_valid_language_input(prompt_text):
    """Loop until a valid language input is received."""
    while True:
        lang_input = input(prompt_text).strip().lower()
        # Convert full name or native script input to language code
        if lang_input in native_lang_map:
            return native_lang_map[lang_input]
        elif lang_input in lang_map:
            return lang_input
        else:
            print("❌ Invalid selection. Please enter a valid language.")
            print("📌 言語をお選びください (日本語, 한국어, 中文, English)")
            print("📌 올바른 언어를 입력하세요 (日本語, 한국어, 中文, English)")
            print("📌 请选择正确的语言 (日本語, 한국어, 中文, English)")

# 🌍 Get Source Language
source_lang = get_valid_language_input(
    "🌍 Enter source language (ja, ko, zh, en) \n"
    "📌 原言語を入力してください (日本語, 한국어, 中文, English) \n"
    "📌 원본 언어를 입력하세요 (日本語, 한국어, 中文, English) \n"
    "📌 请输入源语言 (日本語, 한국어, 中文, English): "
)
print(f"🗣️ Selected Source Language: {lang_map[source_lang]}")

# 🌍 Get Target Language
target_lang = get_valid_language_input(
    "🌍 Enter target language (ja, ko, zh, en) \n"
    "📌 対象言語を入力してください (日本語, 한국어, 中文, English) \n"
    "📌 목표 언어를 입력하세요 (日本語, 한국어, 中文, English) \n"
    "📌 请输入目标语言 (日本語, 한국어, 中文, English): "
)

print(f"🗣️ Translating to: {lang_map[target_lang]}")
# ================================
# 6. Tone Mode Selection
# ================================
tone_mode = input("🎤 Choose tone mode (business/formal): ").strip().lower()
if tone_mode not in ["business", "formal"]:
    print("❌ Invalid tone mode selection. Defaulting to 'business'.")
    tone_mode = "business"
print("🗣️ Tone mode selected:", tone_mode)

# ================================
# 7. Translation Pipeline
# ================================
try:
    translation_path = get_translation_path(source_lang, target_lang)
    if translation_path is None:
        print(f"❌ ERROR: No translation path available from {source_lang} to {target_lang}")
        exit()
    
    print(f"🔄 Translation path: {translation_path}")
    
    # Handling Korean translation with fallback
    if target_lang == "ko" and "en-ko" in translation_path:
        translation_path = [translation_path[0], "ko-en-reverse"]
        print("🔄 Modified translation path for Korean:", translation_path)
    
    translated_text = run_translation_pipeline(transcription_text, translation_path)
    print("✅ Translated Text (Before Tone Adjustment):", translated_text)
except Exception as e:
    print("❌ ERROR: Translation Failed.\n", e)
    exit()

# ================================
# 8. Tone Adjustment & Final Output
# ================================
final_output_messages = {
    "ja": ("✅ 調整後の翻訳:", "Final Translation Output:"),
    "ko": ("✅ 조정된 번역:", "Final Translation Output:"),
    "zh": ("✅ 调整后的翻译:", "Final Translation Output:"),
    "en": ("✅ Final Translation Output:", "Final Translation Output:")
}

try:
    final_output = adjust_tone(translated_text, target_lang, mode=tone_mode)
    
    # Dynamically print based on target language
    print(final_output_messages[target_lang][0], final_output)  # Target language message
    print(final_output_messages["en"][1], final_output)  # Always print in English too
    
    total_time = end_time - start_time
    processing_messages = {
        "ja": "⏳ 処理時間: {:.2f} 秒",
        "ko": "⏳ 처리 시간: {:.2f} 초",
        "zh": "⏳ 处理时间: {:.2f} 秒",
        "en": "⏳ Processing Time: {:.2f} seconds"
    }

    print(processing_messages[target_lang].format(total_time))  # Target language
    print(processing_messages["en"].format(total_time))  # Always print in English too

except Exception as e:
    error_messages = {
        "ja": "❌ 調整に失敗しました。",
        "ko": "❌ 조정 실패.",
        "zh": "❌ 调整失败.",
        "en": "❌ Tone Adjustment Failed."
    }

    print(error_messages[target_lang], e)
    print(error_messages["en"], e)  # Always print in English too