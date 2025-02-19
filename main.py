import onnxruntime as ort
import torch
import numpy as np
import librosa
import time
import re
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from translation_routing import get_translation_path, run_translation_pipeline
from translation_engine import fallback_translation  # for fallback translations

# Global caches for ONNX sessions and tokenizers to avoid reloading
model_cache = {}
tokenizer_cache = {}

def regex_clean(text):
    """
    Remove immediate repeated sequences within sentences using regex.
    """
    pattern = re.compile(r'(\b(?:\S+\s+){2,})(\1)+')
    return pattern.sub(r'\1', text)

def remove_duplicate_phrases(text):
    """
    Remove adjacent duplicate phrases from the text.
    This function splits the text into words and, for windows of 3 words or more,
    checks if the same phrase is repeated immediately and removes the duplicate.
    """
    words = text.split()
    n = len(words)
    window = 3
    while window <= n // 2:
        i = 0
        while i + 2 * window <= n:
            phrase = " ".join(words[i:i+window])
            next_phrase = " ".join(words[i+window:i+2*window])
            if phrase.lower() == next_phrase.lower():
                del words[i+window:i+2*window]
                n = len(words)
            else:
                i += 1
        window += 1
    return " ".join(words)

def clean_pivot_text(text):
    """
    Clean the pivot English text by first applying regex cleaning and then 
    removing duplicate phrases.
    """
    cleaned = regex_clean(text)
    cleaned = remove_duplicate_phrases(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned

# ------------------------------
# 1. Load the Faster Whisper Model for Speech-to-Text (for voice input)
# ------------------------------
MODEL_SIZE = "small"  # Using int8 quantization for efficiency on Snapdragon

def load_whisper_model():
    try:
        model = WhisperModel(MODEL_SIZE, compute_type="int8")
        print(f"✅ Faster-Whisper model loaded ({MODEL_SIZE}). (Quantization Enabled)")
        return model
    except Exception as e:
        print("❌ ERROR: Failed to load Faster-Whisper model.", e)
        exit()

# ------------------------------
# 2. Audio Preprocessing Function
# ------------------------------
def load_audio(audio_path):
    """Loads an audio file and converts it to 16kHz mono format."""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"🔊 Audio Loaded: Shape {audio.shape}, Sample Rate {sr}")
        return audio
    except Exception as e:
        raise Exception("Audio loading failed: " + str(e))

# ------------------------------
# Input Mode Selection
# ------------------------------
input_mode = input("Choose input mode: type 'v' for voice input or 't' for text input: ").strip().lower()
if input_mode not in ['v', 't']:
    print("Invalid selection. Defaulting to voice input.")
    input_mode = 'v'

if input_mode == 'v':
    # Voice input: load from audio file
    AUDIO_FILE = "output.mp3"
    try:
        audio = load_audio(AUDIO_FILE)
    except Exception as e:
        print("❌ ERROR: Failed to load or preprocess audio file.\n", e)
        exit()
    # Load Whisper model and transcribe
    whisper_model = load_whisper_model()
    try:
        print("🛠 Running Faster-Whisper Speech-to-Text...")
        start_time = time.time()
        segments, info = whisper_model.transcribe(audio, beam_size=3)
        transcription_text = " ".join([segment.text for segment in segments])
        end_time = time.time()
        print(f"✅ Whisper Transcription Output: {transcription_text}")
    except Exception as e:
        print("❌ ERROR: Speech-to-Text Failed.\n", e)
        exit()
else:
    # Text input: prompt user for text
    transcription_text = input("Enter the text to translate: ").strip()
    start_time = time.time()
    end_time = time.time()
    print(f"✅ Received text input: {transcription_text}")

# ------------------------------
# Memory Profiling: Log initial memory usage
# ------------------------------
process = psutil.Process()
mem_before = process.memory_info().rss / (1024 * 1024)
print(f"📊 Memory Usage Before Translation: {mem_before:.2f} MB")

# ------------------------------
# 3. Select Source & Target Languages
# ------------------------------
lang_map = {
    "ja": "Japanese (日本語)", 
    "ko": "Korean (한국어)", 
    "zh": "Chinese (中文)", 
    "en": "English"
}

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
        if lang_input in native_lang_map:
            return native_lang_map[lang_input]
        elif lang_input in lang_map:
            return lang_input
        else:
            print("❌ Invalid selection. Please enter a valid language.")
            print("📌 言語をお選びください (日本語, 한국어, 中文, English)")
            print("📌 올바른 언어를 입력하세요 (日本語, 한국어, 中文, English)")
            print("📌 请选择正确的语言 (日本語, 한국어, 中文, English)")

source_lang = get_valid_language_input(
    "🌍 Enter source language (ja, ko, zh, en) \n"
    "📌 原言語を入力してください (日本語, 한국어, 中文, English) \n"
    "📌 원본 언어를 입력하세요 (日本語, 한국어, 中文, English) \n"
    "📌 请输入源语言 (日本語, 한국어, 中文, English): "
)
print(f"🗣️ Selected Source Language: {lang_map[source_lang]}")

target_lang = get_valid_language_input(
    "🌍 Enter target language (ja, ko, zh, en) \n"
    "📌 対象言語を入力してください (日本語, 한국어, 中文, English) \n"
    "📌 목표 언어를 입력하세요 (日本語, 한국어, 中文, English) \n"
    "📌 请输入目标语言 (日本語, 한국어, 中文, English): "
)
print(f"🗣️ Translating to: {lang_map[target_lang]}")

# ------------------------------
# 4. Tone Mode Selection (Cultural tone adjustments disabled)
# ------------------------------
print("Note: Cultural tone adjustments have been disabled.")
tone_mode = "none"

# ------------------------------
# 5. Fallback for Unsupported Language Pairs
# ------------------------------
# Fallback for English→Japanese:
if source_lang == "en" and target_lang == "ja":
    print("🔄 English→Japanese translation requested. Using fallback translator.")
    translated_text = fallback_translation(transcription_text, "en-ja")
    print("✅ Final Translation Output:", translated_text)
    exit()

# Fallback for Korean→Japanese:
if source_lang == "ko" and target_lang == "ja":
    print("🔄 Korean→Japanese translation requested. Using pivot translation (ko→en) then fallback for en→ja.")
    pivot_text = run_translation_pipeline(transcription_text, ["ko-en"])
    translated_text = fallback_translation(pivot_text, "en-ja")
    print("✅ Final Translation Output:", translated_text)
    exit()

# Fallback for Chinese→Japanese:
if source_lang == "zh" and target_lang == "ja":
    print("🔄 Chinese→Japanese translation requested. Using pivot translation (zh→en) then fallback for en→ja.")
    pivot_text = run_translation_pipeline(transcription_text, ["zh-en"])
    translated_text = fallback_translation(pivot_text, "en-ja")
    print("✅ Final Translation Output:", translated_text)
    exit()

# ------------------------------
# 6. Fixed Phrases Protection on Original Transcription (for Japanese/Korean)
# ------------------------------
if source_lang in ["ja", "ko"]:
    from fixed_phrases import protect_fixed_phrases
    protected_text, marker_dict = protect_fixed_phrases(transcription_text)
    if protected_text != transcription_text:
        print("🔄 Protected transcription text based on cultural phrases:")
        print("   Before: ", transcription_text)
        print("   After:  ", protected_text)
        transcription_text = protected_text

# ------------------------------
# 7. Translation Pipeline with Fixed Phrases Restoration on Pivot Translations
# ------------------------------
try:
    translation_path = get_translation_path(source_lang, target_lang)
    if translation_path is None:
        print(f"❌ ERROR: No translation path available from {source_lang} to {target_lang}")
        exit()
    
    print(f"🔄 Translation path: {translation_path}")
    
    # For Korean target with pivot, modify path accordingly.
    if target_lang == "ko" and "en-ko" in translation_path:
        translation_path = [translation_path[0], "ko-en-reverse"]
        print("🔄 Modified translation path for Korean:", translation_path)
    
    # If there's a single-step translation, enable batch processing for multi-sentence inputs.
    if len(translation_path) == 1:
        sentences = re.split(r'(?<=[\.\!\?])\s+', transcription_text)
        if len(sentences) > 1:
            print("📊 Batch processing enabled for multi-sentence input.")
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(run_translation_pipeline, sentence, translation_path) for sentence in sentences]
                translated_text = " ".join([f.result() for f in futures])
        else:
            translated_text = run_translation_pipeline(transcription_text, translation_path)
    # Otherwise, if pivot translation is required and source is Japanese/Korean, restore fixed phrases.
    elif len(translation_path) > 1 and translation_path[0].endswith("-en") and source_lang in ["ja", "ko"]:
        pivot_text = run_translation_pipeline(transcription_text, [translation_path[0]])
        from fixed_phrases import restore_fixed_phrases
        pivot_text = restore_fixed_phrases(pivot_text, marker_dict)
        pivot_text = clean_pivot_text(pivot_text)
        if not pivot_text.strip():
            print("⚠️ Pivot translation returned empty. Using fallback translator for ja-en.")
            pivot_text = fallback_translation(transcription_text, "ja-en")
        print("🔄 Pivot translation after restoring and cleaning fixed phrases:")
        print("   ", pivot_text)
        remaining_path = translation_path[1:]
        translated_text = run_translation_pipeline(pivot_text, remaining_path)
    else:
        translated_text = run_translation_pipeline(transcription_text, translation_path)
    
    # Additional check: if direct ja-en output is empty, trigger fallback.
    if source_lang == "ja" and target_lang == "en" and not translated_text.strip():
        print("⚠️ Direct ja-en output empty. Using fallback translator.")
        translated_text = fallback_translation(transcription_text, "ja-en")
    
    print("✅ Translated Text (Before Final Output):", translated_text)
except Exception as e:
    print("❌ ERROR: Translation Failed.\n", e)
    exit()

# ------------------------------
# 8. Final Output
# ------------------------------
print("✅ Final Translation Output:", translated_text)

# ------------------------------
# Memory Profiling: Log final memory usage and cleanup
# ------------------------------
mem_after = process.memory_info().rss / (1024 * 1024)
print(f"📊 Memory Usage After Translation: {mem_after:.2f} MB")
print(f"📊 Memory Usage Change: {mem_after - mem_before:.2f} MB")
gc.collect()

total_time = end_time - start_time
print(f"⏳ Processing Time: {total_time:.2f} seconds")

# ------------------------------
# Batch Processing / Parallelization (Already applied if multiple sentences)
# ------------------------------
sentences = re.split(r'(?<=[\.\!\?])\s+', transcription_text)
if len(sentences) > 1:
    print("📊 Batch processing was enabled for multi-sentence input.")
else:
    print("📊 Single-sentence input; batch processing not required.")
