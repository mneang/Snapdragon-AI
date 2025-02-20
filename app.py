import streamlit as st
import time
import psutil
import gc
import re
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from translation_routing import get_translation_path, run_translation_pipeline
from translation_engine import fallback_translation
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # for forced Korean→English

# ------------------------------
# Set Page Config First
# ------------------------------
st.set_page_config(
    page_title="Multilingual On-Device Translator",
    layout="wide"
)

# ------------------------------
# Custom CSS Styling (Qualcomm-inspired)
# ------------------------------
st.markdown(
    """
    <style>
    /* General Body Styling */
    body {
        background-color: #F4F7F9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Headings in Qualcomm Blue */
    h1, h2, h3, h4, h5, h6 {
        color: #0072C6;
    }
    /* Buttons Styling */
    .stButton>button {
        background-color: #0072C6;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #005A9E;
    }
    /* Sidebar Styling: Qualcomm Blue background, text white by default */
    .stSidebar {
        background-color: #0072C6;
        padding: 1em;
        color: white !important;
    }
    .stSidebar, .stSidebar * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Global Caching of Heavy Resources
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model_cached():
    return WhisperModel("small", compute_type="int8")

@st.cache_resource(show_spinner=False)
def load_audio_cached(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        st.write(f"🔊 Audio Loaded: Shape {audio.shape}, Sample Rate {sr}")
        return audio
    except Exception as e:
        raise Exception("Audio loading failed: " + str(e))

# ------------------------------
# M2M100 for Korean→English
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_m2m100_ko_en():
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return model, tokenizer

def force_ko_en_m2m100(text):
    model, tokenizer = load_m2m100_ko_en()
    tokenizer.src_lang = "ko"
    inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("en"),
        max_length=300,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# ------------------------------
# Utility Functions
# ------------------------------
def regex_clean(text):
    pattern = re.compile(r'(\b(?:\S+\s+){2,})(\1)+')
    return pattern.sub(r'\1', text)

def remove_duplicate_phrases(text):
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
    cleaned = regex_clean(text)
    cleaned = remove_duplicate_phrases(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned

def remove_excessive_repetition(text):
    text = re.sub(r"(\'s)+", "'s", text)
    text = re.sub(r'([,.!?])\1+', r'\1', text)
    return text

def remove_excessive_repetition_tokens(text):
    prev_text = None
    pattern = re.compile(r"([A-Za-z]+)(?:'s){2,}", re.IGNORECASE)
    while text != prev_text:
        prev_text = text
        text = pattern.sub(r"\1's", text)
    return text

def additional_safety_cleanup(text):
    text = remove_excessive_repetition(text)
    text = remove_excessive_repetition_tokens(text)
    return text

def calculate_repetition_ratio(text):
    words = text.split()
    if not words:
        return 0.0
    return (len(words) - len(set(words))) / len(words)

# ------------------------------
# Language Mappings
# ------------------------------
label_to_code = {
    "English": "en",
    "Japanese (日本語)": "ja",
    "Korean (한국어)": "ko",
    "Chinese (中文)": "zh"
}
language_options = list(label_to_code.keys())

# ------------------------------
# Streamlit Layout & Instructions
# ------------------------------
st.title("Multilingual On-Device Translator")
st.markdown("""
**Welcome!**

This application translates between **English, Japanese, Korean, and Chinese**.  
Select your source and target languages from the options below, choose your input mode, and enter your text.
""")

# Sidebar Settings
input_mode = st.sidebar.radio("Select Input Mode:", ["Text Input", "Voice Input"])
st.sidebar.markdown("**Note:** Cultural tone adjustments have been disabled. / 文化的なトーン調整は無効です。 / 문화적 톤 조정은 비활성화되었습니다。 / 文化调节已禁用。")

# Main Area: Language Selection with Subtitles
col1, col2 = st.columns(2)
with col1:
    st.subheader("Source Language")
    st.markdown("<medium>**原言語 | 원본 언어 | 源语言**</medium>", unsafe_allow_html=True)
    source_lang_label = st.selectbox("", language_options, index=2)
with col2:
    st.subheader("Target Language")
    st.markdown("<medium>**対象言語 | 목표 언어 | 目标语言**</medium>", unsafe_allow_html=True)
    target_lang_label = st.selectbox("", language_options, index=0)

source_lang = label_to_code[source_lang_label]
target_lang = label_to_code[target_lang_label]

# Input Section with Subtitle
st.subheader("Input Text")
st.markdown("<medium>**入力 | 입력 | 输入**</medium>", unsafe_allow_html=True)
if input_mode == "Text Input":
    transcription_text = st.text_area("", "", height=150, placeholder="Type your text here")
else:
    st.audio("output.mp3", format="audio/mp3")
    st.info("Voice input detected. Transcribing audio...")
    try:
        audio = load_audio_cached("output.mp3")
        whisper_model = load_whisper_model_cached()
        result = whisper_model.transcribe(audio, beam_size=3)
        transcription_text = " ".join([segment.text for segment in result[0]])
        st.write("Transcribed text:", transcription_text)
    except Exception as e:
        st.error("Voice transcription failed. Please use text input.")
        transcription_text = st.text_area("", "", height=150, placeholder="Type your text here")

# ------------------------------
# Translation Trigger
# ------------------------------
if st.button("▶️ Translate"):
    if not transcription_text.strip():
        st.error("❌ Please provide text to translate")
    else:
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()
        
        # For Korean→English, force M2M100
        if source_lang == "ko" and target_lang == "en":
            st.info("Forcing M2M100 for Korean→English to avoid repetition.")
            translated_text = force_ko_en_m2m100(transcription_text)
        else:
            if source_lang == "en" and target_lang == "ja":
                st.info("🔄 English → Japanese translation requested. Using fallback translator.")
                translated_text = fallback_translation(transcription_text, "en-ja")
            elif source_lang == "ko" and target_lang == "ja":
                st.info("🔄 Korean → Japanese translation requested. Using pivot (ko→en) then fallback (en→ja).")
                pivot_text = run_translation_pipeline(transcription_text, ["ko-en"])
                translated_text = fallback_translation(pivot_text, "en-ja")
            elif source_lang == "zh" and target_lang == "ja":
                st.info("🔄 Chinese → Japanese translation requested. Using pivot (zh→en) then fallback (en→ja).")
                pivot_text = run_translation_pipeline(transcription_text, ["zh-en"])
                translated_text = fallback_translation(pivot_text, "en-ja")
            else:
                translation_path = get_translation_path(source_lang, target_lang)
                if translation_path is None:
                    st.error(f"No translation path available from {source_lang_label} to {target_lang_label}")
                    translated_text = ""
                else:
                    st.write("Translation path:", translation_path)
                    if len(translation_path) == 1:
                        sentences = re.split(r'(?<=[\.!?])\s+', transcription_text)
                        if len(sentences) > 1:
                            st.write("Multi-sentence input detected. Processing in batch...")
                            with ThreadPoolExecutor() as executor:
                                futures = [executor.submit(run_translation_pipeline, sentence, translation_path) for sentence in sentences]
                                translated_text = " ".join([f.result() for f in futures])
                        else:
                            translated_text = run_translation_pipeline(transcription_text, translation_path)
                    elif len(translation_path) > 1 and translation_path[0].endswith("-en") and source_lang in ["ja", "ko"]:
                        pivot_text = run_translation_pipeline(transcription_text, [translation_path[0]])
                        pivot_text = clean_pivot_text(pivot_text)
                        rep_ratio = calculate_repetition_ratio(pivot_text)
                        if rep_ratio > 0.20:
                            st.warning("High repetition detected in pivot translation. Using fallback translator.")
                            pivot_text = fallback_translation(transcription_text, "ja-en")
                        if not pivot_text.strip():
                            st.warning("Pivot translation returned empty. Using fallback translator.")
                            pivot_text = fallback_translation(transcription_text, "ja-en")
                        st.write("Pivot translation:", pivot_text)
                        remaining_path = translation_path[1:]
                        try:
                            translated_text = run_translation_pipeline(pivot_text, remaining_path)
                        except Exception as e:
                            st.error("Error in processing pivot translation. Using fallback translator for ko-en-reverse.")
                            translated_text = fallback_translation(pivot_text, "ko-en-reverse")
                    else:
                        translated_text = run_translation_pipeline(transcription_text, translation_path)
                    
                    if source_lang == "ja" and target_lang == "en" and not translated_text.strip():
                        st.warning("Direct ja→en output empty. Using fallback translator.")
                        translated_text = fallback_translation(transcription_text, "ja-en")
                    
                    if source_lang == "ko" and target_lang == "en":
                        rep_ratio = calculate_repetition_ratio(translated_text)
                        if rep_ratio > 0.20:
                            st.warning("High repetition detected in Korean→English output. Using fallback translator.")
                            translated_text = fallback_translation(transcription_text, "ko-en")
        
        # Final Cleanup
        translated_text = additional_safety_cleanup(translated_text)
        
        end_time = time.time()
        total_time = end_time - start_time
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        st.subheader("Translation Result")
        st.write(translated_text)
        st.markdown(f"**Processing Time:** {total_time:.2f} seconds")
        st.markdown(f"**Memory Usage Increase:** {mem_after - mem_before:.2f} MB")
        
        gc.collect()
        
        sentences = re.split(r'(?<=[\.!?])\s+', transcription_text)
        if len(sentences) > 1:
            st.write("📊 Batch processing was enabled for multi-sentence input.")
        else:
            st.write("📊 Single-sentence input; batch processing not required.")
