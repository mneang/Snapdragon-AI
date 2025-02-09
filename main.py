import onnxruntime as ort
import whisper
import torch
import numpy as np
import librosa
import time

# ================================
# 1. Load the Whisper Model for Decoding
# ================================
MODEL_SIZE = "small"  # You can change to "small" or "medium" for better accuracy (but slower)
model = whisper.load_model(MODEL_SIZE)
print("✅ Whisper model loaded ({}).".format(MODEL_SIZE))

# ================================
# 2. Load the ONNX-optimized Encoder
# ================================
MODEL_PATH = "models/whisper_optimized.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("✅ ONNX Model Loaded Successfully!")
except Exception as e:
    print("❌ ERROR: Failed to load ONNX Model.\n", e)
    exit()

# ================================
# 3. Audio Preprocessing Function
# ================================
def load_audio(audio_path):
    """
    Loads an audio file, converts it to a log mel spectrogram,
    and ensures the output has shape (1, 80, 3000).
    """
    # Load audio at 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    # Compute log-mel spectrogram (this function returns a tensor of shape (80, T))
    mel = whisper.log_mel_spectrogram(torch.tensor(audio))
    mel = mel.numpy().astype(np.float32)
    
    # Ensure time dimension is exactly 3000 frames
    target_length = 3000
    current_length = mel.shape[1]
    if current_length < target_length:
        pad_width = target_length - current_length
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel = mel[:, :target_length]
    
    # Expand dims to create batch dimension -> (1, 80, 3000)
    mel = np.expand_dims(mel, axis=0)
    print("🔍 DEBUG: Audio Input Shape (Fixed):", mel.shape)
    return mel

# ================================
# 4. Load and Preprocess the Audio File
# ================================
AUDIO_FILE = "sample_audio.wav"
try:
    mel_input = load_audio(AUDIO_FILE)
    inputs = {"audio_input": mel_input}
except Exception as e:
    print("❌ ERROR: Failed to load or preprocess audio file.\n", e)
    exit()

# ================================
# 5. Run Encoder and Prepare Output for Decoding
# ================================
try:
    # Option A: Use the ONNX encoder output (currently causing channel mismatch)
    # raw_output = session.run(None, {"audio_input": mel_input})[0]
    # print("🔍 DEBUG: Raw ONNX Output Shape:", raw_output.shape)
    # ... (your current transposition and padding logic) ...

    # Option B: Use the native encoder output instead:
    with torch.no_grad():
        native_encoder_output = model.encoder(torch.tensor(mel_input).to(model.device))
    print("🔍 DEBUG: Native Encoder Output Shape:", native_encoder_output.shape)
    
    # Use the native output for decoding
    encoder_output = native_encoder_output

    # (If needed, convert to CPU tensor; likely already on CPU)
    encoder_output = encoder_output.to("cpu")
except Exception as e:
    print("❌ ERROR: Failed during encoder inference.\n", e)
    exit()

# ================================
# 6. Select Language for Decoding
# ================================
lang_map = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "en": "English"}
lang_code = input("🌍 Enter language (ja = Japanese, ko = Korean, zh = Chinese, en = English): ").strip().lower()
if lang_code not in lang_map:
    print("❌ ERROR: Invalid language selection. Defaulting to English.")
    lang_code = "en"
print("🗣️ Selected Language:", lang_map[lang_code])

# ================================
# 7. Run Whisper Decoder with Beam Search Options (No best_of)
# ================================
try:
    options = whisper.DecodingOptions(
        language=lang_code,
        temperature=0.2,    # Use 0 for determinism, try 0.1 if needed
        beam_size=15       # Increase beam_size for better candidate exploration
    )
    start_time = time.time()
    result = whisper.decode(model, encoder_output, options)
    end_time = time.time()
    
    # Check if result is a list; if so, take the first candidate
    if isinstance(result, list):
        transcription = result[0].text
    else:
        transcription = result.text

    print("✅ FINAL AI Model Transcription Output:")
    print(transcription)
    print("⏳ Total Transcription Time: {:.2f} seconds".format(end_time - start_time))
except Exception as e:
    print("❌ ERROR: Speech-to-Text Failed during decoding.\n", e)
