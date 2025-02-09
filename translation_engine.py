import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer

# Load ONNX models dynamically based on request
def load_onnx_model(model_path):
    """
    Load an ONNX translation model from the given path.
    """
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"✅ ONNX Model Loaded: {model_path}")
        return session
    except Exception as e:
        print(f"❌ ERROR: Failed to load ONNX Model {model_path}.\n", e)
        return None

# Translation function using ONNX models
def translate_text(text, model_key):
    """
    Translate text using the specified ONNX translation model.
    """
    model_path = AVAILABLE_MODELS.get(model_key)
    
    if model_path is None:
        print(f"❌ ERROR: No available ONNX model for {model_key}")
        return text  # Return original text if no model is found

    session = load_onnx_model(model_path)
    if session is None:
        return text  # Return original text if model loading fails

    # Load the tokenizer based on model name
    tokenizer_name = f"Helsinki-NLP/opus-mt-{model_key}"
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int64)

    # Run ONNX model inference
    ort_inputs = {"input_ids": input_ids}
    ort_outputs = session.run(None, ort_inputs)

    # Decode the output tokens into text
    translated_text = tokenizer.decode(ort_outputs[0][0], skip_special_tokens=True)
    
    print(f"📝 Translated [{model_key}]: {translated_text}")
    return translated_text

# Dictionary of available ONNX models
AVAILABLE_MODELS = {
    "ja-en": "models/marian_ja-en.onnx",
    "en-zh": "models/marian_en-zh.onnx",
    "zh-en": "models/marian_zh-en.onnx",
    "ko-en": "models/marian_ko-en.onnx",
    "en-ko": "models/marian_en-ko.onnx",  
}

# Export functions for use in `translation_routing.py`
__all__ = ["translate_text"]
