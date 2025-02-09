import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer, MarianMTModel

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

def flatten(nested_list):
    """Recursively flattens a nested list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def translate_text(text, model_key, max_length=50):
    """
    Translate text using the specified ONNX translation model with iterative greedy decoding.
    This version obtains decoder_start_token_id and eos_token_id from the model config.
    """
    model_path = AVAILABLE_MODELS.get(model_key)
    if model_path is None:
        print(f"❌ ERROR: No available ONNX model for {model_key}")
        return text

    print(f"🔄 Loading ONNX Model for {model_key}...")
    session = load_onnx_model(model_path)
    if session is None:
        print(f"⚠️ Skipping translation for {model_key}. Model failed to load.")
        return text

    # Construct the tokenizer name from the model key.
    tokenizer_name = f"Helsinki-NLP/opus-mt-{model_key}"
    print(f"📢 Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"❌ ERROR: Failed to load tokenizer {tokenizer_name}\n", e)
        return text

    # Load the MarianMT model (PyTorch) to extract the decoder_start_token_id and eos_token_id.
    try:
        pt_model = MarianMTModel.from_pretrained(tokenizer_name)
        decoder_start_token_id = pt_model.config.decoder_start_token_id
        eos_token_id = pt_model.config.eos_token_id
        print(f"🔍 Model config: decoder_start_token_id = {decoder_start_token_id}, eos_token_id = {eos_token_id}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model config from {tokenizer_name}\n", e)
        decoder_start_token_id = None
        eos_token_id = None

    if decoder_start_token_id is None:
        # Fallback: try bos_token_id or default to a nonzero value (e.g., 101)
        decoder_start_token_id = getattr(tokenizer, "bos_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = 101  # Use a common fallback value
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    batch_size = input_ids.shape[0]

    print(f"🔍 INPUT TEXT: {text}")
    print(f"🔍 TOKENIZED INPUT_IDS: {input_ids}")
    print(f"🔍 ATTENTION MASK: {attention_mask}")
    print(f"🔍 Using decoder_start_token_id: {decoder_start_token_id}")

    # Initialize decoder_input_ids with the decoder start token
    decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)

    # Iterative greedy decoding loop
    for i in range(max_length):
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
        ort_outputs = session.run(None, ort_inputs)
        logits = ort_outputs[0]
        # Debug: print the shape of logits at each iteration
        print(f"Iteration {i}: logits shape = {logits.shape}")
        # Get logits for the last token position
        last_token_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)
        # Greedy decoding: choose the token with highest logit
        next_token_id = np.argmax(last_token_logits, axis=-1).reshape(batch_size, 1)
        print(f"Iteration {i}: next token id = {next_token_id}")
        # Append the predicted token to decoder_input_ids
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=1)
        # Stop if EOS token is generated (if eos_token_id is defined)
        if eos_token_id is not None and next_token_id[0, 0] == eos_token_id:
            print(f"Iteration {i}: EOS token {eos_token_id} generated. Stopping.")
            break

    # Remove the initial decoder start token before decoding
    final_ids = decoder_input_ids[:, 1:]
    # Flatten token IDs for decoding
    final_ids_flat = flatten(final_ids[0])
    print(f"🔍 Final token ids: {final_ids_flat}")

    try:
        translated_text = tokenizer.decode(final_ids_flat, skip_special_tokens=True)
    except Exception as e:
        print(f"❌ ERROR during decoding: {e}")
        return text

    print(f"📝 FINAL TRANSLATION [{model_key}]: {translated_text}")
    return translated_text

# Dictionary of available ONNX models
AVAILABLE_MODELS = {
    "ja-en": "models/marian_ja-en.onnx",
    "en-zh": "models/marian_en-zh.onnx",
    "zh-en": "models/marian_zh-en.onnx",
    "ko-en": "models/marian_ko-en.onnx",
    "en-ko": "models/marian_en-ko.onnx",  
}

__all__ = ["translate_text"]

# Standalone test for translation_engine.py
if __name__ == "__main__":
    test_text = "こんにちは、これはテストです。"
    test_model = "ja-en"
    print("\n🛠 Running Standalone Translation Test...\n")
    result = translate_text(test_text, test_model)
    print(f"\n✅ TRANSLATION RESULT: {result}\n")
