from transformers import MarianMTModel, MarianTokenizer
import torch
import os

# Define the language pairs we want to export.
# Note: Only include model identifiers that are available on Hugging Face.
LANGUAGE_PAIRS = {
    "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    # "en-ja": "Helsinki-NLP/opus-mt-en-ja",  # This model seems unavailable.
    "ja-ko": "Helsinki-NLP/opus-mt-ja-ko",
    "ko-ja": "Helsinki-NLP/opus-mt-ko-ja",
    "ja-zh": "Helsinki-NLP/opus-mt-ja-zh",
    "zh-ja": "Helsinki-NLP/opus-mt-zh-ja",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-ko": "Helsinki-NLP/opus-mt-en-ko",
    "ko-en": "Helsinki-NLP/opus-mt-ko-en",
    "ko-zh": "Helsinki-NLP/opus-mt-ko-zh",
    "zh-ko": "Helsinki-NLP/opus-mt-zh-ko",
}

# Create models folder if it doesn't exist.
os.makedirs("models", exist_ok=True)

for pair, model_name in LANGUAGE_PAIRS.items():
    try:
        print(f"🔄 Exporting {model_name} to ONNX...")
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Use a dummy text (using a Japanese test sentence for consistency)
        dummy_text = ["こんにちは、これは日本語の翻訳テストです。"]
        tokens = tokenizer(dummy_text, return_tensors="pt")

        dummy_input_ids = tokens.input_ids
        dummy_attention_mask = tokens.attention_mask
        dummy_decoder_input_ids = torch.tensor(
            [[model.config.decoder_start_token_id]], dtype=torch.long
        )

        onnx_path = f"models/marian_{pair}.onnx"
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_decoder_input_ids),
            onnx_path,
            opset_version=16,
            input_names=["input_ids", "attention_mask", "decoder_input_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence"},
                "output": {0: "batch_size", 1: "decoder_sequence"},
            },
        )
        print(f"✅ Exported {model_name} → {onnx_path}")
    except Exception as e:
        print(f"⚠️ Skipping {model_name} due to an error: {e}")

print("🎯 Export process completed.")