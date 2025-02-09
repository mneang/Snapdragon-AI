from transformers import MarianMTModel, MarianTokenizer
import torch
import onnx

# ----- Configuration -----
# For Japanese -> English translation; change the model name for other language pairs:
MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"  
onnx_path = f"models/marian_{MODEL_NAME.split('/')[-1]}.onnx"

# ----- Load Model & Tokenizer -----
print("Loading MarianMT model and tokenizer...")
model = MarianMTModel.from_pretrained(MODEL_NAME)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

# ----- Create Dummy Inputs -----
# Prepare a dummy text for the purpose of creating dummy input tokens.
dummy_text = ["こんにちは、これは日本語の翻訳テストです。"]
tokens = tokenizer(dummy_text, return_tensors="pt")

# Encoder inputs:
dummy_input_ids = tokens.input_ids        # Shape: [batch_size, sequence_length]
dummy_attention_mask = tokens.attention_mask

# For the decoder, we need to provide decoder_input_ids.
# Typically, this is set to the model's decoder_start_token_id.
# Here, we create a minimal dummy decoder input.
dummy_decoder_input_ids = torch.tensor(
    [[model.config.decoder_start_token_id]], dtype=torch.long
)  # Shape: [1, 1]

# ----- Export the Model to ONNX -----
print("Exporting model to ONNX...")
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, dummy_decoder_input_ids),
    onnx_path,
    opset_version=16,  # Use opset 16 for support of required operators.
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence"},
        "output": {0: "batch_size", 1: "decoder_sequence"}
    }
)

print(f"✅ Translation model successfully exported to {onnx_path}")
