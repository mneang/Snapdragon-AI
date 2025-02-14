from transformers import MarianMTModel, MarianTokenizer
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ja"
print(f"🔄 Exporting {MODEL_NAME} to ONNX...")

model = MarianMTModel.from_pretrained(MODEL_NAME)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

inputs = tokenizer(["Hello, this is a test."], return_tensors="pt", padding=True, truncation=True)

onnx_path = "models/marian_en-ja.onnx"
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
)

print(f"✅ Successfully exported ONNX model: {onnx_path}")
