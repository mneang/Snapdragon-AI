import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer, MarianMTModel
import re
import torch

# ------------------------------
# Utility Functions
# ------------------------------

def load_onnx_model(model_path):
    """Loads an ONNX model from the given path using CPUExecutionProvider."""
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"✅ ONNX Model Loaded: {model_path}")
        return session
    except Exception as e:
        print(f"❌ ERROR: Failed to load ONNX Model {model_path}.\n{e}")
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

def trim_repeating_pattern(tokens, pattern_length=2, min_repeats=3):
    """
    Trims a repeating pattern at the end of a token list.
    For example, if the last 2 tokens repeat at least min_repeats times, they are trimmed.
    """
    if len(tokens) < pattern_length * min_repeats:
        return tokens
    pattern = tokens[-pattern_length:]
    count = 0
    i = len(tokens) - pattern_length
    while i >= 0 and tokens[i:i+pattern_length] == pattern:
        count += 1
        i -= pattern_length
    if count >= min_repeats:
        return tokens[:i+pattern_length]
    return tokens

def postprocess_translation(text):
    """
    Postprocesses the decoded text by reducing excessive trailing punctuation.
    For example, it converts multiple periods at the end to a single period.
    """
    text = re.sub(r'([\.\!\?]){2,}$', r'\1', text)
    return text

def deduplicate_similar_sentences(text, threshold=0.6):
    """
    Splits the text into sentences and removes any sentence that is highly similar 
    (based on Jaccard similarity) to any previously kept sentence.
    This helps remove repeated tail fragments in the Japanese translation.
    """
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    kept = []
    for s in sentences:
        # Only consider non-empty sentences with more than 3 words.
        if len(s.split()) < 4:
            continue
        s_words = set(s.lower().split())
        duplicate = False
        for prev in kept:
            prev_words = set(prev.lower().split())
            if prev_words:
                jaccard = len(s_words.intersection(prev_words)) / len(s_words.union(prev_words))
                if jaccard >= threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(s)
    return " ".join(kept)

# ------------------------------
# Beam Search Decoding for Chinese Translations
# ------------------------------

def beam_search_decode_force(session, input_ids, attention_mask, decoder_start_token_id, eos_token_id,
                             beam_width=14, forced_length=120, length_penalty=0.75, repetition_penalty=1.03):
    """
    Performs forced beam search decoding for Chinese translations,
    forcing generation of at least 'forced_length' tokens.
    """
    beams = [([decoder_start_token_id], 0.0)]
    for t in range(forced_length):
        new_beams = []
        for seq, score in beams:
            dec_input = np.array([seq], dtype=np.int64)
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": dec_input,
            }
            logits = session.run(None, ort_inputs)[0]
            last_logits = logits[0, -1, :]
            log_probs = np.log(np.maximum(np.exp(last_logits - np.max(last_logits)) /
                                          np.sum(np.exp(last_logits - np.max(last_logits))), 1e-12))
            for token in set(seq):
                count = seq.count(token)
                log_probs[token] /= (repetition_penalty ** count)
            top_indices = np.argsort(log_probs)[-beam_width:]
            for token in top_indices:
                new_seq = seq + [int(token)]
                lp = ((5 + len(new_seq)) / 6) ** length_penalty
                new_score = (score + log_probs[token]) / lp
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    best_seq = sorted(beams, key=lambda x: x[1], reverse=True)[0][0]
    return best_seq

def beam_search_decode(session, input_ids, attention_mask, decoder_start_token_id, eos_token_id,
                       beam_width=14, max_length=300, length_penalty=0.75, repetition_penalty=1.03, min_length=120):
    """
    Performs standard beam search decoding with a minimum length constraint for Chinese translations.
    """
    beams = [([decoder_start_token_id], 0.0)]
    finished = []
    for t in range(max_length):
        new_beams = []
        for seq, score in beams:
            if t >= min_length and eos_token_id is not None and seq[-1] == eos_token_id:
                finished.append((seq, score))
                continue
            dec_input = np.array([seq], dtype=np.int64)
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": dec_input,
            }
            logits = session.run(None, ort_inputs)[0]
            last_logits = logits[0, -1, :]
            log_probs = np.log(np.maximum(np.exp(last_logits - np.max(last_logits)) /
                                          np.sum(np.exp(last_logits - np.max(last_logits))), 1e-12))
            for token in set(seq):
                count = seq.count(token)
                log_probs[token] /= (repetition_penalty ** count)
            top_indices = np.argsort(log_probs)[-beam_width:]
            for token in top_indices:
                new_seq = seq + [int(token)]
                lp = ((5 + len(new_seq)) / 6) ** length_penalty
                new_score = (score + log_probs[token]) / lp
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if eos_token_id is not None and all(seq[-1] == eos_token_id for seq, _ in beams):
            finished.extend(beams)
            break
    best_seq = sorted(finished, key=lambda x: x[1], reverse=True)[0][0] if finished else beams[0][0]
    return best_seq

# ------------------------------
# Fallback Translation using MarianMTModel.generate()
# ------------------------------

def remove_duplicate_adjacent(text):
    """
    Removes duplicate adjacent words and duplicate phrases using regex.
    For example, "complete and complete translation" becomes "complete and translation".
    """
    words = text.split()
    if not words:
        return text
    new_words = [words[0]]
    for word in words[1:]:
        if word != new_words[-1]:
            new_words.append(word)
    result = " ".join(new_words)
    result = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', result)
    return result

def fallback_translation(text, model_name):
    """
    Uses MarianMTModel.generate() as a fallback for translation.
    Applies sampling and post-processing to remove duplicate words/phrases.
    """
    print(f"🔄 Fallback: Using MarianMTModel generate() for {model_name}.")
    pt_model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{model_name}")
    tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{model_name}")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = pt_model.generate(**inputs, max_length=300, do_sample=True, top_p=0.95, temperature=0.8)
    fallback_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    fallback_text = remove_duplicate_adjacent(fallback_text)
    fallback_text = postprocess_translation(fallback_text)
    return fallback_text

# ------------------------------
# Input Augmentation for Chinese
# ------------------------------

def augment_input(text, model_name):
    """Augments the input text for Chinese translations if too short."""
    if model_name in ["en-zh", "zh-en"]:
        if len(text.strip()) < 30:
            return text.strip() + " Please provide a detailed and complete translation."
    return text

# ------------------------------
# Validity Checks for Output
# ------------------------------

def is_valid_chinese(text, threshold=0.9):
    """Returns True if a high fraction of characters in text are Chinese."""
    cleaned = re.sub(r"[^\u4e00-\ud7a3]", "", text)
    if not text.strip():
        return False
    return len(cleaned) / len(text.strip()) >= threshold

def is_valid_english(text, threshold=0.7):
    """Returns True if a high fraction of characters in text are alphabetic."""
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return False
    alpha_count = sum(1 for c in cleaned if c.isalpha())
    return (alpha_count / len(cleaned)) >= threshold

def postprocess_translation(text):
    """
    Postprocesses the decoded text by reducing excessive trailing punctuation.
    For example, it converts multiple periods at the end to a single period.
    """
    text = re.sub(r'([\.\!\?]){2,}$', r'\1', text)
    return text

def remove_repeated_tail(text):
    """
    Splits the text into sentences and removes any final sentence that is highly similar
    to earlier content. This helps remove repeated tail fragments.
    """
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    if not sentences:
        return text
    kept = sentences[:-1]  # keep all but the last sentence initially
    last = sentences[-1]
    # Calculate Jaccard similarity between last and each kept sentence.
    last_words = set(last.lower().split())
    for s in kept:
        s_words = set(s.lower().split())
        if s_words:
            similarity = len(last_words & s_words) / len(last_words | s_words)
            if similarity >= 0.6:  # threshold (tweak as needed)
                # Remove the last sentence if very similar.
                return " ".join(kept)
    return text

# ------------------------------
# Alternative Fallback for English→Korean Using M2M100
# ------------------------------

def fallback_en_to_ko_m2m100(text):
    """
    Uses Facebook's M2M100 model for direct English→Korean translation.
    This serves as our best alternative for a direct en-ko model.
    """
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        print("🔄 Fallback: Using M2M100 for direct English→Korean translation.")
        m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        m2m_tokenizer.src_lang = "en"
        encoded = m2m_tokenizer(text, return_tensors="pt")
        generated_tokens = m2m_model.generate(
            **encoded,
            forced_bos_token_id=m2m_tokenizer.get_lang_id("ko"),
            max_length=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        result = m2m_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return result
    except Exception as e:
        print(f"⚠️ Fallback M2M100 translation failed: {e}")
        return "[Direct English→Korean model missing]"

# ------------------------------
# Reverse Translation Fallback for English→Korean
# ------------------------------

def contains_hangul(text):
    """Checks if text contains any Hangul characters."""
    return bool(re.search(r'[\uac00-\ud7a3]', text))

def fallback_en_to_ko(text):
    """
    Uses a direct en-ko translation via M2M100 if available.
    If that fails, falls back to using MarianMTModel generate() with the reverse (ko-en) model.
    Checks that the output contains Hangul.
    """
    result = fallback_en_to_ko_m2m100(text)
    if contains_hangul(result):
        return result
    else:
        print("⚠️ Direct en-ko translation via M2M100 did not produce Korean characters.")
        print("🔄 Fallback: Using MarianMTModel generate() for en→ko translation via reverse fallback.")
        pt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = pt_model.generate(
            **inputs,
            max_length=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        fallback_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not contains_hangul(fallback_text):
            print("⚠️ Fallback output does not contain Korean characters. Direct en-ko model required.")
            return "[Direct English→Korean model missing]"
        return fallback_text

# ------------------------------
# Main Translation Function
# ------------------------------

def translate_text(text, model_name, max_length=50):
    """
    Translates text using the specified ONNX model.
    For Chinese translations (en-zh, zh-en), uses forced beam search decoding with refined parameters.
    For reverse modes (e.g., 'ko-en-reverse'), uses our alternative fallback (M2M100 first, then MarianMTModel generate() via reverse fallback).
    For Japanese→English (ja-en), uses greedy decoding and then applies post‑processing to trim repeated tail fragments.
    For other directions, uses greedy decoding with trailing eos tokens trimmed.
    """
    # Augment input if needed.
    text = augment_input(text, model_name)
    
    # Handle reverse mode for English→Korean.
    if model_name == "ko-en-reverse":
        print("🔄 Reverse mode detected for English→Korean translation. Using fallback translator.")
        result = fallback_en_to_ko(text)
        print(f"📝 FALLBACK TRANSLATION [ko-en-reverse]: {result}")
        return result

    model_path = f"models/marian_{model_name}.onnx"
    print(f"🔄 Loading ONNX Model for {model_name}...")
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"❌ ERROR: Failed to load ONNX model {model_name}.\n{e}")
        return text

    tokenizer_name = f"Helsinki-NLP/opus-mt-{model_name}"
    print(f"📢 Loading tokenizer: {tokenizer_name}...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"❌ ERROR: Failed to load tokenizer {tokenizer_name}.\n{e}")
        return text

    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    batch_size = input_ids.shape[0]

    try:
        pt_model = MarianMTModel.from_pretrained(tokenizer_name)
        decoder_start_token_id = pt_model.config.decoder_start_token_id
        eos_token_id = pt_model.config.eos_token_id
        print(f"🔍 Model config: decoder_start_token_id = {decoder_start_token_id}, eos_token_id = {eos_token_id}")
    except Exception as e:
        print(f"❌ ERROR: Could not load model config from {tokenizer_name}: {e}")
        decoder_start_token_id = getattr(tokenizer, "bos_token_id", 101)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(tokenizer, "bos_token_id", 101)
    print(f"🔍 Using decoder_start_token_id: {decoder_start_token_id}")
    
    # For Chinese models, if eos_token_id is 0, override to None.
    if model_name in ["en-zh", "zh-en"] and eos_token_id == 0:
        print("🔍 Warning: eos_token_id is 0 for Chinese model; overriding to None.")
        eos_token_id = None

    # Branch based on model_name:
    if model_name == "ja-en":
        # For Japanese→English, use greedy decoding.
        decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)
        for i in range(max_length):
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
            }
            ort_outputs = session.run(None, ort_inputs)
            logits = ort_outputs[0]
            next_token = np.argmax(logits[:, -1, :], axis=-1).reshape(batch_size, 1)
            decoder_input_ids = np.concatenate([decoder_input_ids, next_token], axis=1)
        final_ids = decoder_input_ids[:, 1:]
        final_ids_list = flatten(final_ids[0])
        # Remove trailing eos tokens (assumed to be 0) and trim repeating patterns.
        while final_ids_list and final_ids_list[-1] == 0:
            final_ids_list.pop()
        final_ids_list = trim_repeating_pattern(final_ids_list, pattern_length=2, min_repeats=3)
        final_ids_flat = final_ids_list
    elif model_name in ["en-zh", "zh-en"]:
        # For Chinese translations, use forced beam search decoding.
        if eos_token_id is None:
            print("🔍 Warning: eos_token_id is None for Chinese model; using fallback translation.")
            return fallback_translation(text, model_name)
        beam_width = 14
        forced_length = 120
        max_dec_length = 300
        length_penalty = 0.75
        repetition_penalty = 1.03
        print("🔄 Using forced beam search decoding for Chinese translation.")
        best_seq = beam_search_decode_force(session, input_ids, attention_mask,
                                            decoder_start_token_id, eos_token_id,
                                            beam_width=beam_width, forced_length=forced_length,
                                            length_penalty=length_penalty, repetition_penalty=repetition_penalty)
        if len(best_seq) < forced_length or len(flatten(best_seq[1:])) < 10:
            print("🔍 Forced beam search output too short; using fallback translation.")
            return fallback_translation(text, model_name)
        final_ids_flat = best_seq[1:]  # Remove the initial token.
    else:
        # For other directions, use greedy decoding.
        decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)
        for i in range(max_length):
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
            }
            ort_outputs = session.run(None, ort_inputs)
            logits = ort_outputs[0]
            next_token = np.argmax(logits[:, -1, :], axis=-1).reshape(batch_size, 1)
            decoder_input_ids = np.concatenate([decoder_input_ids, next_token], axis=1)
        final_ids = decoder_input_ids[:, 1:]
        final_ids_list = flatten(final_ids[0])
        if model_name == "ja-en":
            while final_ids_list and final_ids_list[-1] == 0:
                final_ids_list.pop()
            final_ids_list = trim_repeating_pattern(final_ids_list, pattern_length=2, min_repeats=3)
        final_ids_flat = final_ids_list

    print(f"🔍 Final token ids: {final_ids_flat}")
    try:
        translated_text = tokenizer.decode(final_ids_flat, skip_special_tokens=True)
    except Exception as e:
        print(f"❌ ERROR during decoding: {e}")
        return text

    translated_text = postprocess_translation(translated_text)
    if model_name == "ja-en":
        # Further remove repeated tail fragments.
        translated_text = remove_repeated_tail(translated_text)
        translated_text = deduplicate_similar_sentences(translated_text)
    print(f"📝 FINAL TRANSLATION [{model_name}]: {translated_text}")

    # Validity checks.
    if model_name == "en-zh":
        if len(translated_text.split()) < 10 or not is_valid_chinese(translated_text, threshold=0.9):
            print("⚠️ Output appears too short or not valid Chinese. Using fallback translation.")
            translated_text = fallback_translation(text, model_name)
            print(f"📝 FALLBACK TRANSLATION [{model_name}]: {translated_text}")
    if model_name == "zh-en":
        if len(translated_text.split()) < 10 or not is_valid_english(translated_text, threshold=0.7):
            print("⚠️ Output appears too short or not valid English. Using fallback translation.")
            translated_text = fallback_translation(text, model_name)
            print(f"📝 FALLBACK TRANSLATION [{model_name}]: {translated_text}")

    return translated_text

def deduplicate_similar_sentences(text, threshold=0.6):
    """
    Splits the text into sentences and removes any sentence that is highly similar
    to any previously seen sentence (using Jaccard similarity).
    """
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    kept = []
    for s in sentences:
        if len(s.split()) < 4:
            continue
        s_words = set(s.lower().split())
        duplicate = False
        for prev in kept:
            prev_words = set(prev.lower().split())
            if prev_words:
                jaccard = len(s_words & prev_words) / len(s_words | prev_words)
                if jaccard >= threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(s)
    return " ".join(kept)

# ------------------------------
# Export
# ------------------------------
__all__ = ["translate_text"]

# ------------------------------
# Standalone Test
# ------------------------------
if __name__ == "__main__":
    test_cases = [
        # Extended Japanese test case to capture full nuance.
        ("こんにちは、これはテストです。本日もよろしくお願いいたします。皆様と素晴らしい一日を過ごせることを楽しみにしております。どうぞよろしくお願いいたします。", "ja-en"),
        (
            "Hello, this is a test of our complete translation system designed for multinational business meetings. "
            "Our system is capable of handling complex, nuanced language and providing culturally appropriate translations that are essential for effective international communication.",
            "en-zh"
        ),
        (
            "你好，这是对我们为多国商务会议设计的完整翻译系统的测试。该系统能够处理复杂而细致的语言，并提供符合文化规范的翻译，对于有效的国际交流至关重要。",
            "zh-en"
        ),
        ("Hello, this is a test of our multinational business communication system.", "ko-en-reverse"),
    ]
    for text, model in test_cases:
        print("\n🔍 INPUT TEXT:", text)
        result = translate_text(text, model)
        print(f"\n✅ TRANSLATION RESULT ({model}): {result}\n")
