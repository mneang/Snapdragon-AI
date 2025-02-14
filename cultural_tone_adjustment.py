import re

def adjust_tone(text, lang, mode="formal"):
    """
    Adjusts the tone of the input text based on language and mode.
    For Japanese:
      - Formal: Ensure the sentence ends with "です。" only if not already polite.
      - Business: Replace common verbs with their honorific forms and end with "でございます。"
    For other languages, no changes are applied.
    """
    if lang == "ja":
        # Remove any trailing whitespace or punctuation for processing.
        text = text.strip()
        
        # For formal tone:
        if mode == "formal":
            # If the text already ends with the formal suffix, leave it.
            if text.endswith("です。"):
                return text
            # If the text ends with "ます" but not followed by "です", then leave it as is.
            # We check the last few characters.
            if re.search(r'ます[\。\．]?$', text):
                return text
            # Otherwise, append the formal suffix.
            return text + "です。"
        
        # For business tone:
        elif mode == "business":
            # Define a simple mapping for common verbs (this can be expanded).
            mapping = {
                "行きます": "いらっしゃいます",
                "食べます": "召し上がります",
                "話します": "お話しになります",
                # You can add more verb mappings here.
            }
            # Replace occurrences using the mapping.
            for verb, honorific in mapping.items():
                text = re.sub(verb, honorific, text)
            # Remove any existing polite suffixes.
            text = re.sub(r'(です。|でございます。)$', '', text)
            # Append the business suffix.
            return text + "でございます。"
    
    # If not Japanese, return the text unchanged.
    return text

# Standalone test
if __name__ == "__main__":
    # Test cases for cultural tone adjustment
    sample_text = "行きます 食べます 話します"
    print("Formal (Japanese):", adjust_tone(sample_text, "ja", mode="formal"))
    print("Business (Japanese):", adjust_tone(sample_text, "ja", mode="business"))
