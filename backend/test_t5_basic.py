from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from paths import T5_MODEL_DIR

def main():
    print("=== BASIC T5 MODEL VERIFICATION TEST ===\n")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        T5_MODEL_DIR,
        local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        T5_MODEL_DIR,
        local_files_only=True
    )

    print("Model loaded successfully ✅\n")

    # Very simple input
    text = (
        "Albert Einstein proposed the theory of relativity. "
        "It introduced concepts like time dilation and space-time."
    )

    prompt = "summarize: " + text

    print("Input text:")
    print(text)
    print("\nRunning inference...\n")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_length=60,
        min_length=20,
        num_beams=4,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("T5 Output Summary:")
    print(summary)

    print("\n=== TEST FINISHED ===")

if __name__ == "__main__":
    main()
